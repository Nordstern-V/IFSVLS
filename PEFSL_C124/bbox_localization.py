"""Boundary proposals for bright objects; callers must classify every proposal."""
import cv2
import numpy as np


def bright_body_contour(image):
    """Describe one connected neutral body, not the union of scene edges.

    Work at a fixed size, as saved registration thumbnails have already been
    resized. A relative brightness threshold tolerates moderate lighting changes.
    Keep the contour's concavities; a convex hull would erase object structure.
    """
    if image is None or image.size == 0:
        return None, 0.
    small = cv2.resize(image, (96, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    threshold = max(145., .82 * float(np.percentile(hsv[..., 2], 90)))
    mask = ((hsv[..., 1] < 20) & (hsv[..., 2] > threshold)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None, 0.
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = map(int, stats[selected])
    fraction = area / float(mask.size)
    if min(w, h) < 6 or fraction < .04:
        return None, fraction
    contours, _ = cv2.findContours((labels == selected).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    if contour is not None and cv2.contourArea(contour) < 24:
        contour = None
    return contour, fraction


def endpoint_boundary_support(frame,box):
    """Evidence for both ends of a bounded object, rather than an open seam.

    Perpendicular gradients near the two long-axis endpoints must span part
    of its width. This is supporting evidence, not a full segmentation mask.
    A real image border is recorded as an unknown endpoint, not a closed edge.
    """
    x1,y1,x2,y2=map(int,box);w,h=x2-x1,y2-y1
    if min(w,h)<12:return (0.,0.),False
    horizontal=w>=h;short=min(w,h);margin=max(2,int(.15*short))
    left,top=max(0,x1-margin),max(0,y1-margin)
    right,bottom=min(frame.shape[1],x2+margin+1),min(frame.shape[0],y2+margin+1)
    roi=frame[top:bottom,left:right]
    if min(roi.shape[:2],default=0)<3:return (0.,0.),False
    gray=cv2.GaussianBlur(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),(3,3),0)
    gx=np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0));gy=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1))
    along,across=(gx,gy) if horizontal else (gy,gx)
    threshold=max(18.,.3*float(np.percentile(np.maximum(gx,gy),90)))
    edge=(along>=threshold)&(along>=1.2*across)
    supports=[];visible=[]
    for end in ((x1,x2) if horizontal else (y1,y2)):
        if horizontal:
            crop=edge[max(0,y1-top):y2-top,max(0,end-left-margin):end-left+margin+1]
            support=np.any(crop,axis=1)
        else:
            crop=edge[max(0,end-top-margin):end-top+margin+1,max(0,x1-left):x2-left]
            support=np.any(crop,axis=0)
        value=float(support.mean()) if support.size else 0.
        supports.append(value)
        limit=frame.shape[1] if horizontal else frame.shape[0]
        if 2<end<limit-2:visible.append(value)
    good=bool(visible) and max(visible)>=.30 and min(visible)>=.12
    return tuple(supports),good


def transparent_extension_density(frame, box):
    """Visible neutral edge support in a proposed extension, excluding skin."""
    x1, y1, x2, y2 = map(int, box)
    image = frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
    if image.size == 0 or min(image.shape[:2]) < 4:
        return 0.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.dilate(cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),30,90),
                       np.ones((3,3),np.uint8)) > 0
    skin = (hsv[...,0]<=30) & (hsv[...,1]>=12) & (hsv[...,1]<=180) & (hsv[...,2]>=50)
    neutral = (hsv[...,1]<35) & (hsv[...,2]>100) & ~skin
    return float((edges & neutral).mean())


def competing_shell_evidence(frame, first, second):
    """Compare only the disjoint sides of two opposing shell proposals.

    Dense edges on a dark patterned shirt can beat the real clear shell over
    a hand. Count dark pixels separately on each extension, rather than letting
    the common white body dilute that evidence. This is ranking evidence only;
    both proposals still need the caller's learned identity and normal gates.
    """
    a,b=tuple(first),tuple(second)
    if a[3]-a[1]>a[2]-a[0]:
        return competing_shell_evidence(cv2.transpose(frame),
            (a[1],a[0],a[3],a[2]),(b[1],b[0],b[3],b[2]))
    height=min(a[3]-a[1],b[3]-b[1])
    if height<16 or max(abs(a[1]-b[1]),abs(a[3]-b[3]))>.12*height:
        return None
    left,right=max(a[0],b[0]),min(a[2],b[2])
    if right-left<.5*min(a[2]-a[0],b[2]-b[0]):
        return None
    if b[0]<a[0]-4 and b[2]<a[2]-4:
        extra_b=(b[0],max(a[1],b[1]),left,min(a[3],b[3]))
        extra_a=(right,max(a[1],b[1]),a[2],min(a[3],b[3]))
    elif b[0]>a[0]+4 and b[2]>a[2]+4:
        extra_b=(right,max(a[1],b[1]),b[2],min(a[3],b[3]))
        extra_a=(a[0],max(a[1],b[1]),left,min(a[3],b[3]))
    else:
        return None
    if min(extra_a[2]-extra_a[0],extra_b[2]-extra_b[0])<max(6,.08*height):
        return None
    def dark_fraction(box):
        x1,y1,x2,y2=box
        image=frame[y1:y2,x1:x2]
        if image.size==0:return 1.
        return float((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<90).mean())
    return dict(first_dark=dark_fraction(extra_a),second_dark=dark_fraction(extra_b),
                second_edges=transparent_extension_density(frame,extra_b))


def measured_tapered_boundary(frame, box, aspect):
    """Validate a narrow transparent tip using measured paired shell rails.

    A tapered tip need not span a fixed fraction of the body's height. This
    alternative still needs a connected opaque anchor and measured converging
    rails ending at the proposed box; it never supplies an extrapolated box.
    The caller must independently verify learned identity and classification.
    """
    x1,y1,x2,y2=map(int,box)
    if min(x2-x1,y2-y1)<24:
        return False
    if y2-y1>x2-x1:
        return measured_tapered_boundary(cv2.transpose(frame),(y1,x1,y2,x2),aspect)
    contour,fraction=bright_body_contour(frame[y1:y2,x1:x2])
    if contour is None or fraction<.12:
        return False
    bx,by,bw,bh=cv2.boundingRect(contour)
    # bright_body_contour reports coordinates in its fixed 96 x 64 image.
    w,h=x2-x1,y2-y1
    if bh/64.<.55 or not .40<=bw/96.<=.85:
        return False
    padding=max(2,int(.03*w))
    anchor=(max(x1,x1+int(bx*w/96)-padding),y1,
            min(x2,x1+int(np.ceil((bx+bw)*w/96))+padding),y2)
    return shell_extension_support(frame,anchor,box,aspect)>0.


def _tapered_shell_candidates(frame, anchor, aspect, neutral_reflections=False):
    """Measure two converging transparent rails attached to an opaque body.

    Thin triangular tips cannot occupy fixed top/bottom bands. Paired measured
    line segments allow tapering, without extrapolating a tip from aspect alone.
    """
    x1,y1,x2,y2=map(int,anchor);height=y2-y1;width=x2-x1
    if height<16 or width<height or not aspect or aspect<1.4:
        return []
    reach=min(int(.75*width),max(0,int(height*aspect*1.1)-width))
    if reach<8:return []
    scale=min(1.,240./max(width+2*reach,height))
    output=[]
    for direction in (-1,1):
        left=max(0,x1-reach) if direction<0 else max(x1,x2-int(.15*height))
        right=min(x2,x1+int(.15*height)) if direction<0 else min(frame.shape[1],x2+reach)
        image=frame[y1:y2,left:right]
        if image.size==0:continue
        image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        edge=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),30,90)
        # Neutral reflections on clear plastic can share the hue of the hand
        # behind it. The optional proposal path keeps only low-saturation
        # edges, then still demands TWO measured converging rails.
        skin_min=65 if neutral_reflections else 12
        skin=(hsv[...,0]<=30)&(hsv[...,1]>=skin_min)&(hsv[...,1]<=180)&(hsv[...,2]>=50)
        edge[skin|(hsv[...,1]>=(65 if neutral_reflections else 55))|(hsv[...,2]<65)]=0
        sh=height*scale
        lines=cv2.HoughLinesP(edge,1,np.pi/180,threshold=max(8,int(.15*sh)),
            minLineLength=max(8,int(.28*sh)),maxLineGap=max(2,int(.07*sh)))
        if lines is None:continue
        rails=[]
        attach=(x1-left)*scale if direction<0 else (x2-left)*scale
        for a,b,c,d in lines.reshape(-1,4)[:80]:
            if a>c:a,b,c,d=c,d,a,b
            if c-a<max(6,.18*sh):continue
            slope=(d-b)/float(c-a)
            if not .10<=abs(slope)<=1.5:continue
            near=c if direction<0 else a
            if abs(near-attach)>.25*sh:continue
            tip=a if direction<0 else c
            if direction*(tip-attach)<.15*sh:continue
            rails.append((float(a),float(b),float(c),float(d),slope))
        for upper in rails:
            for lower in rails:
                # Moving away from the body, the upper/lower rails converge.
                if upper[4]*direction<.10 or lower[4]*direction>-.10:continue
                tip=max(upper[0],lower[0]) if direction<0 else min(upper[2],lower[2])
                if neutral_reflections:
                    # Partial reflections may break a rail near the apex. Its
                    # intersection must remain close to both measured segments
                    # and have an actual current edge pixel at that location.
                    tip=(lower[1]-lower[4]*lower[0]-upper[1]+upper[4]*upper[0])/(upper[4]-lower[4])
                    ends=(upper[0],lower[0]) if direction<0 else (upper[2],lower[2])
                    if max(abs(tip-end) for end in ends)>.35*sh:continue
                    ty=upper[1]+upper[4]*(tip-upper[0])
                    ix,iy=int(round(tip)),int(round(ty));radius=max(2,int(.05*sh))
                    if not (0<=ix<edge.shape[1] and 0<=iy<edge.shape[0]):continue
                    cap=edge[max(0,iy-radius):iy+radius+1,max(0,ix-radius):ix+radius+1]
                    if not np.any(cap):continue
                ua=upper[1]+upper[4]*(attach-upper[0])
                la=lower[1]+lower[4]*(attach-lower[0])
                ut=upper[1]+upper[4]*(tip-upper[0])
                lt=lower[1]+lower[4]*(tip-lower[0])
                if not .45*sh<=la-ua<=1.10*sh or not 0.<=lt-ut<=.35*sh:continue
                if abs((ut+lt)/2.-sh/2.)>.30*sh:continue
                if direction*(tip-attach)<.18*sh:continue
                end=int(round(left+tip/scale))
                box=(max(0,end-3),y1,x2,y2) if direction<0 else (x1,y1,min(frame.shape[1],end+3),y2)
                quality=float(min(1.,(la-ua)/sh)*(1.-(lt-ut)/sh))
                output.append((quality,box))
    return sorted(output,reverse=True)[:4]


def shell_extension_support(frame, anchor, candidate, aspect=None):
    """Require connected side rails and a visible cap beyond an opaque anchor.

    A bright hand or isolated background edge cannot establish a shell. The
    learned aspect only bounds proposals; actual current pixels support them.
    Return zero for a translation, shrink or cross-axis expansion.
    """
    ax1, ay1, ax2, ay2 = map(int, anchor)
    x1, y1, x2, y2 = map(int, candidate)
    horizontal = ax2-ax1 >= ay2-ay1
    if not horizontal:
        return shell_extension_support(cv2.transpose(frame),
            (ay1, ax1, ay2, ax2), (y1, x1, y2, x2),aspect)
    height = ay2-ay1
    if height < 12 or x1 > ax1 or x2 < ax2 \
            or abs(y1-ay1) > .08*height or abs(y2-ay2) > .08*height:
        return 0.
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return 0.
    if x2-x1 > 1.75*(ax2-ax1):
        return 0.
    if aspect:
        for support, measured in _tapered_shell_candidates(frame,anchor,aspect):
            if max(abs(a-b) for a,b in zip(measured,candidate))<=max(4,int(.04*height)):
                return support
        return 0.
    extensions = [(x1, ax1), (ax2, x2)]
    values = []
    for left, right in extensions:
        if right-left < max(4, int(.04*height)):
            continue
        image = frame[y1:y2, left:right]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((3,3), np.uint8)) > 0
        skin = (hsv[...,0]<=30) & (hsv[...,1]>=12) & (hsv[...,1]<=180) & (hsv[...,2]>=50)
        edges &= (hsv[...,1]<45) & (hsv[...,2]>80) & ~skin
        smooth=cv2.GaussianBlur(gray,(3,3),0)
        gx=np.abs(cv2.Sobel(smooth,cv2.CV_32F,1,0))
        gy=np.abs(cv2.Sobel(smooth,cv2.CV_32F,0,1))
        rails=edges & (gy>=.8*gx)
        band = max(2, int(.30*image.shape[0]))
        top = np.any(rails[:band], axis=0)
        bottom = np.any(rails[-band:], axis=0)
        supported = top | bottom
        # One long rail plus support on the opposite side tolerates reflections
        # while rejecting a single seam or an isolated bright end pixel.
        if supported.mean() < .60 or min(top.mean(), bottom.mean()) < .20:
            return 0.
        gaps = np.flatnonzero(np.diff(np.r_[True, supported, True].astype(np.int8)))
        longest_gap = max((b-a for a,b in zip(gaps[::2],gaps[1::2])), default=0)
        if longest_gap > max(3, int(.25*height)):
            return 0.
        cap, _ = endpoint_boundary_support(frame, candidate)
        endpoint = cap[0] if left == x1 else cap[1]
        # An image border is visibility truncation, never a complete shell cap.
        end = x1 if left == x1 else x2
        if not 2 < end < frame.shape[1]-2 or endpoint < .20:
            return 0.
        cap_width=max(2,min(image.shape[1],int(.12*height)))
        cap_pixels=edges & (gx>=1.2*gy) & (gx>=18.)
        cap_pixels=cap_pixels[:,:cap_width] if left==x1 else cap_pixels[:,-cap_width:]
        if float(np.any(cap_pixels,axis=1).mean())<.20:
            return 0.
        values.append(float(min(top.mean(), bottom.mean(), endpoint)))
    return min(values) if values else 0.


def anchored_shell_candidates(frame, anchor, aspect):
    """Only measured, bounded shell extensions; caller must verify identity."""
    x1,y1,x2,y2=anchor
    if y2-y1>x2-x1:
        transposed=anchored_shell_candidates(cv2.transpose(frame),(y1,x1,y2,x2),
            max(aspect,1./aspect) if aspect and aspect>0 else aspect)
        return [(b[1],b[0],b[3],b[2]) for b in transposed]
    tapered=[box for _,box in _tapered_shell_candidates(frame,anchor,aspect)]
    return list(dict.fromkeys(tapered))[:4]


def reflected_shell_candidates(frame, anchor, aspect):
    """Additional measured proposals; callers must reclassify the full shell."""
    x1,y1,x2,y2=anchor
    if y2-y1>x2-x1:
        return [(b,a,d,c) for a,b,c,d in reflected_shell_candidates(
            cv2.transpose(frame),(y1,x1,y2,x2),aspect)]
    candidates=list(dict.fromkeys(box for quality,box in
        _tapered_shell_candidates(frame,anchor,aspect,neutral_reflections=True)
        if quality>=.55))[:2]
    for box in _transparent_candidates(frame,anchor,aspect,neutral_reflections=True):
        if box!=tuple(anchor) and box[2]-box[0]<=1.75*(x2-x1) and box not in candidates:
            candidates.append(box)
    return candidates[:4]


def measured_narrow_tip(frame, box, aspect):
    """Resolve a narrow cap only with current paired rails near that endpoint."""
    x1,y1,x2,y2=map(int,box);w,h=x2-x1,y2-y1
    if min(w,h)<24:return False
    if h>w:
        return measured_narrow_tip(cv2.transpose(frame),(y1,x1,y2,x2),aspect)
    support,complete=endpoint_boundary_support(frame,box)
    if complete or max(support)<.30 or not 0.<min(support)<.12:
        return False
    left_tip=support[0]<support[1]
    anchor=(x1+int(.35*w),y1,x2,y2) if left_tip else (x1,y1,x2-int(.35*w),y2)
    tolerance=max(4,int(.15*h))
    for quality,candidate in _tapered_shell_candidates(frame,anchor,aspect,neutral_reflections=True):
        if quality<.55:continue
        if max(abs(a-b) for a,b in zip(candidate,box))<=tolerance:
            return True
    return False


def _transparent_candidates(frame, fitted, aspect, neutral_reflections=False):
    """Trace visible pale edges on each side, without filling an inferred tail."""
    x1, y1, x2, y2 = fitted
    height, width = y2-y1, x2-x1
    if aspect is None or aspect <= 1.4 or width/height >= .90*aspect:
        return [fitted]
    # The learned ratio bounds the search. Actual connected edge pixels provide
    # the returned endpoints; the ratio never creates a foreground rectangle.
    remaining = max(0, int(height*aspect*1.15)-width)
    if remaining < 4:
        return [fitted]
    left, right = max(0,x1-remaining), min(frame.shape[1],x2+remaining)
    band = frame[y1:y2,left:right]
    hsv = cv2.cvtColor(band,cv2.COLOR_BGR2HSV)
    edge = cv2.Canny(cv2.cvtColor(band,cv2.COLOR_BGR2GRAY),30,90)
    edge = cv2.dilate(edge,np.ones((3,3),np.uint8))>0
    pale_edges = edge & (hsv[...,1]<(65 if neutral_reflections else 35)) & (hsv[...,2]>100)
    # Low saturation skin and glasses can connect the opposite side of a
    # white body to a face. They do not establish a transparent extension.
    skin = (hsv[...,0]<=30) & (hsv[...,1]>=(65 if neutral_reflections else 12)) & (hsv[...,1]<=180) & (hsv[...,2]>=50)
    pale_edges &= ~skin
    graph = cv2.dilate(pale_edges.astype(np.uint8),np.ones((5,5),np.uint8))
    # Opaque body is the anchor only. Neither side's search area is filled.
    graph[:,x1-left:x2-left] = 1
    _, labels, stats, _ = cv2.connectedComponentsWithStats(graph,8)
    anchor = int(labels[height//2,(x1+x2)//2-left])
    if not anchor:
        return [fitted]
    ex, _, ew, _, _ = map(int,stats[anchor])
    endpoints = (left+ex,left+ex+ew)
    variants = []
    min_extra = max(4,int(height*.03))
    if x1-endpoints[0] >= min_extra:
        extra = pale_edges[:,endpoints[0]-left:x1-left]
        density = float(extra.mean()) if extra.size else 0.
        if density >= .015:
            variants.append((density,(endpoints[0],y1,x2,y2)))
    if endpoints[1]-x2 >= min_extra:
        extra = pale_edges[:,x2-left:endpoints[1]-left]
        density = float(extra.mean()) if extra.size else 0.
        if density >= .015:
            variants.append((density,(x1,y1,endpoints[1],y2)))
    # Preserve each direction separately: a connected window on the opposite
    # side cannot expand an otherwise correct tail proposal.
    return [box for _,box in sorted(variants,reverse=True)]+[fitted]


def bright_component_candidates(frame,box,aspect=None,accept_seed=None,max_candidates=4):
    """Return a bounded set of content-based boxes, not recognized detections.

    Several independently structured seeds are tried when the leading seed
    fails. Artificial search borders still reject incomplete foreground; real
    image borders may return the visible portion for the caller's strong check.
    """
    max_candidates=max(0,int(max_candidates))
    if not max_candidates:
        return []
    x1,y1,x2,y2=map(int,box)
    x1,y1=max(0,x1),max(0,y1)
    x2,y2=min(frame.shape[1],x2),min(frame.shape[0],y2)
    roi=frame[y1:y2,x1:x2]
    if min(roi.shape[:2],default=0)<16:
        return []
    scale=min(1.,192./max(roi.shape[:2]))
    small=cv2.resize(roi,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    mask=((hsv[...,1]<25)&(hsv[...,2]>165)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    edges=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),40,120)
    mask[cv2.dilate(edges,np.ones((3,3),np.uint8))>0]=0
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    h,w=mask.shape
    choices=[]
    for i in range(1,n):
        bx,by,bw,bh,area=map(int,stats[i])
        # Real frame edges are visibility limits, rather than evidence that
        # a search window has cut the object. Uniform huge seeds remain barred.
        touched=sum((bx<=1 and x1>0,by<=1 and y1>0,
                     bx+bw>=w-1 and x2<frame.shape[1],
                     by+bh>=h-1 and y2<frame.shape[0]))
        if area<max(16,.002*w*h) or area>.8*w*h or touched>=2:
            continue
        seed_box=(x1+int(bx/scale),y1+int(by/scale),
                  min(x2,x1+int(np.ceil((bx+bw)/scale))),
                  min(y2,y1+int(np.ceil((by+bh)/scale))))
        if accept_seed is not None and not accept_seed(seed_box):
            continue
        distance=((centers[i,0]-w/2)/w)**2+((centers[i,1]-h/2)/h)**2
        # Position gently weights a real component rather than subtracting
        # enough score to eliminate small objects at the frame's corners.
        choices.append((float(area)/(w*h)/(1.+4.*distance),i))
    if not choices:
        return []
    groups=[]
    # Bounded GrabCut calls; proposals from each seed will be classified as one
    # batch by the caller. Trying alternatives must not create a job backlog.
    seed_limit=min(6,max(3,min(max_candidates,6)))
    for _,seed_id in sorted(choices,reverse=True)[:seed_limit]:
        seed=labels==seed_id
        gc=np.full(mask.shape,cv2.GC_PR_BGD,np.uint8)
        spread=cv2.dilate(seed.astype(np.uint8),np.ones((9,9),np.uint8))>0
        gc[spread]=cv2.GC_PR_FGD
        core=cv2.erode(seed.astype(np.uint8),np.ones((3,3),np.uint8))>0
        gc[core]=cv2.GC_FGD
        skin=(hsv[...,0]<=30)&(hsv[...,1]>=45)&(hsv[...,2]>=55)
        gc[skin&~spread]=cv2.GC_BGD
        gc[:2]=gc[-2:]=cv2.GC_BGD
        gc[:,:2]=gc[:,-2:]=cv2.GC_BGD
        bx,by,bw,bh,_=map(int,stats[seed_id])
        # Restrict all opaque-body fits, including interior objects, so a
        # similarly bright neighboring window cannot merge into the body.
        # The separate edge trace can still recover transparent extensions.
        mx,my=max(2,int(bw*.2)),max(2,int(bh*.2))
        safe=np.zeros(gc.shape,bool)
        safe[max(0,by-my):min(h,by+bh+my),max(0,bx-mx):min(w,bx+bw+mx)]=True
        gc[~safe]=cv2.GC_BGD
        if not (gc==cv2.GC_FGD).any():
            continue
        try:
            cv2.grabCut(small,gc,None,np.zeros((1,65),np.float64),
                        np.zeros((1,65),np.float64),2,cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        fg=((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD)).astype(np.uint8)
        n,parts,part_stats,_=cv2.connectedComponentsWithStats(fg,8)
        if n<=1:
            continue
        overlaps=[(int(((parts==i)&seed).sum()),i) for i in range(1,n)]
        part_id=max(overlaps)[1]
        bx,by,bw,bh,area=map(int,part_stats[part_id])
        # Retain meaningful support from this seed rather than requiring a
        # small object to occupy two percent of an entire camera frame.
        if bw<8 or bh<8 or area<max(16,.5*stats[seed_id][4]):
            continue
        component=parts==part_id
        rows=component.sum(axis=1)
        strong_rows=np.flatnonzero(rows>=.35*rows.max())
        if strong_rows.size:
            bottom=min(by+bh,int(strong_rows[-1])+1)
            by=max(by,int(strong_rows[0]));bh=bottom-by
            area=int(component[by:by+bh,bx:bx+bw].sum())
        if ((bx<=2 and x1>0) or (by<=2 and y1>0)
                or (bx+bw>=w-2 and x2<frame.shape[1])
                or (by+bh>=h-2 and y2<frame.shape[0])):
            continue
        fitted=(max(x1,x1+int((bx-2)/scale)),max(y1,y1+int((by-2)/scale)),
                min(x2,x1+int(np.ceil((bx+bw+2)/scale))),
                min(y2,y1+int(np.ceil((by+bh+2)/scale))))
        occupancy=float(area)/float(max(1,bw*bh))
        groups.append([(b,occupancy) for b in _transparent_candidates(frame,fitted,aspect)])
    # Keep seed diversity before filling the remaining budget with alternate
    # directions. A bad leading seed cannot monopolize every output position.
    output=[];seen=set()
    for depth in range(max((len(g) for g in groups),default=0)):
        for group in groups:
            if depth<len(group) and group[depth][0] not in seen:
                output.append(group[depth]);seen.add(group[depth][0])
                if len(output)>=max_candidates:
                    return output
    return output


def fit_bright_component(frame,box,aspect=None,accept_seed=None):
    """Compatibility wrapper; new searches should classify all candidates."""
    candidates=bright_component_candidates(frame,box,aspect,accept_seed,max_candidates=4)
    return candidates[0] if candidates else None
