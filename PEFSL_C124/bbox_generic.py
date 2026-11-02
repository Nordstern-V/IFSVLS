"""Object-independent registration references; no category or colour presets."""
import cv2
import numpy as np
import time


def object_region_candidates(frame, max_candidates=96):
    """Bounded, category-independent proposals from boundaries and Lab regions.

    These are hypotheses, not detections or foreground masks. In particular a
    hand, monitor or wall can produce one; the caller must verify learned
    identity on the original pixels. No learned colour or preferred position
    is used to construct the regions.
    """
    h, w = frame.shape[:2]
    if min(h, w) < 24:
        return []
    scale = min(1., 320. / max(h, w))
    image = cv2.resize(frame, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    smooth = cv2.bilateralFilter(image, 5, 24, 3)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.int32)
    proposals = {}

    def keep(x, y, rw, rh, support, source):
        if min(rw, rh) < max(7, round(20*scale)):
            return
        area = rw*rh
        if area > .90*image.shape[0]*image.shape[1]:
            return
        # A small symmetric border compensates for downsampling/closing.
        pad = max(1, round(min(rw, rh)*.02))
        box = (max(0, int((x-pad)/scale)), max(0, int((y-pad)/scale)),
               min(w, int(np.ceil((x+rw+pad)/scale))),
               min(h, int(np.ceil((y+rh+pad)/scale))))
        quality = float(support)*np.sqrt(area)
        if box not in proposals or quality > proposals[box][0]:
            proposals[box] = (quality, dict(region_source=source,
                                           region_support=round(float(support), 3)))

    for low, high in ((30, 90), (65, 160)):
        edges = cv2.Canny(gray, low, high)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, rw, rh = cv2.boundingRect(contour)
            fill = cv2.contourArea(contour)/max(1., rw*rh)
            if fill >= .30:
                keep(x, y, rw, rh, fill, 'edges')

    # Offset quantizers reduce fragmentation at a colour bin boundary. Black,
    # neutral and saturated regions follow exactly the same construction.
    for offset in (0, 16):
        codes = (lab+offset)//32
        codes = codes[..., 0]*100+codes[..., 1]*10+codes[..., 2]
        values, counts = np.unique(codes, return_counts=True)
        for value in values[counts >= max(20, int(300*scale*scale))]:
            mask = (codes == value).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((3, 3), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for x, y, rw, rh, pixels in stats[1:n]:
                fill = pixels/max(1., rw*rh)
                if fill >= .45:
                    keep(int(x), int(y), int(rw), int(rh), fill, 'lab_region')
    out = []
    for box, (_, proof) in sorted(proposals.items(), key=lambda item: item[1][0],
                                  reverse=True):
        area = (box[2]-box[0])*(box[3]-box[1])
        if any((max(0, min(box[2], b[2])-max(box[0], b[0]))
                * max(0, min(box[3], b[3])-max(box[1], b[1])))
               / max(1, area+(b[2]-b[0])*(b[3]-b[1])
                     -max(0, min(box[2], b[2])-max(box[0], b[0]))
                     *max(0, min(box[3], b[3])-max(box[1], b[1]))) > .85
               for b, _ in out):
            continue
        out.append((box, proof))
        if len(out) >= max_candidates:
            break
    return out


def add_reference_view(bank, image, limit=8):
    """Keep complete, diverse ROI images; similar sharper views replace blur."""
    if image is None or image.size == 0:
        return
    if limit < 1:
        return
    h, w = image.shape[:2]
    scale = min(1., 256. / max(h,w))
    view = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                      interpolation=cv2.INTER_AREA).copy()
    small = cv2.resize(view, (32, 32)).astype(np.float32)
    def quality(value):
        gray = cv2.cvtColor(cv2.resize(value, (96, 64)), cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    distances = [float(np.abs(small-cv2.resize(old, (32, 32)).astype(np.float32)).mean())
                 for old in bank]
    if distances and min(distances) < 4.:
        nearest = int(np.argmin(distances))
        if quality(view) > 1.15*quality(bank[nearest]):
            bank[nearest] = view
        return
    bank.append(view)
    if len(bank) > limit:
        # Remove one member of the most redundant pair, rather than repeatedly
        # deleting early views. Raw spatial differences preserve actual faces;
        # this is a bounded view reservoir, not semantic pose estimation.
        thumbs = [cv2.resize(v, (32, 32)).astype(np.float32) for v in bank]
        _, i, j = min((float(np.abs(thumbs[i]-thumbs[j]).mean()), i, j)
                      for i in range(len(bank)) for j in range(i+1, len(bank)))
        remove = i if quality(bank[i]) < quality(bank[j]) else j
        del bank[remove]


def pack_reference_views(banks):
    views = [v for bank in banks for v in bank]
    images = np.zeros((len(views), 256, 256, 3), np.uint8)
    sizes = np.zeros((len(views), 2), np.int64)
    for i, view in enumerate(views):
        h, w = view.shape[:2]
        images[i, :h, :w] = view
        sizes[i] = h, w
    return dict(reference_view_counts=np.array([len(b) for b in banks], np.int64),
                reference_view_sizes=sizes, reference_views=images)


def unpack_reference_views(payload, classes):
    keys = ('reference_view_counts', 'reference_view_sizes', 'reference_views')
    if not any(k in payload for k in keys):
        return [[] for _ in range(classes)]
    if not all(k in payload for k in keys):
        raise ValueError('incomplete registration reference bank')
    counts, sizes, images = (payload[k] for k in keys)
    if (counts.shape != (classes,) or counts.dtype.kind not in 'iu'
            or sizes.dtype.kind not in 'iu' or images.dtype != np.uint8
            or np.any(counts < 0) or np.any(counts > 8)
            or images.shape != (int(counts.sum()), 256, 256, 3)
            or sizes.shape != (len(images), 2) or np.any(sizes < 1) or np.any(sizes > 256)):
        raise ValueError('invalid complete registration reference bank')
    banks = []; offset = 0
    for count in counts:
        bank = []
        for i in range(offset, offset+int(count)):
            h, w = map(int, sizes[i]); bank.append(images[i, :h, :w].copy())
        banks.append(bank); offset += int(count)
    return banks


def generic_colour_query(crop):
    """Prepare one current crop for comparisons with multiple learned views."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    query = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256]).astype(np.float32)
    query /= max(float(query.sum()), 1.)
    return hsv, query


def generic_colour_supported(crop, histogram, threshold, compensate=False, *, prepared=None):
    """Compare learned colours; an optional query belongs to this exact crop."""
    if histogram is None:
        return True
    hsv, query = generic_colour_query(crop) if prepared is None else prepared
    ref = np.asarray(histogram, np.float32).reshape(8, 8, 8)
    ref = ref/max(float(ref.sum()), 1e-9)
    if float(ref[:, 0, :].sum()) > .60:
        query, ref = query.sum(axis=0), ref.sum(axis=0)
    query, ref = query.flatten(), ref.flatten()
    supported=(cv2.compareHist(query, ref, cv2.HISTCMP_BHATTACHARYYA) <= threshold
               or float(np.minimum(query, ref).sum()) >= .35)
    if supported or not compensate or query.size!=512:return supported
    # Caller must first prove strong, distributed CURRENT learned geometry.
    # Fit bounded saturation/value gains only; hue must already agree. This
    # cannot turn a white/grey region or a different hue into the learned hue.
    q=query.reshape(8,8,8);r=ref.reshape(8,8,8)
    if min(float(q[:,2:,:].sum()),float(r[:,2:,:].sum()))<.5:return False
    qh=q.sum(axis=(1,2));rh=r.sum(axis=(1,2))
    if cv2.compareHist(qh,rh,cv2.HISTCMP_BHATTACHARYYA)>.25:return False
    centres=np.arange(8,dtype=np.float32)*32.+16.
    qs=float(q.sum(axis=(0,2))@centres);rs=float(r.sum(axis=(0,2))@centres)
    qv=float(q.sum(axis=(0,1))@centres);rv=float(r.sum(axis=(0,1))@centres)
    sg=rs/max(qs,1.);vg=rv/max(qv,1.)
    if not (.5<=sg<=2. and .7<=vg<=1.4):return False
    adjusted=hsv.copy()
    adjusted[...,1]=np.clip(adjusted[...,1].astype(np.float32)*sg,0,255).astype(np.uint8)
    adjusted[...,2]=np.clip(adjusted[...,2].astype(np.float32)*vg,0,255).astype(np.uint8)
    corrected=cv2.calcHist([adjusted],[0,1,2],None,[8,8,8],[0,180,0,256,0,256]).astype(np.float32)
    corrected/=max(float(corrected.sum()),1.)
    return (cv2.compareHist(corrected.ravel(),ref,cv2.HISTCMP_BHATTACHARYYA)<=threshold
            or float(np.minimum(corrected.ravel(),ref).sum())>=.35)


def scene_reference_features(views,method='orb'):
    detector=(cv2.SIFT_create(nfeatures=600,contrastThreshold=.02,sigma=1.2)
              if method=='sift' else
              cv2.ORB_create(nfeatures=600,edgeThreshold=7,patchSize=15,fastThreshold=5))
    references=[]
    for view_index,image in enumerate(views):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        points,features=detector.detectAndCompute(gray,None)
        if features is not None and len(points)>=12:
            references.append((np.float32([p.pt for p in points]),features,image.shape[:2],view_index))
    return references


def project_reference_candidates(frame,references,method='orb',time_budget=None,state=None,query_cache=None,
                                 progress=None,expand_empty=False):
    """Project complete learned ROIs from distributed current correspondences.

    This is one optional proposal source. Failed/nonplanar correspondences never
    suppress the grid, colour or motion sources. No class names are involved.
    Optional query_cache belongs to ONE frame/region/method only. A state
    rotates reference indices, never image pixels or acceptance results.
    Optional progress distinguishes an exhausted deadline from a completed
    negative search; candidate verification requirements are unchanged.
    """
    if progress is not None:progress.update(completed=False,checked=0)
    if not references:
        if progress is not None:progress['completed']=True
        return []
    deadline=time.perf_counter()+time_budget if time_budget is not None else float('inf')
    fh,fw=frame.shape[:2];scale=min(1.,(384. if method=='sift' else 640.)/max(fw,fh))
    cached=None if query_cache is None else query_cache.get('features')
    if cached is None:
        query=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(query,cv2.COLOR_BGR2GRAY)
        detector=(cv2.SIFT_create(nfeatures=600,contrastThreshold=.02,sigma=1.2)
                  if method=='sift' else
                  cv2.ORB_create(nfeatures=1800,edgeThreshold=7,patchSize=15,fastThreshold=5))
        points,features=detector.detectAndCompute(gray,None)
        xy=np.float32([p.pt for p in points])
        if query_cache is not None:query_cache['features']=(xy,features)
    else:xy,features=cached
    if features is None or len(xy)<12:
        if progress is not None:progress['completed']=True
        return []
    out=[];start=int((state or {}).get('cursor',0))%len(references)
    extend_empty=bool(expand_empty and state is not None and method=='sift')
    limit=min(8 if extend_empty else 3,len(references)) if state is not None else len(references)
    checked=0
    for step in range(limit):
        if extend_empty and step>=3 and out:break
        if time.perf_counter()>=deadline:break
        checked+=1
        index=(start+step)%len(references)
        if state is not None:state['cursor']=(index+1)%len(references)
        source,descriptors,(rh,rw)=references[index][:3]
        pairs=cv2.BFMatcher(cv2.NORM_L2 if method=='sift' else cv2.NORM_HAMMING).knnMatch(descriptors,features,k=2)
        matches=sorted((a for pair in pairs if len(pair)==2 for a,b in [pair]
                        if a.distance<.75*b.distance and a.distance<=(250 if method=='sift' else 60)),key=lambda m:m.distance)
        used=set();unique=[]
        for match in matches:
            if match.trainIdx not in used:unique.append(match);used.add(match.trainIdx)
        if len(unique)<12:continue
        src=np.float32([source[m.queryIdx] for m in unique])
        dst=np.float32([xy[m.trainIdx] for m in unique])
        matrix,mask=cv2.findHomography(src,dst,cv2.RANSAC,2.5)
        if matrix is None or mask is None:continue
        keep=mask.ravel()>0;n=int(keep.sum())
        if n<12 or keep.mean()<.6:continue
        if np.any(np.ptp(src[keep],axis=0)<.30*np.float32([rw,rh])):continue
        if cv2.contourArea(cv2.convexHull(src[keep]))<.06*rw*rh:continue
        predicted=cv2.perspectiveTransform(src[keep][None],matrix)[0]
        error=float(np.median(np.linalg.norm(predicted-dst[keep],axis=1)))
        if error>2.:continue
        quad=cv2.perspectiveTransform(np.float32([[[0,0],[rw-1,0],[rw-1,rh-1],[0,rh-1]]]),matrix)[0]/scale
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):continue
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        if np.min(hi-lo)<24 or np.prod(hi-lo)>1.5*fw*fh:continue
        clipped_lo=np.maximum(lo,0);clipped_hi=np.minimum(hi,[fw,fh])
        if np.prod(np.maximum(0,clipped_hi-clipped_lo))<.55*np.prod(hi-lo):continue
        box=(max(0,int(np.floor(lo[0]))-2),max(0,int(np.floor(lo[1]))-2),
             min(fw,int(np.ceil(hi[0]))+2),min(fh,int(np.ceil(hi[1]))+2))
        if min(box[2]-box[0],box[3]-box[1])<24:continue
        area=float(np.prod(hi-lo))
        intersection=float(np.prod(np.maximum(0,clipped_hi-clipped_lo)))
        out.append((box,dict(reference=index,detector=method,inliers=n,matches=len(unique),verified=True,
                             expanded_reference=bool(extend_empty and step>=3),
                             ratio=round(float(keep.mean()),3),median_error=round(error,3),
                             source_coverage=(np.ptp(src[keep],axis=0)/np.float32([rw,rh])).round(3).tolist(),
                             extent_iou=round(intersection/max(1.,area+fw*fh-intersection),3),
                             observed_points=(dst[keep]/scale).tolist(),
                             full_quad=quad.tolist())))
    # An unsuccessful optional supplement must not change the next ordinary
    # three-reference packet used by the established search rotation.
    if extend_empty and checked>3 and not out:
        state['cursor']=(start+3)%len(references)
    if progress is not None:progress.update(
        completed=checked==limit or bool(extend_empty and checked>=3 and out),checked=checked)
    return sorted(out,key=lambda item:item[1]['inliers'],reverse=True)[:4]


def template_reference_candidates(frame,views,fraction,scales,max_candidates=12,appearance_ok=None,
                                  state=None,time_budget=None,only_variant=None,progress=None):
    """Whole-reference fallback when an object has too few ORB keypoints.

    Retain multiple independently placed hypotheses. A high-scoring background
    cannot prevent the remaining candidates from reaching identity verification.
    """
    h,w=frame.shape[:2];scale=min(1.,480./max(w,h))
    gray=cv2.cvtColor(cv2.resize(frame,None,fx=scale,fy=scale),cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),.8).astype(np.float32)
    nominal=max(1.,float(fraction)*min(w,h)*scale)
    sizes=sorted({round(float(fraction)*s*min(w,h)*scale) for s in scales},
                 key=lambda side:abs(np.log(max(1,side)/nominal)))
    indices=sorted({0,len(views)//2,len(views)-1}) if views else []
    candidates=[]
    # Start near the learned size; don't spend every short job on the
    # smallest scales before the normal-size object gets a chance.
    # Interleave learned views while retaining every scale/orientation.
    variants=[(index,flip,rotation,side) for flip in (False,True)
              for rotation in range(4) for side in sizes for index in indices]
    if only_variant is not None:
        variants=[variant for variant in variants if variant==tuple(only_variant)]
    start=int((state or {}).get('cursor',0))%max(1,len(variants))
    deadline=time.perf_counter()+time_budget if time_budget is not None else float('inf')
    visited=0;responses=0;peak_checks=0;interrupted_peaks=0
    # A cursor remembers only which variant to try next, never old pixels or
    # scores. With no state/budget this preserves the original complete search.
    for step in range(len(variants)):
        if time.perf_counter()>=deadline:break
        index,flip,rotation,side=variants[(start+step)%len(variants)]
        visited+=1
        image=np.ascontiguousarray(np.rot90(views[index][:,::-1] if flip else views[index],rotation))
        rh,rw=image.shape[:2]
        tw,th=(side,max(12,round(side*rh/rw))) if rw>=rh else (max(12,round(side*rw/rh)),side)
        if min(tw,th)<12 or tw>gray.shape[1] or th>gray.shape[0]:continue
        template=cv2.cvtColor(cv2.resize(image,(tw,th)),cv2.COLOR_BGR2GRAY)
        template=cv2.GaussianBlur(template,(3,3),.8).astype(np.float32)
        if float(template.std())<5.:continue
        scores=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        responses+=1
        for peak_index in range(8):
            # The response has already been computed. Inspect its strongest
            # peak even if that indivisible OpenCV call crossed the soft
            # deadline, rather than advancing the cursor without using it.
            if peak_index and time.perf_counter()>=deadline:
                interrupted_peaks+=1
                break
            peak_checks+=1
            _,score,_,(x,y)=cv2.minMaxLoc(scores)
            if not np.isfinite(score) or score<.45:break
            box=(max(0,round(x/scale)),max(0,round(y/scale)),
                 min(w,round((x+tw)/scale)),min(h,round((y+th)/scale)))
            if appearance_ok is None or appearance_ok(box):
                candidates.append((box,dict(reference=index,rotation=rotation,flip=flip,
                    template_similarity=round(score,3),template_variant=(index,flip,rotation,side))))
            scores[max(0,y-th//2):y+th//2+1,max(0,x-tw//2):x+tw//2+1]=-1
    if state is not None:
        state['cursor']=(start+visited)%max(1,len(variants))
    if progress is not None:
        progress.update(completed=visited==len(variants) and not interrupted_peaks,
            variants_visited=visited,variants_total=len(variants),responses=responses,
            peak_checks=peak_checks,interrupted_peaks=interrupted_peaks)
    out=[]
    for box,proof in sorted(candidates,key=lambda item:item[1]['template_similarity'],reverse=True):
        area=(box[2]-box[0])*(box[3]-box[1])
        overlaps=False
        for other,_ in out:
            overlap=max(0,min(box[2],other[2])-max(box[0],other[0]))*max(0,min(box[3],other[3])-max(box[1],other[1]))
            union=area+(other[2]-other[0])*(other[3]-other[1])-overlap
            if overlap/max(1,union)>.6:overlaps=True;break
        if not overlaps:out.append((box,proof))
        if len(out)>=max_candidates:break
    return out


