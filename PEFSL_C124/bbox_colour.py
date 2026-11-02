"""Measured colour components for opaque, chromatic learned objects.

Colour proposes a boundary only. Classification, learned spatial identity,
background and negative checks belong to the caller.
"""
import cv2
import numpy as np


def chromatic_model(histogram):
    if histogram is None:
        return None
    hist=np.asarray(histogram,np.float32).reshape(8,8,8)
    total=float(hist.sum())
    hue=hist[:,2:,2:].sum(axis=(1,2))
    if total<=0 or float(hue.sum())<.35*total:
        return None
    peak=int(hue.argmax())
    if float(hue[peak])<.30*float(hue.sum()):
        return None
    return (peak+.5)*180./8


def colour_component_candidates(frame, histogram, region=None, max_candidates=3):
    """Find visible connected colour planes without imposing learned aspect.

    A primary learned hue anchors each plane. Adjacent saturated colour can
    complete a multicolour cover, within a narrow band around that anchor.
    Skin and dark background cannot grow the plane. Returned bounds always
    come from current component pixels, including at real image borders.
    """
    centre=chromatic_model(histogram)
    if centre is None or max_candidates<=0:
        return []
    fh,fw=frame.shape[:2]
    x1,y1,x2,y2=(0,0,fw,fh) if region is None else tuple(map(int,region))
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(fw,x2),min(fh,y2)
    if min(x2-x1,y2-y1)<24:
        return []
    scale=min(1.,384./max(x2-x1,y2-y1))
    image=cv2.resize(frame[y1:y2,x1:x2],None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hue=hsv[...,0].astype(np.float32)
    delta=np.abs(hue-centre);delta=np.minimum(delta,180.-delta)
    primary=(delta<=18.)&(hsv[...,1]>=45)&(hsv[...,2]>=65)
    primary=cv2.morphologyEx(primary.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    primary=cv2.morphologyEx(primary,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(primary,8)
    choices=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if min(w,h)<12 or area<max(40,.003*image.shape[0]*image.shape[1]):
            continue
        if area/float(w*h)<.45:
            continue
        choices.append((area,i))
    output=[]
    for _,i in sorted(choices,reverse=True)[:max_candidates]:
        x,y,w,h,_=map(int,stats[i]);seed=labels==i
        # Allow a connected second cover colour, e.g. a yellow title strip.
        # A real skin hue outside the learned primary hue remains excluded.
        # Orange/yellow hand reflections have much lower saturation than a
        # printed yellow band. Do not join them merely because hue overlaps.
        secondary=((hue>=20)&(hue<=165)&((hue>=45)|(hsv[...,1]>=110)))
        chromatic=((hsv[...,1]>=45)&(hsv[...,2]>=65)&(secondary|(delta<=18.)))
        # A cropped cover can look wider than tall. Try both axes rather
        # than deciding the colour-band direction from that truncated shape.
        axes=[(0,max(2,int(.40*h))),(max(2,int(.40*w)),0)]
        if w>h:axes.reverse()
        for mx,my in axes:
            allowed=np.zeros(primary.shape,np.uint8)
            allowed[max(0,y-my):min(image.shape[0],y+h+my),
                    max(0,x-mx):min(image.shape[1],x+w+mx)]=1
            graph=cv2.morphologyEx((chromatic&allowed.astype(bool)).astype(np.uint8),
                                  cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            graph[seed]=1
            count,parts,part_stats,_=cv2.connectedComponentsWithStats(graph,8)
            overlaps=[(int(((parts==j)&seed).sum()),j) for j in range(1,count)]
            if not overlaps:
                continue
            part=max(overlaps)[1];bx,by,bw,bh,area=map(int,part_stats[part])
            if bw>(1.6 if mx else 1.2)*w or bh>(1.6 if my else 1.2)*h or area/float(bw*bh)<.50:
                continue
            # A local search border is artificial; it cannot prove the outside
            # extent of a component. A real camera border is a visibility limit.
            if ((bx<=1 and x1>0) or (by<=1 and y1>0)
                    or (bx+bw>=image.shape[1]-1 and x2<fw)
                    or (by+bh>=image.shape[0]-1 and y2<fh)):
                continue
            points=cv2.findNonZero((parts==part).astype(np.uint8))
            hull=cv2.convexHull(points)
            if cv2.contourArea(hull)/float(bw*bh)<.65:
                continue
            box=(max(0,x1+int(bx/scale)),max(0,y1+int(by/scale)),
                 min(fw,x1+int(np.ceil((bx+bw)/scale))),
                 min(fh,y1+int(np.ceil((by+bh)/scale))))
            if min(box[2]-box[0],box[3]-box[1])<24:
                continue
            if any(box==b for b,e in output):continue
            output.append((box,dict(primary_box=(x1+int(x/scale),y1+int(y/scale),
                x1+int((x+w)/scale),y1+int((y+h)/scale)),primary_fraction=round(float(seed[by:by+bh,bx:bx+bw].mean()),3),
                                    occupancy=round(area/float(bw*bh),3),hue=round(centre,2))))
    return output[:max_candidates]
