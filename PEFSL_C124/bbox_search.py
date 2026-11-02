"""Bounded, class-independent proposal selection and local window refinement.

These functions supply hypotheses only. Callers must verify current identity,
features, background and negatives before displaying a detection.
"""
import numpy as np
import cv2


def select_diverse(boxes, features, appearances, width, height, limit=16, registrations=None):
    """Reserve places for different rankings and spatial cells.

    A whole-crop HOG score alone must not discard a partially occluded object.
    Negative feature scores remain vetoes; no identity threshold is changed.
    """
    valid=[i for i in range(len(boxes)) if np.isfinite(features[i]) and features[i]>=0.]
    if not valid:return []
    by_feature=sorted(valid,key=lambda i:(features[i],appearances[i]),reverse=True)
    by_appearance=sorted(valid,key=lambda i:(appearances[i],features[i]),reverse=True)
    by_registration=(sorted(valid,key=lambda i:(registrations[i],features[i]),reverse=True)
                     if registrations is not None else by_appearance)
    by_joint=sorted(valid,key=lambda i:(.5*features[i]+.5*appearances[i]),reverse=True)
    cells={}
    for i in by_joint:
        x,y,r,b=boxes[i]
        cell=(min(2,int((x+r)*1.5/max(1,width))),
              min(1,int((y+b)/max(1,height))))
        cells.setdefault(cell,[]).append(i)
    spatial=[]
    for depth in range(max(map(len,cells.values()))):
        spatial.extend(group[depth] for group in cells.values() if depth<len(group))
    result=[];seen=set()
    # Round-robin, rather than one ranking consuming all available places.
    for depth in range(len(valid)):
        for ranking in (by_registration,by_appearance,by_feature,spatial):
            if depth>=len(ranking):continue
            i=ranking[depth];box=tuple(boxes[i])
            if box in seen:continue
            seen.add(box);result.append(i)
            if len(result)>=limit:return result
    return result


def registration_correlations(frame, boxes, references):
    """Cheap full-reference ranking only; no acceptance or rejection threshold."""
    if not boxes or not references:return np.zeros(len(boxes),np.float32)
    shape=references[0]['gray'].shape
    queries=[]
    for x,y,r,b in boxes:
        gray=cv2.cvtColor(frame[y:b,x:r],cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
        vector=cv2.GaussianBlur(gray,(3,3),.6).astype(np.float32).ravel()
        vector-=vector.mean();vector/=max(float(np.linalg.norm(vector)),1e-6)
        queries.append(vector)
    bank=np.stack([ref['unit'] for ref in references])
    return (np.stack(queries)@bank.T).max(axis=1)


def local_variants(box, width, height):
    """Small measured-window alternatives; never an authorized box expansion."""
    x,y,r,b=map(float,box);w,h=r-x,b-y;cx,cy=(x+r)/2,(y+b)/2
    specs=[(0.,0.,1.,1.)]
    for delta in (.08,.20):
        specs.extend((dx,dy,1.,1.) for dx,dy in
                     ((-delta,0.),(delta,0.),(0.,-delta),(0.,delta)))
    specs.extend((dx,dy,1.,1.) for dx in (-.08,.08) for dy in (-.08,.08))
    specs.extend((0.,0.,sx,sy) for sx,sy in
                 ((.85,.85),(1.15,1.15),(.9,1.),(1.1,1.),(1.,.9),(1.,1.1)))
    result=[];seen=set()
    for dx,dy,sx,sy in specs:
        candidate=(max(0,round(cx+dx*w-sx*w/2)),max(0,round(cy+dy*h-sy*h/2)),
                   min(width,round(cx+dx*w+sx*w/2)),min(height,round(cy+dy*h+sy*h/2)))
        if min(candidate[2]-candidate[0],candidate[3]-candidate[1])<24 or candidate in seen:
            continue
        seen.add(candidate);result.append(candidate)
    return result
