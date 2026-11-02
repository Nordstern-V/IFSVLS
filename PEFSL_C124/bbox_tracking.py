"""Short-term geometric motion estimates; identity is checked by the caller."""
from typing import NamedTuple

import cv2
import numpy as np
import threading

_HOG = cv2.HOGDescriptor((96,64),(16,16),(8,8),(8,8),9)
APPEARANCE_SIZE = _HOG.getDescriptorSize()
_HOG_STATE = threading.local()


def describe_appearance(frame, box):
    x1,y1,x2,y2 = map(int,box)
    roi = frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
    if min(roi.shape[:2],default=0)<12:
        return None
    gray=cv2.resize(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),(96,64),interpolation=cv2.INTER_AREA)
    if not hasattr(_HOG_STATE,"descriptor"):
        _HOG_STATE.descriptor=cv2.HOGDescriptor((96,64),(16,16),(8,8),(8,8),9)
    vector=_HOG_STATE.descriptor.compute(gray).ravel()
    length=float(np.linalg.norm(vector))
    return vector/length if length>1e-6 else None


class MotionEstimate(NamedTuple):
    box: tuple
    points: np.ndarray
    velocity: tuple
    inlier_ratio: float
    coverage: tuple = (1., 1.)
    corners: np.ndarray = None


def seed_points(frame, gray, box, prefer_bright=True):
    x1, y1, x2, y2 = map(int, box)
    h, w = gray.shape
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    if x2-x1 < 12 or y2-y1 < 12:
        return None
    mask = np.zeros_like(gray)
    dx, dy = max(2,int(.05*(x2-x1))), max(2,int(.05*(y2-y1)))
    mask[y1+dy:y2-dy, x1+dx:x2-dx] = 255
    if not prefer_bright:
        # A generic object may be saturated, dark or made of skin-like material.
        # Its interior corners must remain available to optical-flow tracking.
        small=min(x2-x1,y2-y1)<80
        return cv2.goodFeaturesToTrack(gray,maxCorners=80 if small else 60,
            qualityLevel=.01 if small else .015,minDistance=2 if small else 4,
            mask=mask,blockSize=3)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Reflections through transparent plastic do not move rigidly with the
    # object. Prefer corners in the opaque bright body (including its gray
    # internal markings), instead of refractions of the hand behind it.
    bright = ((hsv[...,1] < 25) & (hsv[...,2] > 165) & (mask>0)).astype(np.uint8)
    bright = cv2.morphologyEx(bright,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,_ = cv2.connectedComponentsWithStats(bright,8)
    if n>1:
        selected = 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        component = (labels==selected).astype(np.uint8)
        if int(component.sum()) >= .08*(x2-x1)*(y2-y1):
            contours,_ = cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                core = np.zeros_like(mask)
                cv2.fillConvexPoly(core,cv2.convexHull(max(contours,key=cv2.contourArea)),255)
                mask &= core
    # Prefer structure in the object instead of surrounding skin.
    mask[(hsv[...,1] > 45) | (hsv[...,2] < 40)] = 0
    small=min(x2-x1,y2-y1)<80
    return cv2.goodFeaturesToTrack(gray, maxCorners=80 if small else 60,
                                  qualityLevel=.01 if small else .015,
                                  minDistance=2 if small else 4, mask=mask, blockSize=3)


class LocalCorrelationTracker:
    """Bounded position proposals; caller independently verifies identity."""
    def __init__(self,frame,box):
        self.scale=min(1.,400./max(frame.shape[:2]))
        self.last_box=tuple(box)
        params=cv2.TrackerCSRT_Params()
        params.use_color_names=False
        params.use_rgb=False
        params.use_gray=True
        params.template_size=100.
        params.number_of_scales=9
        params.admm_iterations=2
        params.num_hog_channels_used=9
        self.tracker=cv2.TrackerCSRT_create(params)
        image=cv2.resize(frame,None,fx=self.scale,fy=self.scale,interpolation=cv2.INTER_AREA)
        x1,y1,x2,y2=box
        self.tracker.init(image,(int(x1*self.scale),int(y1*self.scale),
                                 max(2,int((x2-x1)*self.scale)),max(2,int((y2-y1)*self.scale))))

    def update(self,frame):
        image=cv2.resize(frame,None,fx=self.scale,fy=self.scale,interpolation=cv2.INTER_AREA)
        ok,(x,y,w,h)=self.tracker.update(image)
        if not ok:return None
        box=(max(0,int(round(x/self.scale))),max(0,int(round(y/self.scale))),
             min(frame.shape[1],int(round((x+w)/self.scale))),
             min(frame.shape[0],int(round((y+h)/self.scale))))
        if min(box[2]-box[0],box[3]-box[1])<12:return None
        self.last_box=box
        return box


def track_points(previous, current, points, box, corners=None):
    if (previous is None or not isinstance(points, np.ndarray)
            or points.ndim not in (2, 3) or len(points) < 6):
        return None
    # Camera gaps or invalid motion state are failed proposals, not reasons
    # to terminate the GUI. Keep valid measurements and their gates unchanged.
    if (not isinstance(previous, np.ndarray) or not isinstance(current, np.ndarray)
            or previous.ndim != 2 or previous.shape != current.shape
            or previous.dtype != np.uint8 or current.dtype != np.uint8
            or previous.size == 0):
        return None
    points = np.asarray(points)
    if (points.dtype != np.float32 or not np.isfinite(points).all()
            or not (points.ndim == 2 and points.shape[1:] == (2,)
                    or points.ndim == 3 and points.shape[1:] == (1, 2))):
        return None
    try:
        bounds = np.asarray(box)
    except (ValueError, TypeError):
        return None
    if (bounds.shape != (4,) or bounds.dtype.kind not in 'iuf'
            or not np.isfinite(bounds).all() or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]):
        return None
    if corners is not None:
        try:
            corners = np.asarray(corners)
        except (ValueError, TypeError):
            return None
        if corners.shape != (4, 2) or corners.dtype.kind not in 'iuf' or not np.isfinite(corners).all():
            return None
    options = dict(winSize=(21,21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
    try:
        new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    except cv2.error:
        return None
    if new is None or status is None:
        return None
    try:
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new, None, **options)
    except cv2.error:
        return None
    if back is None or back_status is None:
        return None
    src = points.reshape(-1,2)
    dst = new.reshape(-1,2)
    error = np.linalg.norm(src-back.reshape(-1,2), axis=1)
    h,w = current.shape
    good = (status.ravel()>0) & (back_status.ravel()>0) & (error<1.5)
    good &= np.isfinite(dst).all(axis=1) & (dst[:,0]>=0) & (dst[:,0]<w)
    good &= (dst[:,1]>=0) & (dst[:,1]<h)
    if np.count_nonzero(good) < 6:
        return None
    try:
        transform, inliers = cv2.estimateAffinePartial2D(src[good], dst[good],
            method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=500, confidence=.99)
    except cv2.error:
        return None
    if transform is None or inliers is None:
        return None
    inliers = inliers.ravel().astype(bool)
    ratio = float(inliers.mean())
    scale = float(np.hypot(transform[0,0],transform[0,1]))
    rotation = abs(float(np.degrees(np.arctan2(transform[1,0],transform[0,0]))))
    if np.count_nonzero(inliers) < 6 or ratio < .65 or not .8 <= scale <= 1.25 or rotation > 25:
        return None
    x1,y1,x2,y2 = map(int,box)
    if corners is None:
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float64)
    moved = corners @ transform[:,:2].T + transform[:,2]
    lo,hi = moved.min(axis=0),moved.max(axis=0)
    fitted = (max(0,int(round(lo[0]))),max(0,int(round(lo[1]))),
              min(w,int(round(hi[0]))),min(h,int(round(hi[1]))))
    if fitted[2]-fitted[0] < 12 or fitted[3]-fitted[1] < 12:
        return None
    displacement = np.median(dst[good][inliers]-src[good][inliers],axis=0)
    if np.linalg.norm(displacement) > .75*max(x2-x1,y2-y1):
        return None
    supported = dst[good][inliers]
    span = np.ptp(supported, axis=0)
    coverage = (float(span[0]/max(1,fitted[2]-fitted[0])),
                float(span[1]/max(1,fitted[3]-fitted[1])))
    return MotionEstimate(fitted, supported.reshape(-1,1,2).astype(np.float32),
                          tuple(float(v) for v in displacement), ratio, coverage, moved)


def shift_box(box, velocity, width, height):
    x1,y1,x2,y2 = box
    dx,dy = velocity
    bw,bh = x2-x1,y2-y1
    x1 = int(np.clip(round(x1+dx),0,max(0,width-bw)))
    y1 = int(np.clip(round(y1+dy),0,max(0,height-bh)))
    return (x1,y1,x1+bw,y1+bh)


def reference_sharpness(gray, box):
    """Compare detail at a fixed size so zoom alone does not refresh a template."""
    x1,y1,x2,y2=map(int,box)
    roi=gray[max(0,y1):min(gray.shape[0],y2),max(0,x1):min(gray.shape[1],x2)]
    if min(roi.shape,default=0)<12:return 0.
    roi=cv2.resize(roi,(96,64),interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(roi,cv2.CV_32F).var())


def recovery_region(box,velocity,width,height,fail=0):
    """Local search grows with displacement; never changes the identity gates."""
    predicted=shift_box(box,velocity,width,height)
    bw,bh=box[2]-box[0],box[3]-box[1]
    margin=.3+.15*min(3,max(0,int(fail)))
    mx=int(margin*bw+min(.5*bw,2*abs(velocity[0])))
    my=int(margin*bh+min(.5*bh,2*abs(velocity[1])))
    return (max(0,min(box[0],predicted[0])-mx),max(0,min(box[1],predicted[1])-my),
            min(width,max(box[2],predicted[2])+mx),min(height,max(box[3],predicted[3])+my))


def relocate_template(previous,current,box,min_similarity=.80,search_box=None):
    """Locate recent source content; the caller must reclassify the new box."""
    if previous is None or current is None or previous.shape!=current.shape:return None
    height,width=current.shape
    x1,y1,x2,y2=map(int,box)
    x1,y1=max(0,x1),max(0,y1);x2,y2=min(width,x2),min(height,y2)
    if min(x2-x1,y2-y1)<24:return None
    down=min(1.,320./max(width,height))
    before=cv2.GaussianBlur(cv2.resize(previous,None,fx=down,fy=down),(3,3),0)
    now=cv2.GaussianBlur(cv2.resize(current,None,fx=down,fy=down),(3,3),0)
    sx1,sy1,sx2,sy2=[int(round(v*down))for v in (x1,y1,x2,y2)]
    bw,bh=sx2-sx1,sy2-sy1
    dx,dy=max(1,int(.08*bw)),max(1,int(.08*bh))
    template=before[sy1+dy:sy2-dy,sx1+dx:sx2-dx]
    if min(template.shape,default=0)<8 or float(template.std())<5.:return None
    left,top=max(0,sx1-int(.6*bw)),max(0,sy1-int(.6*bh))
    right,bottom=min(now.shape[1],sx2+int(.6*bw)),min(now.shape[0],sy2+int(.6*bh))
    if search_box is not None:
        left,top,right,bottom=[int(round(v*down))for v in search_box]
        left,top=max(0,left),max(0,top)
        right,bottom=min(now.shape[1],right),min(now.shape[0],bottom)
    search=now[top:bottom,left:right];best=None
    for scale in (.70,.85,1.,1.15,1.30):
        tw,th=max(8,int(round(template.shape[1]*scale))),max(8,int(round(template.shape[0]*scale)))
        if tw>search.shape[1] or th>search.shape[0]:continue
        query=cv2.resize(template,(tw,th),interpolation=cv2.INTER_AREA)
        _,similarity,_,location=cv2.minMaxLoc(cv2.matchTemplate(search,query,cv2.TM_CCOEFF_NORMED))
        if not np.isfinite(similarity) or similarity<min_similarity:continue
        px=(left+location[0]-dx*scale)/down;py=(top+location[1]-dy*scale)/down
        fitted=(max(0,int(round(px))),max(0,int(round(py))),
                min(width,int(round(px+(x2-x1)*scale))),min(height,int(round(py+(y2-y1)*scale))))
        if best is None or similarity>best[1]:best=(fitted,float(similarity))
    return best
