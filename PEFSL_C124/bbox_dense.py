"""Bounded whole-reference alignment for newly registered objects.

No object names, learned colour assumptions or recent tracking templates are
used. This is independent identity evidence; callers still apply feature,
colour, background and explicit-negative vetoes on the current frame.
"""
import threading
import time
import cv2
import numpy as np

_STATE = threading.local()
_SIDE = 80

# A deadline exit is not a completed negative identity measurement. Callers
# must retry this reference on current pixels instead of recording a failure.
IDENTITY_INCOMPLETE = object()


def _gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (_SIDE, _SIDE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (3, 3), .6).astype(np.float32)


def _unit(image, mask=None):
    values = image.ravel() if mask is None else image[mask]
    values = values.astype(np.float32)
    values -= values.mean()
    length = float(np.linalg.norm(values))
    return values / max(length, 1e-6)


def _hog(gray):
    if not hasattr(_STATE, 'hog'):
        _STATE.hog = cv2.HOGDescriptor((96,64),(16,16),(8,8),(8,8),9)
    image = cv2.resize(np.clip(gray, 0, 255).astype(np.uint8), (96,64))
    vector = _STATE.hog.compute(image).ravel()
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def dense_reference_features(views):
    references = []
    for index, image in enumerate(views):
        for flip in (False, True):
            for rotation in range(4):
                view = np.ascontiguousarray(np.rot90(image[:, ::-1] if flip else image, rotation))
                gray = _gray(view)
                if float(gray.std()) < 8.:
                    continue
                points=cv2.goodFeaturesToTrack(gray,maxCorners=80,qualityLevel=.02,
                                               minDistance=3,blockSize=3)
                references.append(dict(gray=gray, unit=_unit(gray), hog=_hog(gray),
                                       points=points,
                                       reference=index, flip=flip, rotation=rotation))
    return references


def indexed_registration_patterns(image,references,views,cache,groups=3):
    """Select reference orientations cheaply; full search remains a fallback.

    Pattern construction matches the original 64-pixel registration templates.
    This only changes which groups a time-critical tracking pass tries first.
    """
    query=_gray(image);unit=_unit(query);descriptor=_hog(query)
    ranked=sorted(((.6*float(ref['unit']@unit)+.4*float(ref['hog']@descriptor),i)
                   for i,ref in enumerate(references)),reverse=True)
    out=[]
    for _,index in ranked[:groups]:
        ref=references[index];key=(ref['reference'],ref['flip'],ref['rotation'])
        if key not in cache:
            square=cv2.resize(views[ref['reference']],(64,64),interpolation=cv2.INTER_AREA)
            view=np.ascontiguousarray(np.rot90(square[:,::-1] if ref['flip'] else square,ref['rotation']))
            inner=view[4:-4,4:-4];patterns=[]
            if float(cv2.cvtColor(inner,cv2.COLOR_BGR2GRAY).std())>=8.:
                for side in (32,40,48,56,64):
                    for aspect in (.75,1.,1.33):
                        small=cv2.resize(inner,(int(side*aspect),side),interpolation=cv2.INTER_AREA)
                        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
                        patterns.append(cv2.GaussianBlur(gray,(3,3),.8).astype(np.float32))
            cache[key]=patterns
        out.extend(cache[key])
    return out


def _verify_dense_warp(query, ref, warp):
    """Common acceptance gates for direct and ECC-proposed current alignments."""
    corners = np.float32([[0,0],[_SIDE-1,0],[_SIDE-1,_SIDE-1],[0,_SIDE-1]])
    if not np.isfinite(warp).all() or np.linalg.det(warp[:,:2]) <= 0:
        return None
    scales = np.linalg.svd(warp[:,:2], compute_uv=False)
    if scales.min() < .75 or scales.max() > 1.3 or scales.max()/scales.min() > 1.4:
        return None
    quad = cv2.transform(corners[None], warp)[0]
    lo, hi = quad.min(axis=0), quad.max(axis=0)
    area = float(np.prod(hi-lo))
    intersection = float(np.prod(np.maximum(0, np.minimum(hi, _SIDE)-np.maximum(lo, 0))))
    extent = intersection/max(1., area+_SIDE*_SIDE-intersection)
    if extent < .78:
        return None
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(query, warp, (_SIDE, _SIDE), flags=flags)
    valid = cv2.warpAffine(np.ones_like(query), warp, (_SIDE, _SIDE), flags=flags) > .99
    if valid.mean() < .82:
        return None
    correlation = float(_unit(ref['gray'], valid) @ _unit(aligned, valid))
    if correlation < .82:
        return None
    # Invalid pixels cannot improve HOG at the rim.
    aligned[~valid] = ref['gray'][~valid]
    appearance = float(_hog(aligned) @ ref['hog'])
    if appearance < .80:
        return None
    informative = []; supported = []
    for ty in range(3):
        for tx in range(3):
            ys = slice(round(ty*_SIDE/3), round((ty+1)*_SIDE/3))
            xs = slice(round(tx*_SIDE/3), round((tx+1)*_SIDE/3))
            region = valid[ys,xs]
            template = ref['gray'][ys,xs]
            if region.mean() < .7 or float(template[region].std()) < 7.:
                continue
            informative.append((tx,ty))
            score = float(_unit(template, region) @ _unit(aligned[ys,xs], region))
            if score >= .55:
                supported.append((tx,ty))
    if (len(informative)<4 or len(supported)<max(4, int(np.ceil(.7*len(informative))))
            or len({p[0] for p in supported})<2 or len({p[1] for p in supported})<2):
        return None
    proof = dict(verified=True, correlation=round(correlation,4), aligned_hog=round(appearance,4),
                 extent_iou=round(extent,4), valid_fraction=round(float(valid.mean()),4),
                 supported_tiles=len(supported), informative_tiles=len(informative),
                 reference=ref['reference'], rotation=ref['rotation'], flip=ref['flip'])
    result = dict(similarity=correlation, dense=proof, full_quad=(quad*96./_SIDE).tolist(),
                  patch=(0.,0.,96.,96.))
    supported_points=[]
    if ref['points'] is not None:
        for point in ref['points'].reshape(-1,2):
            x,y=map(lambda v:int(round(float(v))),point)
            if not (3<=x<_SIDE-3 and 3<=y<_SIDE-3):continue
            mask=valid[y-2:y+3,x-2:x+3]
            template=ref['gray'][y-2:y+3,x-2:x+3]
            if not mask.all() or float(template.std())<7.:continue
            if float(_unit(template)@_unit(aligned[y-2:y+3,x-2:x+3]))>=.65:
                supported_points.append(point)
    if len(supported_points)>=6:
        observed=cv2.transform(np.float32(supported_points)[None],warp)[0]
        result['motion_points']=(observed*96./_SIDE).tolist()
    proof['motion_points']=len(supported_points)
    return result


def recheck_dense_hint(image,references,hint):
    """Revalidate a previous affine proposal against learned and CURRENT pixels."""
    proof={} if hint is None else hint.get('dense',{})
    if not proof or image is None or image.size==0:return None
    ref=next((r for r in references if all(r.get(k)==proof.get(k)
        for k in ('reference','rotation','flip'))),None)
    if ref is None:return None
    quad=np.asarray(hint.get('full_quad',[]),np.float32)
    if quad.shape!=(4,2):return None
    source=np.float32([[0,0],[_SIDE-1,0],[0,_SIDE-1]])
    warp=cv2.getAffineTransform(source,quad[[0,1,3]]*(_SIDE/96.))
    return _verify_dense_warp(_gray(image),ref,warp)


def match_dense_reference(image, references, max_attempts=3, max_iterations=24, prefer_direct=False):
    """Align complete ROIs; refuse local logos, flat scenes and extreme warps.

    ECC only proposes an affine alignment. Acceptance uses the remeasured
    whole-image correlation, aligned HOG, distributed textured tile support,
    valid pixels and projected complete extent. A high ECC return value alone
    never confirms identity.
    """
    if not references or image is None or min(image.shape[:2], default=0) < 20:
        return None
    query = _gray(image)
    if float(query.std()) < 8.:
        return None
    unit = _unit(query)
    ranked = sorted(((float(ref['unit'] @ unit), index) for index, ref in enumerate(references)),
                    reverse=True)
    best = None
    if prefer_direct:
        # Optical flow has already supplied the current crop. A direct match
        # against LEARNED pixels may pass the same gates without iterative ECC.
        for initial, index in ranked[:max_attempts]:
            if initial < .82:
                continue
            result = _verify_dense_warp(query, references[index], np.eye(2,3,dtype=np.float32))
            if result is not None and (best is None or result['similarity'] > best['similarity']):
                best = result
        if best is not None:
            return best
    if max_iterations<=0:
        return best
    for initial, index in ranked[:max_attempts]:
        if initial < .25:
            continue
        ref = references[index]
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ref['gray']/255., query/255., warp,
                cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, max_iterations, .001),
                None, 3)
        except cv2.error:
            continue
        result = _verify_dense_warp(query, ref, warp)
        if result is None:
            continue
        correlation = result['similarity']
        appearance = result['dense']['aligned_hog']
        if best is None or correlation > best['similarity']:
            best = result
        if correlation >= .94 and appearance >= .9:
            break
    return best


# Distributed correspondence supplements whole-image ECC when a changed rim
# or partial occlusion disrupts it. The projected complete reference must still
# agree with the candidate. No object names or newly pretrained weights.
def _spatial_gray(image):
    return cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(96,96)),(3,3),.6).astype(np.float32)

def spatial_reference_features(views,patch_size=24):
    if patch_size not in (16,24):raise ValueError('unsupported spatial patch size')
    result=[]
    for vi,image in enumerate(views):
        for flip in (False,True):
            for rotation in range(4):
                view=np.ascontiguousarray(np.rot90(image[:,::-1] if flip else image,rotation))
                g=_spatial_gray(view);patches=[]
                stride=3*patch_size//4
                for y in range(0,97-patch_size,stride):
                    for x in range(0,97-patch_size,stride):
                        tile=g[y:y+patch_size,x:x+patch_size]
                        if tile.std()>=8.:patches.append((x,y,tile.copy()))
                result.append(dict(gray=g,hog=_hog(g),patches=patches,reference=vi,flip=flip,rotation=rotation,
                                   patch_size=patch_size))
    return result

def recheck_spatial_hint(image,hint,deadline=None):
    reference=None if hint is None else hint.get('_hint_spatial_ref')
    if reference is None:return None
    if reference.get('patch_size',24)==16:
        return match_fine_spatial_reference(image,[reference],deadline=deadline)
    return match_spatial_reference(image,[reference],deadline=deadline)


def _spatial_choice_indices(appearance,refs):
    ranked=sorted(((float(appearance@r['hog']),i) for i,r in enumerate(refs)),reverse=True)
    chosen=[i for score,i in ranked[:3] if score>=.60]
    lookup={(r['reference'],r['flip'],r['rotation']):i for i,r in enumerate(refs)}
    for i in tuple(chosen):
        ref=refs[i];partner=lookup.get((ref['reference'],not ref['flip'],ref['rotation']))
        if partner is not None and partner not in chosen and float(appearance@refs[partner]['hog'])>=.60:
            chosen.append(partner)
    return chosen


def spatial_reference_choices(image,refs):
    """Select learned choices only; do not measure or certify correspondences."""
    if image is None or image.size==0 or min(image.shape[:2])<16 or not refs:return []
    return [refs[i] for i in _spatial_choice_indices(_hog(_spatial_gray(image)),refs)]


def match_fine_spatial_reference(image,refs,deadline=None,stop_on_first=False):
    """Supplement failed large tiles with twelve fresh consistent small tiles.

    First-pass matches propose geometry only. Each final correspondence is
    remeasured near that prediction on the current crop. Full learned extent,
    distributed support and affine bounds remain required.
    """
    if image is None or image.size==0 or min(image.shape[:2])<16 or not refs:return None
    query=_spatial_gray(image);appearance=_hog(query)
    chosen=_spatial_choice_indices(appearance,refs)
    best=None
    for index in chosen:
        ref=refs[index]
        if ref.get('patch_size')!=16:raise ValueError('fine spatial reference must use 16 pixel tiles')
        source=[];observed=[]
        for x,y,tile in ref['patches']:
            if deadline is not None and time.perf_counter()>=deadline:
                return best if best is not None else IDENTITY_INCOMPLETE
            left,top=max(0,x-12),max(0,y-12);right,bottom=min(96,x+28),min(96,y+28)
            _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(query[top:bottom,left:right],tile,cv2.TM_CCOEFF_NORMED))
            if score>=.80:source.append([x+8,y+8]);observed.append([left+loc[0]+8,top+loc[1]+8])
        if len(source)<4:continue
        initial,mask=cv2.estimateAffine2D(np.float32(source),np.float32(observed),method=cv2.RANSAC,
            ransacReprojThreshold=2.5,maxIters=500,confidence=.99)
        if initial is None or mask is None or not np.isfinite(initial).all() or int(mask.sum())<4:continue
        source=[];observed=[];scores=[]
        for x,y,tile in ref['patches']:
            if deadline is not None and time.perf_counter()>=deadline:
                return best if best is not None else IDENTITY_INCOMPLETE
            prediction=initial@np.float64([x+8,y+8,1.])-[8,8]
            left=max(0,x-12,int(np.ceil(prediction[0]-3)));top=max(0,y-12,int(np.ceil(prediction[1]-3)))
            right=min(96,x+28,int(np.floor(prediction[0]+3))+16)
            bottom=min(96,y+28,int(np.floor(prediction[1]+3))+16)
            if right-left<16 or bottom-top<16:continue
            _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(query[top:bottom,left:right],tile,cv2.TM_CCOEFF_NORMED))
            if score>=.80:
                source.append([x+8,y+8]);observed.append([left+loc[0]+8,top+loc[1]+8]);scores.append(score)
        if len(source)<12:continue
        source=np.float32(source);observed=np.float32(observed)
        matrix,mask=cv2.estimateAffine2D(source,observed,method=cv2.RANSAC,
            ransacReprojThreshold=2.5,maxIters=500,confidence=.99)
        if matrix is None or mask is None or not np.isfinite(matrix).all():continue
        inside=mask.ravel()>0;count=int(inside.sum())
        if count<12 or inside.mean()<.70:continue
        coverage=np.ptp(source[inside],axis=0)/96.
        if coverage.min()<.35:continue
        scales=np.linalg.svd(matrix[:,:2],compute_uv=False)
        if np.linalg.det(matrix[:,:2])<=0 or scales.min()<.75 or scales.max()>1.3 or scales.max()/scales.min()>1.4:continue
        quad=cv2.transform(np.float32([[[0,0],[96,0],[96,96],[0,96]]]),matrix)[0]
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        intersection=float(np.prod(np.maximum(0,np.minimum(hi,96)-np.maximum(lo,0))))
        extent=intersection/max(1.,float(np.prod(hi-lo))+96*96-intersection)
        if extent<.78:continue
        proof=dict(verified=True,reference=ref['reference'],flip=ref['flip'],rotation=ref['rotation'],
            matches=len(source),inliers=count,coverage=coverage.tolist(),extent=extent,
            mean_ncc=float(np.asarray(scores)[inside].mean()),full_quad=quad.tolist(),patch_size=16,
            source_points=source[inside].tolist(),observed_points=observed[inside].tolist())
        if best is None or count>best['spatial']['inliers']:
            best=dict(similarity=proof['mean_ncc'],spatial=proof,full_quad=quad.tolist(),
                      patch=(0,0,96,96),_hint_spatial_ref=ref)
            if stop_on_first:return best
    return best


def spatial_scene_candidates(image,refs,expected_size,deadline=None):
    """Propose learned ROIs from scene tile votes, then verify each fresh crop.

    Scene correspondences only locate a candidate. Admission still requires
    the stricter fine spatial matcher; callers retain feature and veto gates.
    """
    if image is None or image.size==0 or not refs:return []
    if len(expected_size)!=2 or not np.isfinite(expected_size).all() or min(expected_size)<=0:return []
    height,width=image.shape[:2];queries={};ranked=[]
    def expired():return deadline is not None and time.perf_counter()>=deadline
    for index,ref in enumerate(refs):
        if expired():return []
        parity=ref['rotation']%2
        if parity not in queries:
            w,h=expected_size[::-1] if parity else expected_size
            sx,sy=96./w,96./h
            shape=(int(round(width*sx)),int(round(height*sy)))
            # Tiny or extremely thin learned ROIs keep the existing search
            # fallback; never allocate an unbounded enlarged camera frame.
            if min(shape)<96 or shape[0]*shape[1]>512*512:continue
            gray=cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),shape),(3,3),.6).astype(np.float32)
            queries[parity]=(gray,sx,sy)
        query,_,_=queries[parity]
        score=float(cv2.minMaxLoc(cv2.matchTemplate(query,ref['gray'],cv2.TM_CCOEFF_NORMED))[1])
        ranked.append((score,index))
    chosen=[i for _,i in sorted(ranked,reverse=True)[:2]]
    lookup={(r['reference'],r['flip'],r['rotation']):i for i,r in enumerate(refs)}
    for index in tuple(chosen):
        ref=refs[index];partner=lookup.get((ref['reference'],not ref['flip'],ref['rotation']))
        if partner is not None and partner not in chosen:chosen.append(partner)
    for index in chosen:
        ref=refs[index];query,sx,sy=queries[ref['rotation']%2];source=[];observed=[]
        for x,y,tile in ref['patches']:
            if expired():return []
            response=cv2.matchTemplate(query,tile,cv2.TM_CCOEFF_NORMED)
            for _ in range(3):
                _,score,_,loc=cv2.minMaxLoc(response)
                if score<.80:break
                source.append([x+8,y+8]);observed.append([loc[0]+8,loc[1]+8])
                lx,ly=loc;response[max(0,ly-8):ly+9,max(0,lx-8):lx+9]=-1
        if len(source)<8:continue
        matrix,mask=cv2.estimateAffine2D(np.float32(source),np.float32(observed),method=cv2.RANSAC,
            ransacReprojThreshold=2.5,maxIters=2500,confidence=.99)
        if matrix is None or mask is None or not np.isfinite(matrix).all():continue
        inside=mask.ravel()>0
        if inside.sum()<8 or np.ptp(np.float32(source)[inside],axis=0).min()/96.<.35:continue
        scales=np.linalg.svd(matrix[:,:2],compute_uv=False)
        if np.linalg.det(matrix[:,:2])<=0 or scales.min()<.75 or scales.max()>1.3 or scales.max()/scales.min()>1.4:continue
        quad=cv2.transform(np.float32([[[0,0],[96,0],[96,96],[0,96]]]),matrix)[0]/[sx,sy]
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        x,y=np.floor(lo).astype(int);r,b=np.ceil(hi).astype(int)
        if x<0 or y<0 or r>width or b>height or min(r-x,b-y)<24:continue
        current=match_fine_spatial_reference(image[y:b,x:r],[ref],deadline=deadline)
        if current is IDENTITY_INCOMPLETE:return []
        if current is not None:return [((int(x),int(y),int(r),int(b)),current)]
    return []


def recheck_learned_points(image,hint,deadline=None):
    """Remeasure selected LEARNING corners, not a previous camera template."""
    ref=hint.get('_hint_point_ref')
    quad=hint.get('_hint_dense_seed',hint).get('full_quad')
    if ref is None or quad is None or image is None or image.size==0:return None
    quad=np.asarray(quad,np.float32)
    if quad.shape!=(4,2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):return None
    if deadline is not None and time.perf_counter()>=deadline:return IDENTITY_INCOMPLETE
    previous=ref['gray'];points=ref['points']
    if points is None or len(points)<12:return None
    query=cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(96,96)),(3,3),.6)
    initial=cv2.getPerspectiveTransform(np.float32([[0,0],[95,0],[95,95],[0,95]]),np.float32(quad))
    # Rectify the CURRENT crop to learned coordinates before LK. An initial
    # point prediction alone cannot undo perspective inside each LK window.
    # The seed only proposes the warp; current pixels must still pass every
    # forward/backward, texture, correspondence and complete-extent check.
    if not np.isfinite(initial).all() or abs(np.linalg.det(initial))<1e-8:return None
    rectified=cv2.warpPerspective(query,initial,(96,96),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    rectified_valid=cv2.warpPerspective(np.ones((96,96),np.uint8),initial,(96,96),
        flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)
    predicted=points.copy()
    options=dict(winSize=(15,15),maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,12,.03))
    observed,status,_=cv2.calcOpticalFlowPyrLK(previous,rectified,points,predicted,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**options)
    if observed is None:return None
    back,back_status,_=cv2.calcOpticalFlowPyrLK(rectified,previous,observed,None,**options)
    if back is None:return None
    src=points.reshape(-1,2);dst=observed.reshape(-1,2)
    good=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(src-back.reshape(-1,2),axis=1)<1.)
    good &= np.isfinite(dst).all(axis=1)&(dst.min(axis=1)>=4)&(dst.max(axis=1)<92)
    for i in np.flatnonzero(good):
        x,y=np.rint(dst[i]).astype(int)
        if not rectified_valid[y-4:y+5,x-4:x+5].all():good[i]=False
    if good.sum()<12:return None
    source=src[good];target=dst[good]
    matrix,mask=cv2.findHomography(source,target,cv2.RANSAC,1.5,maxIters=300,confidence=.99)
    if matrix is None or mask is None:return None
    inside=mask.ravel()>0
    if inside.sum()<12 or inside.mean()<.65:return None
    matrix=initial@matrix
    aligned=cv2.warpPerspective(query,matrix,(96,96),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    valid=cv2.warpPerspective(np.ones((96,96),np.uint8),matrix,(96,96),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)>0
    supported=[];scores=[]
    for i in np.flatnonzero(inside):
        x,y=np.rint(source[i]).astype(int)
        if not(4<=x<92 and 4<=y<92) or not valid[y-4:y+5,x-4:x+5].all():continue
        a=previous[y-4:y+5,x-4:x+5].astype(np.float32);b=aligned[y-4:y+5,x-4:x+5].astype(np.float32)
        if a.std()<5.:continue
        a-=a.mean();b-=b.mean();score=float(np.sum(a*b)/max(1e-6,np.linalg.norm(a)*np.linalg.norm(b)))
        if score>=.80:supported.append(i);scores.append(score)
    if len(supported)<12 or len(supported)/len(source)<.65:return None
    coverage=np.ptp(source[supported],axis=0)/96.
    if coverage.min()<.30:return None
    full=cv2.perspectiveTransform(np.float32([[[0,0],[95,0],[95,95],[0,95]]]),matrix)[0]
    if not np.isfinite(full).all() or not cv2.isContourConvex(full):return None
    lo,hi=full.min(axis=0),full.max(axis=0)
    intersection=float(np.prod(np.maximum(0,np.minimum(hi,96)-np.maximum(lo,0))))
    extent=intersection/max(1.,float(np.prod(hi-lo))+96*96-intersection)
    if extent<.78:return None
    proof=dict(verified=True,inliers=len(supported),ratio=len(supported)/len(source),
        source_coverage=coverage.tolist(),extent_iou=extent,mean_ncc=float(np.mean(scores)))
    metadata={k:v for k,v in hint.items() if k.startswith('_hint_')}
    if metadata.get('_hint_dense_ref') is not None:
        metadata['_hint_dense_seed']=dict(hint.get('_hint_dense_seed',{}),full_quad=full.tolist())
    else:
        # full_quad below remains available for the learned-point path;
        # an affine/dense seed is reusable only together with its reference.
        metadata.pop('_hint_dense_seed',None)
    return dict(metadata,similarity=float(np.mean(scores)),geometry=proof,full_quad=full.tolist(),
        patch=tuple(lo)+tuple(hi-lo),
        motion_points=cv2.perspectiveTransform(target[supported][None],initial)[0].tolist())


def match_spatial_reference(image,refs,deadline=None):
    if image is None or image.size == 0 or min(image.shape[:2]) < 16 or not refs:
        return None
    q=_spatial_gray(image);h=_hog(q)
    ranked=sorted(((float(h@r['hog']),i) for i,r in enumerate(refs)),reverse=True)
    best=None
    for appearance,index in ranked[:3]:
        if deadline is not None and time.perf_counter()>=deadline:return IDENTITY_INCOMPLETE
        if appearance<.60:continue
        ref=refs[index];source=[];observed=[];scores=[]
        for x,y,tile in ref['patches']:
            if deadline is not None and time.perf_counter()>=deadline:return IDENTITY_INCOMPLETE
            left,top=max(0,x-12),max(0,y-12)
            right,bottom=min(96,x+36),min(96,y+36)
            response=cv2.matchTemplate(q[top:bottom,left:right],tile,cv2.TM_CCOEFF_NORMED)
            _,score,_,location=cv2.minMaxLoc(response)
            if score>=.80:
                source.append([x+12,y+12]);observed.append([left+location[0]+12,top+location[1]+12]);scores.append(score)
        item=dict(hog=appearance,matches=len(source),reference=ref['reference'],flip=ref['flip'],rotation=ref['rotation'])
        if len(source)<8:continue
        source=np.float32(source);observed=np.float32(observed)
        transform,inliers=cv2.estimateAffine2D(source,observed,method=cv2.RANSAC,
            ransacReprojThreshold=2.5,maxIters=500,confidence=.99)
        if transform is None:continue
        inside=inliers.ravel()>0;count=int(inside.sum());item['inliers']=count
        coverage=np.ptp(source[inside],axis=0)/96.
        if count<8 or inside.mean()<.70 or coverage.min()<.35:continue
        scales=np.linalg.svd(transform[:,:2],compute_uv=False)
        if np.linalg.det(transform[:,:2])<=0 or scales.min()<.75 or scales.max()>1.3 or scales.max()/scales.min()>1.4:continue
        full=cv2.transform(np.float32([[[0,0],[96,0],[96,96],[0,96]]]),transform)[0]
        lo,hi=full.min(axis=0),full.max(axis=0)
        intersection=float(np.prod(np.maximum(0,np.minimum(hi,96)-np.maximum(lo,0))))
        extent=intersection/max(1.,float(np.prod(hi-lo))+96*96-intersection);item['extent']=extent
        if extent<.78:continue
        item.update(verified=True,coverage=coverage.tolist(),mean_ncc=float(np.asarray(scores)[inside].mean()),
                    full_quad=full.tolist(),source_points=source[inside].tolist(),observed_points=observed[inside].tolist())
        if best is None or item['mean_ncc']>best['mean_ncc']:
            best=item
            best_reference=ref
    if best is None:
        return None
    # Layout patch centres are not optical-flow corners. Keep the original
    # corner-based tracker; this weaker evidence supplements identity only.
    return dict(similarity=best['mean_ncc'],spatial=best,
                full_quad=best['full_quad'],patch=(0,0,96,96),_hint_spatial_ref=best_reference)
