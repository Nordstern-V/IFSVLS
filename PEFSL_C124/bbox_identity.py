"""Spatial registration checks, independent of recent tracking templates."""
from functools import lru_cache

import cv2
import numpy as np


def _gray(image, size):
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), .8).astype(np.float32)


def registration_patterns(images):
    """Cache inner learned texture at several scales, orientations and aspects.

    Keep a substantial patch: tiny individual edges are not identity evidence.
    Uniform registration images cannot supply a useful correlation template.
    """
    patterns = []
    for image in images:
        if image is None or image.size == 0:
            continue
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        for flip in (False, True):
            for rotation in range(4):
                view = np.ascontiguousarray(np.rot90(image[:, ::-1] if flip else image, rotation))
                inner = view[4:-4, 4:-4]
                if float(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).std()) < 8.:
                    continue
                for side in (32, 40, 48, 56, 64):
                    for aspect in (.75, 1., 1.33):
                        patterns.append(_gray(inner, (int(side * aspect), side)))
    return patterns


def _distributed_pattern_support(query, pattern, location):
    """Require matching texture across the interior, beyond a shared edge."""
    x,y=location;h,w=pattern.shape;crop=query[y:y+h,x:x+w]
    textured=0;supported=[]
    for row in range(3):
        for col in range(3):
            region=np.s_[row*h//3:(row+1)*h//3,col*w//3:(col+1)*w//3]
            a=pattern[region];b=crop[region]
            if float(a.std())<5.:continue
            textured+=1;a=a-a.mean();b=b-b.mean()
            correlation=float(np.sum(a*b)/max(1e-6,np.linalg.norm(a)*np.linalg.norm(b)))
            if correlation>=.60:supported.append((row,col))
    return (len(supported)>=max(4,int(np.ceil(.4*textured)))
            and len({r for r,c in supported})>=2
            and len({c for r,c in supported})>=2)


@lru_cache(maxsize=128)
def _registration_extent_mask(qh, qw, ph, pw, min_extent, inverse_key):
    """Cache geometry only; current scores and texture are never cached.

    The key includes the complete transform and dtype. Shared masks are
    immutable; the bounded cache is safe for foreground/background readers.
    """
    inverse = (None if inverse_key is None else
               np.frombuffer(inverse_key[1], dtype=inverse_key[0]).reshape(2, 3))
    yy,xx=np.indices((qh-ph+1,qw-pw+1),dtype=np.float32)
    corners=np.stack([np.stack([xx,yy],-1),np.stack([xx+pw,yy],-1),
                      np.stack([xx+pw,yy+ph],-1),np.stack([xx,yy+ph],-1)],-2)
    if inverse is not None:
        corners=corners@inverse[:,:2].T+inverse[:,2]
    valid=np.all((corners>=0)&(corners<=np.float32([qw,qh])),axis=(-2,-1))
    centre=corners.mean(axis=-2,keepdims=True)
    full=centre+(corners-centre)*(64./56.)
    lo=full.min(axis=-2);hi=full.max(axis=-2)
    intersection=np.prod(np.maximum(0,np.minimum(hi,[qw,qh])-np.maximum(lo,0)),axis=-1)
    extent=intersection/np.maximum(1.,np.prod(hi-lo,axis=-1)+qw*qh-intersection)
    valid &= extent>=min_extent
    valid.setflags(write=False)
    return valid


def _extent_match(query, pattern, min_extent=None, inverse=None):
    """Select NCC only among placements with a complete reference extent.

    The templates contain the inner 56/64 of a reference. Filtering after
    minMaxLoc loses a valid larger match whenever a small logo scores higher.
    Coordinates and the inferred rim use the same 96-unit space as the caller.
    Legacy callers keep the original unrestricted maximum.
    """
    ph,pw=pattern.shape;qh,qw=query.shape
    if ph>qh or pw>qw:return None
    if min_extent is not None and inverse is None:
        # Intersection cannot exceed the full inferred reference area.
        if pw*ph*(64./56.)**2/(qw*qh)<min_extent:return None
    scores=cv2.matchTemplate(query,pattern,cv2.TM_CCOEFF_NORMED)
    if min_extent is not None:
        inverse_key = (None if inverse is None else
                       (inverse.dtype.str, inverse.tobytes()))
        valid = _registration_extent_mask(qh, qw, ph, pw, min_extent, inverse_key)
        valid = valid & np.isfinite(scores)
        if not np.any(valid):return None
        scores=np.where(valid,scores,-2.)
    _,score,_,location=cv2.minMaxLoc(scores)
    if min_extent is not None and score>=.68 and not _distributed_pattern_support(query,pattern,location):
        return None
    return (float(score),location) if np.isfinite(score) else None


def match_registration_pattern(image, patterns, min_extent=None):
    """Compare spatial structure inside this candidate, never the whole scene."""
    if not patterns or image is None or image.size == 0:
        return None
    query = _gray(image, (96, 96))
    best = -1.
    patch = None
    selected = None
    for pattern in patterns:
        matched=_extent_match(query,pattern,min_extent)
        if matched is None:continue
        score,location=matched
        if np.isfinite(score) and score > best:
            best = float(score)
            patch = tuple(location) + (int(pattern.shape[1]), int(pattern.shape[0]))
            selected = pattern
        # This is already strong independent support; avoid scanning the rest
        # of a large multi-view bank on every confirmed frame.
        if best >= .90:
            break
    return dict(similarity=best, patch=patch,_hint_pattern=selected,
                _hint_size=(96,96),_hint_angle=0.,_hint_min_extent=min_extent) if patch is not None else None


def recheck_registration_hint(image, hint, angle_delta=0.):
    """Remeasure ONE learned template on current pixels, with original gates.

    A hint selects work only. Never reuse its score, geometry or identity.
    """
    if not hint or image is None or image.size==0:return None
    pattern=hint.get('_hint_pattern');size=hint.get('_hint_size')
    if pattern is None or size is None:return None
    query=_gray(image,tuple(map(int,size)))
    angle=float(hint.get('_hint_angle',0.))+float(angle_delta);inverse=None
    if angle:
        transform=cv2.getRotationMatrix2D((size[0]/2.,size[1]/2.),angle,1.)
        query=cv2.warpAffine(query,transform,tuple(map(int,size)),flags=cv2.INTER_LINEAR)
        inverse=cv2.invertAffineTransform(transform)
    matched=_extent_match(query,pattern,hint.get('_hint_min_extent'),inverse)
    if matched is None:return None
    score,(x,y)=matched;h,w=pattern.shape
    quad=np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    if inverse is not None:quad=cv2.transform(quad[None],inverse)[0]
    if np.any(quad<0) or np.any(quad[:,0]>size[0]) or np.any(quad[:,1]>size[1]):return None
    quad*=np.float32([96./size[0],96./size[1]])
    lo,hi=quad.min(axis=0),quad.max(axis=0)
    return dict(similarity=float(score),patch=tuple(lo)+tuple(hi-lo),quad=quad.tolist(),
                _hint_pattern=pattern,_hint_size=size,_hint_angle=angle,
                _hint_min_extent=hint.get('_hint_min_extent'))


def registered_identity_supported(hog, pattern):
    if pattern is None:
        return hog >= .58
    spatial = pattern['similarity']
    return (hog >= .58 and spatial >= .68) or (hog >= .52 and spatial >= .78)


def colour_registration_features(images, preserve_aspect=False):
    orb=cv2.ORB_create(nfeatures=500,scaleFactor=1.2,nlevels=8,
                      edgeThreshold=7,patchSize=15,fastThreshold=5)
    references=[]
    for view_index,image in enumerate(images):
        h,w=image.shape[:2]
        size=(max(16,round(128*w/max(w,h))),max(16,round(128*h/max(w,h)))) if preserve_aspect else (128,128)
        gray=cv2.cvtColor(cv2.resize(image,size),cv2.COLOR_BGR2GRAY)
        points,descriptors=orb.detectAndCompute(gray,None)
        if descriptors is not None and len(points)>=12:
            entry=(np.float32([p.pt for p in points]),descriptors)
            references.append(entry+(size,view_index) if preserve_aspect else entry)
    return references


def match_colour_geometry(image,references,min_extent=None,max_iterations=2000):
    """Verify learned lettering by distributed, consistent local correspondences."""
    if not references:return None
    gray=cv2.cvtColor(cv2.resize(image,(192,192)),cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create(nfeatures=500,scaleFactor=1.2,nlevels=8,
                      edgeThreshold=7,patchSize=15,fastThreshold=5)
    points,descriptors=orb.detectAndCompute(gray,None)
    if descriptors is None or len(points)<12:return None
    query=np.float32([p.pt for p in points]);best=None
    for entry in references:
        source,features=entry[:2]
        rw,rh=entry[2] if len(entry)>2 else (128,128)
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(features,descriptors,k=2)
        good=sorted((a for pair in pairs if len(pair)==2
                     for a,b in [pair] if a.distance<.75*b.distance and a.distance<=60),
                    key=lambda match:match.distance)
        unique=[];used=set()
        for match in good:
            if match.trainIdx not in used:unique.append(match);used.add(match.trainIdx)
        if len(unique)<12:continue
        src=np.float32([source[m.queryIdx] for m in unique]);dst=np.float32([query[m.trainIdx] for m in unique])
        matrix,inliers=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=max_iterations)
        if matrix is None or inliers is None:continue
        inside=inliers.ravel()>0;count=int(inside.sum());ratio=float(inside.mean())
        if count<12 or ratio<.60 or np.any(np.ptp(src[inside],axis=0)<.30*np.float32([rw,rh])):continue
        if cv2.contourArea(cv2.convexHull(src[inside]))<.06*rw*rh:continue
        predicted=cv2.perspectiveTransform(src[inside][None],matrix)[0]
        error=float(np.median(np.linalg.norm(predicted-dst[inside],axis=1)))
        if error>2.:continue
        # Use the measured registration interior; exclude its unreliable rim.
        if len(entry)>2:
            source_quad=np.float32([[[.1*rw,.1*rh],[.9*rw,.1*rh],[.9*rw,.9*rh],[.1*rw,.9*rh]]])
        else:
            source_quad=np.float32([[[13,13],[114,13],[114,114],[13,114]]])
        quad=cv2.perspectiveTransform(source_quad,matrix)[0]
        if not np.isfinite(quad).all() or np.any(quad<0) or np.any(quad>=192):continue
        if not cv2.isContourConvex(quad) or not .12<=cv2.contourArea(quad)/(192*192)<=.85:continue
        lengths=np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1)
        if lengths.min()<16 or lengths.max()/lengths.min()>3.5:continue
        proof=dict(verified=True,inliers=count,matches=len(unique),ratio=round(ratio,3),
                   median_error=round(error,3),source_coverage=(np.ptp(src[inside],axis=0)/np.float32([rw,rh])).round(3).tolist())
        if len(entry)>3:proof['reference_view']=int(entry[3])
        if len(entry)>2:
            full=cv2.perspectiveTransform(np.float32([[[0,0],[rw-1,0],[rw-1,rh-1],[0,rh-1]]]),matrix)[0]
            if not np.isfinite(full).all() or not cv2.isContourConvex(full):continue
            lo,hi=full.min(axis=0),full.max(axis=0)
            area=float(np.prod(hi-lo))
            intersection=float(np.prod(np.maximum(0,np.minimum(hi,192)-np.maximum(lo,0))))
            proof['extent_iou']=round(intersection/max(1.,area+192*192-intersection),3)
        if min_extent is not None and proof.get('extent_iou',0.)<min_extent:continue
        if best is None or count>best['geometry']['inliers']:
            lower=quad.min(axis=0)/2.;upper=quad.max(axis=0)/2.
            best=dict(geometry=proof,quad=(quad/2.).tolist(),patch=tuple(lower)+tuple(upper-lower))
            if len(entry)>2:
                best['full_quad']=(full/2.).tolist()
    return best


def match_colour_registration_pattern(image, patterns, geometry_refs=(), min_extent=None,
                                      rotation_limit=8):
    """Keep a cover's aspect when comparing a learned interior registration.

    A square text registration and a tall complete cover must not stretch the
    letters differently. Retain the original square comparison as well, and
    report the winning patch in the caller's existing 96-unit coordinates.
    White-object matching continues to use the original function exclusively.
    """
    best=match_registration_pattern(image,patterns,min_extent)
    if best is None and min_extent is None:
        return best
    if image is None or image.size==0:return None
    if best is None:best=dict(similarity=-1.,patch=(0.,0.,96.,96.))
    geometric=match_colour_geometry(image,geometry_refs,
                                    min_extent=.75 if min_extent is not None else None)
    if geometric is not None and (min_extent is None or
            geometric.get('geometry',{}).get('extent_iou',0.)>=.75):
        best.update(geometric)
        return best
    if best['patch'][2]*best['patch'][3]<.15*96*96:
        best=dict(similarity=-1.,patch=(0.,0.,96.,96.))
    if best['similarity']>=.90:return best
    h,w=image.shape[:2]
    size=(max(64,int(round(128*w/max(w,h)))),max(64,int(round(128*h/max(w,h)))))
    query=_gray(image,size)
    ranked=[]
    for pattern in patterns:
        ph,pw=pattern.shape
        if ph>query.shape[0] or pw>query.shape[1] or pw*ph<.15*query.size:
            continue
        matched=_extent_match(query,pattern,min_extent)
        if matched is None:continue
        score,location=matched
        if np.isfinite(score):ranked.append((float(score),len(ranked),pattern))
        if np.isfinite(score) and score>best['similarity']:
            best=dict(similarity=float(score),
                      patch=(location[0]*96./size[0],location[1]*96./size[1],
                             pw*96./size[0],ph*96./size[1]),aspect_preserved=True,
                      _hint_pattern=pattern,_hint_size=size,_hint_angle=0.,_hint_min_extent=min_extent)
        if best['similarity']>=.90:
            break
    # Covers rotate by a few degrees during handheld use. Quarter-turn-only
    # templates miss those poses. Recheck just the best eight template shapes
    # on small rotations, and map every winning corner back to real pixels.
    if best['similarity']<.85:
        selected=[item[2] for item in sorted(ranked,key=lambda item:item[0],reverse=True)[:rotation_limit]]
        for angle in (-8.,8.,-16.,16.):
            transform=cv2.getRotationMatrix2D((size[0]/2.,size[1]/2.),angle,1.)
            rotated=cv2.warpAffine(query,transform,size,flags=cv2.INTER_LINEAR)
            inverse=cv2.invertAffineTransform(transform)
            for pattern in selected:
                ph,pw=pattern.shape
                matched=_extent_match(rotated,pattern,min_extent,inverse)
                if matched is None:continue
                score,location=matched
                if not np.isfinite(score) or score<=best['similarity']:continue
                x,y=location
                corners=np.float32([[x,y],[x+pw,y],[x+pw,y+ph],[x,y+ph]])
                original=cv2.transform(corners[None],inverse)[0]
                if np.any(original<0) or np.any(original[:,0]>=size[0]) or np.any(original[:,1]>=size[1]):continue
                quad=original*np.float32([96./size[0],96./size[1]])
                lower=quad.min(axis=0);upper=quad.max(axis=0)
                best=dict(similarity=float(score),patch=tuple(lower)+tuple(upper-lower),
                          quad=quad.tolist(),aspect_preserved=True,rotation=angle,
                          _hint_pattern=pattern,_hint_size=size,_hint_angle=angle,_hint_min_extent=min_extent)
            if best['similarity']>=.88:break
    return best
