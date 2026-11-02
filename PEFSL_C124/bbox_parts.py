"""Proposal experiment using local learned texture votes; no acceptance rights."""
import cv2
import numpy as np

def part_vote_candidates(image,reference,expected_size,max_candidates=4):
    height,width=image.shape[:2];w,h=expected_size
    sx,sy=96./w,96./h
    shape=(round(width*sx),round(height*sy))
    if min(shape)<96 or shape[0]*shape[1]>512*512:return []
    gray=cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),shape),(3,3),.6).astype(np.float32)
    out_shape=(shape[1]-95,shape[0]-95)
    support=np.zeros(out_shape,np.float32);strength=np.zeros(out_shape,np.float32)
    groups=np.zeros((4,*out_shape),np.float32)
    # A tile may move locally under pose/deformation. It supplies a vote,
    # never a measured warp or a sufficient identity proof.
    kernel=np.ones((17,17),np.uint8)
    for x,y,tile in reference['patches']:
        response=cv2.matchTemplate(gray,tile,cv2.TM_CCOEFF_NORMED)
        response=cv2.dilate(response,kernel)
        aligned=response[y:y+out_shape[0],x:x+out_shape[1]]
        valid=np.isfinite(aligned)&(aligned>=.80)
        support+=valid
        strength+=np.where(valid,aligned,0.)
        groups[(y>=48)*2+(x>=48)]+=valid
    available=sum(bool(np.any(v)) for v in groups)
    if available<3:return []
    coverage=(groups>=1).sum(axis=0)
    ranking=np.where((support>=8)&(coverage>=3),support+strength/np.maximum(1,support),-1.)
    candidates=[]
    for _ in range(max_candidates):
        _,score,_,(x,y)=cv2.minMaxLoc(ranking)
        if score<0:break
        box=(max(0,round(x/sx)),max(0,round(y/sy)),min(width,round((x+96)/sx)),min(height,round((y+96)/sy)))
        candidates.append((box,dict(tile_votes=int(support[y,x]),quadrants=int(coverage[y,x]),
            mean_ncc=float(strength[y,x]/max(1,support[y,x])),reference=int(reference['reference']),
            flip=bool(reference['flip']),rotation=int(reference['rotation']))))
        ranking[max(0,y-24):y+25,max(0,x-24):x+25]=-1.
    return candidates
