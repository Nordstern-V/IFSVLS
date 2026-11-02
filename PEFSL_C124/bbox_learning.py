"""Conservative on-site capture; motion is evidence of location, not identity."""
from dataclasses import dataclass
import time

import cv2
import numpy as np

from bbox_tracking import seed_points, track_points, reference_sharpness, relocate_template


@dataclass
class CaptureDecision:
    box: tuple
    action: str
    reason: str = ''


class RegistrationCapture:
    """Follow the labelled region, refusing uncertain motion and redundant crops.

    This does not segment an object from a hand or infer unseen views. Failure
    never resets the anchor to arbitrary current pixels. ESC/reselection is
    required after sustained loss. Duplicate observations may finish a static
    capture, but create no extra positive features.
    """
    def __init__(self, box):
        self.box = tuple(map(int, box))
        self.gray = self.points = self.corners = None
        self.best_sharpness = 0.
        self.appearances = []
        self.failures = 0
        self.accepted = self.duplicates = self.skipped = 0
        self.lost = False
        self.failure_since = None
        self.last_recovery_attempt = float('-inf')
        self.last_crop = None
        self.initial_area = max(1, (box[2]-box[0])*(box[3]-box[1]))

    def _pause(self, reason, motion_failure=False):
        self.skipped += 1
        if motion_failure:
            self.failures += 1
            now = time.perf_counter()
            if self.failure_since is None:
                self.failure_since = now
            self.lost = now-self.failure_since >= 1.5
        return CaptureDecision(self.box, 'pause', reason)

    def _same_content(self, frame, box):
        """Verify a whole recent crop, allowing a modest exposure offset.

        Neither a matching mean brightness nor a small logo authorizes new
        learning pixels. Require distributed texture and retained colour.
        Textureless regions only get a near-unchanged-pixel static fallback.
        """
        if self.last_crop is None:
            return False
        x,y,r,b=box
        if x<0 or y<0 or r>frame.shape[1] or b>frame.shape[0] or min(r-x,b-y)<12:
            return False
        crop=cv2.resize(frame[y:b,x:r],(48,48),interpolation=cv2.INTER_AREA)
        before=cv2.GaussianBlur(cv2.cvtColor(self.last_crop,cv2.COLOR_BGR2GRAY),(3,3),.8).astype(np.float32)
        current=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(3,3),.8).astype(np.float32)
        std=float(before.std())
        if std<6.:
            # With no structure there is no evidence to follow movement or
            # compensate exposure. Permit only a genuinely unchanged crop.
            a=cv2.GaussianBlur(self.last_crop,(3,3),.8).astype(np.float32)
            c=cv2.GaussianBlur(crop,(3,3),.8).astype(np.float32)
            return float(np.abs(c-a).mean())<=2.
        a=before-before.mean();c=current-current.mean()
        gain=float(np.sum(a*c)/max(1e-6,float(np.sum(a*a))))
        correlation=float(np.sum(a*c)/max(1e-6,float(np.linalg.norm(a)*np.linalg.norm(c))))
        if not .65<=gain<=1.5 or correlation<.94:
            return False
        if float(np.abs(c-gain*a).mean())>max(3.,.15*std):
            return False
        supported=[];textured=0
        for row in range(3):
            for col in range(3):
                part=np.s_[row*16:(row+1)*16,col*16:(col+1)*16]
                a=before[part];c=current[part]
                if float(a.std())<5.:continue
                textured+=1;a=a-a.mean();c=c-c.mean()
                score=float(np.sum(a*c)/max(1e-6,float(np.linalg.norm(a)*np.linalg.norm(c))))
                if score>=.80:supported.append((row,col))
        if (len(supported)<max(4,int(np.ceil(.6*textured)))
                or len({r for r,c in supported})<2 or len({c for r,c in supported})<2):
            return False
        # Hue/saturation preserve category colour under moderate illumination
        # changes. They do not require a specific object or a white surface.
        def histogram(image):
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            # Hue is unstable on white/gray or very dark pixels.
            hsv[...,0][(hsv[...,1]<32)|(hsv[...,2]<32)]=0
            hist=cv2.calcHist([hsv],[0,1],None,[8,8],[0,180,0,256])
            return hist/max(float(hist.sum()),1.)
        return cv2.compareHist(histogram(self.last_crop),histogram(crop),cv2.HISTCMP_BHATTACHARYYA)<=.25

    def _recover(self, frame, gray, motion):
        """Propose a nearby location; full-crop evidence must verify it."""
        if self._same_content(frame,self.box):
            return self.box
        valid_motion=False
        if motion is not None:
            x,y,r,b=motion.box
            area=max(1,(r-x)*(b-y))
            old_area=max(1,(self.box[2]-self.box[0])*(self.box[3]-self.box[1]))
            corners=motion.corners
            valid_motion=(corners is not None and np.isfinite(corners).all()
                          and (corners>=0).all()
                          and (corners[:,0]<=frame.shape[1]).all()
                          and (corners[:,1]<=frame.shape[0]).all()
                          and .7<=area/old_area<=1.4
                          and .4<=area/self.initial_area<=2.5)
        if valid_motion and self._same_content(frame,motion.box):
            # Even sparse LK may propose a useful location, but the complete
            # recent crop, not those sparse points, must support this fallback.
            return motion.box
        now=time.perf_counter()
        if now-self.last_recovery_attempt<.15:
            return None
        self.last_recovery_attempt=now
        x,y,r,b=self.box;w,h=r-x,b-y
        search=(max(0,int(x-.3*w)),max(0,int(y-.3*h)),
                min(frame.shape[1],int(r+.3*w)),min(frame.shape[0],int(b+.3*h)))
        match=relocate_template(self.gray,gray,self.box,min_similarity=.94,search_box=search)
        if match is None:return None
        candidate=match[0]
        area=max(1,(candidate[2]-candidate[0])*(candidate[3]-candidate[1]))
        old_area=max(1,w*h)
        if (not .7<=area/old_area<=1.4 or not .4<=area/self.initial_area<=2.5
                or not self._same_content(frame,candidate)):
            return None
        return candidate

    def observe(self, frame):
        if self.lost:
            return self._pause('lost')
        if self.failure_since is not None and time.perf_counter()-self.failure_since>=1.5:
            self.lost=True
            return self._pause('lost')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x, y, r, b = self.box
        if x < 0 or y < 0 or r > w or b > h or min(r-x, b-y) < 12:
            self.lost = True
            return self._pause('bounds')
        proposed = self.box
        motion = None
        if self.gray is not None and not np.array_equal(gray, self.gray):
            motion = track_points(self.gray, gray, self.points, self.box, self.corners)
            reliable=False
            if motion is not None:
                proposed = motion.box
                x, y, r, b = proposed
                area = max(1, (r-x)*(b-y))
                corners = motion.corners
                complete = (corners is not None and np.isfinite(corners).all()
                            and (corners[:, 0] >= 0).all() and (corners[:, 0] <= w).all()
                            and (corners[:, 1] >= 0).all() and (corners[:, 1] <= h).all())
                reliable=(complete and motion.inlier_ratio>=.8
                          and min(motion.coverage)>=.25
                          and .4<=area/self.initial_area<=2.5)
            if not reliable:
                proposed=self._recover(frame,gray,motion)
                if proposed is None:
                    return self._pause('motion', True)
                x,y,r,b=proposed
                # Reseed from actual current pixels. Old LK points/corners
                # cannot be paired with a template-relocated reference image.
                motion=None
        detail = reference_sharpness(gray, proposed)
        if self.best_sharpness > 20. and detail < .3*self.best_sharpness:
            return self._pause('blur')
        crop = frame[y:b, x:r]
        small = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        duplicate = any(float(np.abs(small-old).mean()) < 3.
                        for old in self.appearances)
        self.best_sharpness = max(detail, self.best_sharpness)
        self.box = proposed
        self.gray = gray.copy()
        self.failures = 0
        self.failure_since = None
        self.last_crop = small.astype(np.uint8)
        if motion is not None:
            self.points, self.corners = motion.points, motion.corners
        else:
            self.points=None
            self.corners=np.float32([[x,y],[r,y],[r,b],[x,b]])
        if self.points is None or len(self.points) < 12:
            self.points = seed_points(frame, gray, proposed, prefer_bright=False)
        if duplicate:
            self.duplicates += 1
            return CaptureDecision(proposed, 'duplicate')
        self.appearances.append(small)
        if len(self.appearances) > 60:
            del self.appearances[0]
        self.accepted += 1
        return CaptureDecision(proposed, 'accept')
