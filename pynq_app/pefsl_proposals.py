"""Small C124-style edge/colour proposals; locations have no acceptance rights."""
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None


def region_proposals(image, origin=(0, 58), pixel_scale=2, limit=6):
    """Inspect one bounded scene thumbnail, never the UI or whole-resolution frame.

    Without OpenCV the regular rotating scan still operates. This function is
    called by the inference worker once per cycle, not by the touch/video loop.
    """
    if cv2 is None or image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        fill = cv2.contourArea(contour)/max(1., w*h)
        if min(w, h) >= 10 and w*h < .85*width*height and fill >= .30:
            regions.append((fill*np.sqrt(w*h), x, y, w, h))
    # Only the eight most occupied Lab cells are considered. C124 searches
    # more quantizers; this bound keeps the ARM worker's cost predictable.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int32)//32
    codes = lab[..., 0]*64 + lab[..., 1]*8 + lab[..., 2]
    values, counts = np.unique(codes, return_counts=True)
    for index in np.argsort(counts)[-8:]:
        mask = (codes == values[index]).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for x, y, w, h, area in stats[1:n]:
            if min(w, h) >= 10 and 50 <= area < .85*width*height and area/max(1., w*h) >= .45:
                regions.append((area/np.sqrt(w*h), int(x), int(y), int(w), int(h)))
    boxes = []
    for _, x, y, w, h in sorted(regions, reverse=True):
        box = (max(0, x-1), max(0, y-1), min(width, x+w+1), min(height, y+h+1))
        duplicate = False
        for old in boxes:
            inter = max(0, min(box[2], old[2])-max(box[0], old[0]))*max(0, min(box[3], old[3])-max(box[1], old[1]))
            union = (box[2]-box[0])*(box[3]-box[1])+(old[2]-old[0])*(old[3]-old[1])-inter
            if inter/max(1., union) > .80:
                duplicate = True
                break
        if not duplicate:
            boxes.append(box)
        if len(boxes) >= int(limit):
            break
    return [tuple(int(origin[i % 2]+v*pixel_scale) for i, v in enumerate(box)) for box in boxes]
