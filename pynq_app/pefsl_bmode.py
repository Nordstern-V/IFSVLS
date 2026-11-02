"""C124-inspired bounded B-mode matching. No hardware or training runtime.

Candidate locations and old tracking state never grant identity. Scores,
colour and registered spatial evidence are measured on each current crop.
"""
import numpy as np

from pefsl_scan import (
    FEATURE_DIM, HIST_BINS, _hist_intersection, color_histogram,
    histogram_distance, _unit_rows,
)
from pefsl_identity import compile_references, identity_evidence, shape_descriptor


def colour_supported(query, reference, distance_limit=0.68):
    if reference is None:
        return True
    query = np.asarray(query, np.float32).reshape(HIST_BINS)
    reference = np.asarray(reference, np.float32).reshape(HIST_BINS)
    reference = reference / max(float(reference.sum()), 1e-8)
    # Hue is unstable on neutral surfaces. C124 compares only S/V for these
    # classes rather than deleting skin/white pixels from every category.
    if float(reference[:, 0, :].sum()) > 0.60:
        query, reference = query.sum(axis=0), reference.sum(axis=0)
    distance = histogram_distance(query, reference)
    return (distance is not None and distance <= float(distance_limit)) or \
        _hist_intersection(query, reference) >= 0.35


class BModeMatcher:
    """Compile once per learning revision; use the same gates for ROI/scan."""
    def __init__(self, data):
        self.labels = list(data.labels)
        self.rows = [np.asarray(rows, np.float32).copy() for rows in data.shots]
        self.hists = [None if h is None else np.asarray(h).copy() for h in data.color_hists]
        self.references = [[view.copy() for view in bank] for bank in
                           getattr(data, 'reference_views', [[] for _ in self.labels])]
        self.shapes = [None if s is None else np.asarray(s).copy() for s in
                       getattr(data, 'class_shapes', [None for _ in self.labels])]
        self.raw_flags = list(getattr(data, 'raw_cosine', [False for _ in self.labels]))
        legacy_rows = [rows for rows, raw in zip(self.rows, self.raw_flags) if not raw]
        frozen = getattr(data, 'legacy_mean', None)
        baseline = (np.asarray(frozen, np.float32).copy() if frozen is not None else
                    np.concatenate(legacy_rows, axis=0).mean(axis=0) if legacy_rows else np.zeros(FEATURE_DIM, np.float32))
        self.baseline = baseline
        self.banks = []
        self.view_hists = []
        self.spatial_banks = []
        for index, rows in enumerate(self.rows):
            # A sole positive class has no valid dataset centre. Preserve its
            # direction, even when loading a historical archive.
            raw = self.raw_flags[index] or len(self.labels) == 1
            self.banks.append(_unit_rows(rows if raw else rows - baseline))
            self.view_hists.append([color_histogram(view) for view in self.references[index]])
            self.spatial_banks.append(compile_references(self.references[index]))

    def recognize(self, feature, crop, cos_threshold=0.56, hist_threshold=0.0):
        feature = np.asarray(feature, np.float32).reshape(-1)
        if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
            raise ValueError('B feature must contain 80 finite values')
        if not self.labels:
            return dict(label=None, score=0.0, reason='no learned data')
        query_raw = _unit_rows(feature[None])[0]
        query_centered = _unit_rows((feature-self.baseline)[None])[0]
        hist = color_histogram(crop)
        shape = shape_descriptor(crop)
        accepted = []
        best_score, rejection = 0.0, 'feature'
        for index, label in enumerate(self.labels):
            query = query_raw if self.raw_flags[index] or len(self.labels) == 1 else query_centered
            similarities = np.sort(self.banks[index] @ query)[::-1]
            k = min(5, len(similarities))
            if k == 0:
                continue
            score = float(.7*similarities[0] + .3*similarities[:k].mean())
            if score > best_score:
                best_score, rejection = score, 'feature'
            if score < float(cos_threshold):
                continue
            hists = self.view_hists[index] or [self.hists[index]]
            if not any(colour_supported(hist, ref) for ref in hists):
                rejection = 'colour'
                continue
            if float(hist_threshold) > 0 and not any(
                    _hist_intersection(hist, ref) >= float(hist_threshold) for ref in hists):
                rejection = 'colour'
                continue
            ref_shape = self.shapes[index]
            if ref_shape is not None and ((ref_shape[0] > .01 and shape[0] < .35*ref_shape[0]) or
                                          (ref_shape[1] > 8.0 and shape[1] < .35*ref_shape[1])):
                rejection = 'flat/texture'
                continue
            if self.references[index]:
                evidence = identity_evidence(crop, self.references[index],
                                             prepared_references=self.spatial_banks[index])
                if not evidence['supported']:
                    rejection = 'identity:' + evidence['reason']
                    continue
            accepted.append((score, str(label), bool(self.raw_flags[index])))
        if not accepted:
            return dict(label=None, score=max(0., best_score), reason=rejection)
        accepted.sort(reverse=True)
        score, label, raw = accepted[0]
        # Raw CNN similarities cluster more closely; spatial and colour gates
        # above have already rejected unsupported classes before this test.
        margin = .015 if raw else .04
        if len(accepted)>1 and score-accepted[1][0]<margin:
            return dict(label=None, score=score, reason='ambiguous')
        return dict(label=label, score=score, reason='match')
