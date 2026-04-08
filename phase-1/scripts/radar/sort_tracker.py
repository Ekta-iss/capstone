# sort_tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """Compute IOU between two bboxes in [x1,y1,x2,y2] format."""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    union = ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
             + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - intersection)
    return intersection / union if union > 0 else 0

class KalmanBoxTracker:
    """Single object tracker using a simplified Kalman filter."""
    count = 0
    def __init__(self, bbox):
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.no_losses = 0
    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0
    def predict(self):
        # For simplicity, we assume constant velocity (no Kalman prediction)
        return self.bbox

class Sort:
    """Simplified SORT tracker."""
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0, 4))):
        """dets: N x 4 bounding boxes in [x1,y1,x2,y2] format"""
        updated_tracks = []

        # Associate detections to trackers
        if len(self.trackers) == 0:
            for d in dets:
                self.trackers.append(KalmanBoxTracker(d))
        else:
            trks = np.array([t.predict() for t in self.trackers])
            iou_matrix = np.zeros((len(trks), len(dets)), dtype=np.float32)
            for t, trk in enumerate(trks):
                for d, det in enumerate(dets):
                    iou_matrix[t, d] = iou(trk, det)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            assigned_tracks = set()
            assigned_dets = set()
            for t, d in zip(row_ind, col_ind):
                if iou_matrix[t, d] >= self.iou_threshold:
                    self.trackers[t].update(dets[d])
                    updated_tracks.append((self.trackers[t].bbox, self.trackers[t].id))
                    assigned_tracks.add(t)
                    assigned_dets.add(d)

            # Unassigned detections -> create new trackers
            for d, det in enumerate(dets):
                if d not in assigned_dets:
                    self.trackers.append(KalmanBoxTracker(det))
                    updated_tracks.append((det, self.trackers[-1].id))

            # Remove dead trackers
            for t, trk in enumerate(self.trackers):
                if t not in assigned_tracks:
                    trk.no_losses += 1
            self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]

        return updated_tracks