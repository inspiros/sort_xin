"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai_vision

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from collections import namedtuple

import numpy as np
from filterpy.kalman import KalmanFilter

__all__ = ['Sort']

_SortReturnType = namedtuple('_SortReturnType', ['matches', 'unmatches', 'track_preds'])

try:
    import lap


    def linear_assignment(cost_matrix):
        x, y = lap.lapjv(cost_matrix, extend_cost=True, return_cost=False)
        return np.array([[y[i], i] for i in x if i >= 0])
except ImportError:
    from scipy.optimize import linear_sum_assignment


    def linear_assignment(cost_matrix):
        x, y = linear_sum_assignment(cost_matrix)
        return np.stack([x, y]).T


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IUO between two bboxes in the form [l,t,w,h]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    x = x.flatten()
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score.item()]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, init_bbox=None, filter_score=False, update_with_prediction=False):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # noisy confidence score model
        self.score_kf = KalmanFilter(dim_x=1, dim_z=1)
        self.score_kf.F = np.array([[1.]])
        self.score_kf.H = np.array([[1.]])

        # TODO: tune these parameters
        self.score_kf.R[0, 0] *= 100. if filter_score else 1e-10
        self.score_kf.P[0, 0] *= 1.
        self.score_kf.Q[0, 0] *= 5. if filter_score else 1e10

        if init_bbox is not None:
            self.kf.x[:4] = convert_bbox_to_z(init_bbox)
            self.score_kf.x[0, 0] = init_bbox[4]
        self.update_with_prediction = update_with_prediction

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1  # TODO: count id from 0 or 1
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.score_kf.update(bbox[4:5][np.newaxis])

    def self_update(self):
        """
        Updates the state vector with self predictions.
        """
        if self.update_with_prediction:
            self.kf.update(self.kf.x[:4])
        self.score_kf.update(np.zeros_like(self.score_kf.x))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

        if self.score_kf.x[0] < 0:
            self.score_kf.x[0] = 0.0
        elif self.score_kf.x[0] > 1:
            self.score_kf.x[0] = 1.0
        self.score_kf.predict()

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x, self.score_kf.x))
        return self.history[-1]

    @property
    def bbox_state(self):
        return convert_x_to_bbox(self.kf.x)

    @property
    def score_state(self):
        return self.score_kf.x

    @property
    def state(self):
        return convert_x_to_bbox(self.kf.x, self.score_kf.x)


class Sort(object):
    def __init__(self,
                 max_age=1,
                 min_hits=3,
                 iou_threshold=0.3,
                 filter_score=False,
                 kalman_internal_update=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.filter_score = filter_score
        self.kalman_internal_update = kalman_internal_update

        self.trackers = []
        self.frame_count = 0

    def __call__(self, dets):
        return self.update(dets)

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for
          frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        for j, trk in enumerate(trks):
            pos = self.trackers[j].predict().flatten()
            trk[:] = np.array(pos.tolist() + [self.trackers[j].id])
            if np.any(np.isnan(pos)):
                to_del.append(j)
        for j in reversed(to_del):  # delete nan tracklets
            self.trackers.pop(j)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        unmatched_dets_trks_inds, unmatched_dets_inds, unmatched_trks_inds = self.associate_detections(dets, trks)
        inds = unmatched_dets_trks_inds.tolist()

        # update matched trackers with assigned detections
        for i, j in unmatched_dets_trks_inds:
            self.trackers[j].update(dets[i])

        # update unmatched trackers with 0 confidence score
        for j in unmatched_trks_inds:
            self.trackers[j].self_update()

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets_inds:
            trk = KalmanBoxTracker(dets[i, :], self.filter_score, self.kalman_internal_update)
            self.trackers.append(trk)
            inds.append([i, len(self.trackers) - 1])
        inds = np.array(sorted(inds, key=lambda _: _[0]))

        unmatched_trks = trks[unmatched_trks_inds.tolist()]
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.state.flatten()
            i -= 1
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            else:
                ret.append(None)
                # remove dead tracklet
                if trk.time_since_update > self.max_age:
                    unmatched_trks = unmatched_trks[unmatched_trks[:, -1] != self.trackers[i].id]
                    self.trackers.pop(i)
        ret.reverse()

        unmatches = [id for id, (i, j) in enumerate(inds) if ret[j] is None]
        ret = [ret[j] for i, j in inds if ret[j] is not None]
        return _SortReturnType(np.concatenate(ret).astype(dets.dtype) if len(ret) else
                               np.empty((0, 5), dtype=dets.dtype),
                               unmatches,
                               unmatched_trks)

    def associate_detections(self, detections, trackers):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2), dtype=int)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
