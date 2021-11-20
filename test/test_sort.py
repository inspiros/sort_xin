import os
import re

import numpy as np

from sort_xin import Sort


def load_hien(output_file):
    """
    Load detection output in Hien's format (ultra retard).
    """
    with open(output_file) as f:
        lines = [_.strip('\n') for _ in f.readlines()]

    dets = []
    det = ''
    for line in lines:
        if not line.startswith(' '):
            if len(det):
                det = np.array(eval(re.sub(' +', ' ', det).replace(' ', ',')))
                dets.append(det)
            det = line[line.find(' ') + 1:]
        else:
            det += ' ' + line
    return dets


def main():
    output_file = os.path.join(os.path.dirname(__file__), '../data/out.txt')
    dets = load_hien(output_file)

    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.5, filter_score=True)
    for frame_id, det in enumerate(dets):
        # simulate miss detection by randomly remove some bboxes
        det = det[:-4]

        # update tracklets
        matches, unmatches, track_preds = tracker.update(det)

        print(frame_id)
        print('Matches (matched and filtered detections):')
        print(matches[:, 4:].T)
        print('Tracking predictions (unmatched tracklets under max_age):')
        print(track_preds[:, 4:].T)
        print()


if __name__ == '__main__':
    main()
