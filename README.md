### SORT (siêu xịn) cho các vong

SORT này lọc cả confidence score (**stupid idea** - Tàu ko làm Tây ko làm mà Đao Lồng đòi làm thì chỉ có ăn cám). \
Lần sau tự lên đây mà kéo về chứ ko zip cho nữa đâu.

#### Installation

- Requirements:
    - `numpy`
    - `scipy` (Hungarian algorithm) or  `lap` (Jonker-Volgenant algorithm - faster)
    - `filterpy` (Kalman filter)
- Install:
    - Clone this repo
    - Run `pip install .`

#### How to use

First, run test file `test/test_sort.py` to grab the idea. Sample outputs of detection model are stored
at `data/out.txt` (_**ultra-retarded format alert**_).

```terminal
cd test
python test_sort.py
```

Now, copy folder `sort_xin` to your code base or install it as a package with `pip` (see above). \
Then let's program

```python
from sort_xin import Sort

# initialize
tracker = Sort(max_age=10,
               min_hits=1,
               iou_threshold=0.5,
               conf_threshold=0.1,
               filter_score=True,
               kalman_return_predictions=True,
               kalman_internal_update=True)

while True:
    det = model(img)
    matches, unmatches, track_preds = tracker.update(det)
    # print to see format, do something cool
```

#### Note

Under development. \
Reference: https://github.com/abewley/sort

_Oanh Hiền lười như chó!_
