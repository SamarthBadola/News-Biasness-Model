from collections import Counter

from newsbias.data import Example
from newsbias.folds import make_folds


def _examples():
    ex = []
    plan = [("Center", "Neutral")] * 60 + [("Left", "Neutral")] * 20 + [("Right", "Neutral")] * 20 + \
           [("Center", "Misogynistic")] * 18 + [("Center", "Misandrist")] * 10 + [("Left", "Misandrist")] * 8
    for i, (p, g) in enumerate(plan):
        ex.append(Example(id=f"a{i}", title="t", lede="l", political=p, gender=g, confidence={}, group=f"a{i}"))
    ex[1].group = ex[0].id     # a small near-duplicate group
    ex[2].group = ex[0].id
    return ex


def test_every_row_once_and_groups_together():
    ex = _examples()
    folds = make_folds(ex, k=5, seed=1)
    ids = [i for f in folds for i in f]
    assert sorted(ids) == sorted(e.id for e in ex)
    where = {i: n for n, f in enumerate(folds) for i in f}
    assert where["a0"] == where["a1"] == where["a2"]


def test_rare_classes_spread_evenly():
    ex = _examples()
    by_id = {e.id: e for e in ex}
    for seed in range(5):
        folds = make_folds(ex, k=5, seed=seed)
        for axis in ("gender", "political"):
            for cls in {getattr(e, axis) for e in ex}:
                counts = [Counter(getattr(by_id[i], axis) for i in f)[cls] for f in folds]
                assert max(counts) - min(counts) <= 1, (axis, cls, counts)
