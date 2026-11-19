"""Five-fold split that keeps groups together and balances both label axes at once.

Greedy iterative stratification over groups: groups holding the rarest classes are placed first, each into the
fold that most needs its classes (measured against the per-fold target for every class on both axes, weighted
so rare classes count for more). Ties go to the smaller fold, then to a seeded random choice. With 18 examples
in each rare gender class, each fold gets 3 or 4 of them.
"""
import random
from collections import Counter

from .labels import AXES

CLASS_KEYS = [(axis, c) for axis, classes in AXES.items() for c in classes]


def _counts(examples):
    c = Counter()
    for e in examples:
        for axis in AXES:
            c[(axis, getattr(e, axis))] += 1
    return c


def make_folds(examples, k=5, seed=0):
    """Return a list of k lists of example ids."""
    rng = random.Random(seed)
    groups = {}
    for e in examples:
        groups.setdefault(e.group or e.id, []).append(e)
    total = _counts(examples)
    target = {key: total[key] / k for key in CLASS_KEYS}
    folds = [[] for _ in range(k)]
    fold_counts = [Counter() for _ in range(k)]

    def rarity(members):
        return min(total[key] for key in _counts(members))

    order = list(groups.values())
    rng.shuffle(order)
    order.sort(key=lambda members: (rarity(members), -len(members)))

    for members in order:
        gc = _counts(members)
        best, best_score = [], None
        for f in range(k):
            need = sum(n * (target[key] - fold_counts[f][key]) / total[key] for key, n in gc.items())
            score = (round(need, 9), -len(folds[f]))
            if best_score is None or score > best_score:
                best, best_score = [f], score
            elif score == best_score:
                best.append(f)
        f = rng.choice(best)
        folds[f].extend(e.id for e in members)
        fold_counts[f].update(gc)
    return folds


def fold_table(examples, folds):
    """Per-fold class counts, as rows of dicts, for the report."""
    by_id = {e.id: e for e in examples}
    rows = []
    for i, ids in enumerate(folds):
        c = _counts([by_id[x] for x in ids])
        row = {"fold": i, "rows": len(ids)}
        row.update({f"{axis}:{cls}": c[(axis, cls)] for axis, cls in CLASS_KEYS})
        rows.append(row)
    return rows
