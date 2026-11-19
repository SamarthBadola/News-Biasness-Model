"""Compare cross-validation runs on the same rows, with paired statistics.

    python -m newsbias.compare base no-shortcut-controls no-class-weights
    python -m newsbias.compare base cw-half --out results/comparisons/base-vs-cw-half.md

The first run is the reference. Because every run predicts the same 165 rows on the same folds, differences are
measured *paired*: each bootstrap resample draws one set of rows and scores both runs on it. That removes the
row-sampling noise the two runs share, so a paired interval is much narrower than comparing two reports by eye.

For each axis it shows per-class recall and precision, macro-F1 and balanced accuracy for every run, then for
every other run versus the reference:

* the difference with a 95% paired bootstrap interval, and how often the resampled difference was above zero;
* McNemar counts: rows only the reference got right, rows only the other run got right, and the exact two-sided
  p-value for that split. A difference built on a handful of rows is not evidence of a better setting.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .cv import RESULTS
from .labels import AXES, label_index
from .metrics import confusion, from_confusion
from .report import load_preds


def load_run(name):
    run_dir = Path(name) if Path(name).is_dir() else RESULTS / name
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = json.loads((run_dir / "rows.json").read_text(encoding="utf-8"))
    folds = json.loads((run_dir / "folds.json").read_text(encoding="utf-8"))["folds"]
    by_seed, _ = load_preds(run_dir)
    seed = next((s for s in cfg["seeds"] if len(by_seed.get(s, {})) == len(rows)), None)
    if seed is None:
        raise SystemExit(f"{run_dir}: no seed with predictions for every row; finish the run first")
    return {"name": run_dir.name, "cfg": cfg, "rows": rows, "folds": folds, "preds": by_seed[seed], "seed": seed}


def mcnemar_p(b, c):
    """Exact two-sided McNemar test (binomial on the discordant rows)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _scores(y, p, k):
    r = from_confusion(confusion(y, p, k))
    return r


def compare(runs, n_boot=2000, seed=0):
    ref = runs[0]
    ids = [r["id"] for r in ref["rows"]]
    for run in runs[1:]:
        if [r["id"] for r in run["rows"]] != ids or run["folds"] != ref["folds"]:
            raise SystemExit(f"{run['name']} was not run on the same rows and folds as {ref['name']}")
        if any(a[x] != b[x] for a, b in zip(ref["rows"], run["rows"]) for x in AXES):
            raise SystemExit(f"{run['name']} has different labels from {ref['name']} (data changed between runs)")
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(ids), (n_boot, len(ids)))
    out = {}
    for axis, classes in AXES.items():
        k = len(classes)
        y = np.array([label_index(axis, r[axis]) for r in ref["rows"]])
        preds = {run["name"]: np.array([int(np.argmax(run["preds"][i]["logits"][axis])) for i in ids]) for run in runs}
        res = {"runs": {}, "vs_reference": {}}
        for name, p in preds.items():
            s = _scores(y, p, k)
            res["runs"][name] = {"macro_f1": s["macro_f1"], "balanced_accuracy": s["balanced_accuracy"],
                                 "recall": dict(zip(classes, s["recall"].tolist())),
                                 "precision": dict(zip(classes, s["precision"].tolist()))}
        p0 = preds[ref["name"]]
        for name, p1 in list(preds.items())[1:]:
            diffs = {"macro_f1": [], "balanced_accuracy": []}
            diffs.update({f"recall {c}": [] for c in classes})
            for idx in boots:
                a, b = _scores(y[idx], p0[idx], k), _scores(y[idx], p1[idx], k)
                diffs["macro_f1"].append(b["macro_f1"] - a["macro_f1"])
                diffs["balanced_accuracy"].append(b["balanced_accuracy"] - a["balanced_accuracy"])
                for j, c in enumerate(classes):
                    diffs[f"recall {c}"].append(b["recall"][j] - a["recall"][j])
            full0, full1 = _scores(y, p0, k), _scores(y, p1, k)
            point = {"macro_f1": full1["macro_f1"] - full0["macro_f1"],
                     "balanced_accuracy": full1["balanced_accuracy"] - full0["balanced_accuracy"]}
            point.update({f"recall {c}": full1["recall"][j] - full0["recall"][j] for j, c in enumerate(classes)})
            stats = {}
            for key, vals in diffs.items():
                v = np.array(vals)
                lo, hi = np.percentile(v, [2.5, 97.5])
                stats[key] = {"diff": float(point[key]), "ci": [float(lo), float(hi)], "p_above_zero": float((v > 0).mean())}
            ok0, ok1 = p0 == y, p1 == y
            only_ref, only_new = int((ok0 & ~ok1).sum()), int((~ok0 & ok1).sum())
            per_class = {}
            for j, c in enumerate(classes):
                sel = y == j
                per_class[c] = {"only_reference": int((ok0 & ~ok1 & sel).sum()), "only_this": int((~ok0 & ok1 & sel).sum())}
            res["vs_reference"][name] = {"stats": stats, "only_reference": only_ref, "only_this": only_new,
                                         "mcnemar_p": mcnemar_p(only_ref, only_new), "per_class_flips": per_class}
        out[axis] = res
    return out


def _pct(x):
    return f"{100 * x:.1f}"


def render(runs, res):
    ref = runs[0]["name"]
    L = [f"# Run comparison: {', '.join(r['name'] for r in runs)}", "",
         f"Reference: **{ref}**. Same rows and folds in every run; differences are paired (see "
         "`newsbias/compare.py`). An interval that contains 0 means the data cannot tell the runs apart.", ""]
    for axis, classes in AXES.items():
        r = res[axis]
        L += [f"## {axis.capitalize()}", "",
              "| Run | " + " | ".join(f"recall {c}" for c in classes) + " | " +
              " | ".join(f"precision {c}" for c in classes) + " | macro-F1 | balanced acc. |",
              "|---|" + "---:|" * (2 * len(classes) + 2)]
        for name, s in r["runs"].items():
            L.append(f"| {name} | " + " | ".join(_pct(s["recall"][c]) for c in classes) + " | " +
                     " | ".join(_pct(s["precision"][c]) for c in classes) +
                     f" | {_pct(s['macro_f1'])} | {_pct(s['balanced_accuracy'])} |")
        L.append("")
        for name, v in r["vs_reference"].items():
            L += [f"**{name} minus {ref}** (percentage points, 95% paired interval, share of resamples above 0):", "",
                  "| Metric | Difference | Interval | P(> 0) |", "|---|---:|---:|---:|"]
            for key, s in v["stats"].items():
                L.append(f"| {key} | {100 * s['diff']:+.1f} | {100 * s['ci'][0]:+.1f} to {100 * s['ci'][1]:+.1f} | "
                         f"{_pct(s['p_above_zero'])}% |")
            flips = ", ".join(f"{c} {f['only_reference']}→{f['only_this']}" for c, f in v["per_class_flips"].items())
            L += ["", f"Rows only {ref} got right: {v['only_reference']}; only {name}: {v['only_this']} "
                      f"(McNemar p = {v['mcnemar_p']:.2f}). Per gold class ({ref}-only → {name}-only): {flips}.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names under results/ (the first is the reference)")
    ap.add_argument("--out", help="also write the comparison to this Markdown file")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    if len(args.runs) < 2:
        ap.error("give at least two runs")
    runs = [load_run(r) for r in args.runs]
    text = render(runs, compare(runs, args.bootstrap))
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
