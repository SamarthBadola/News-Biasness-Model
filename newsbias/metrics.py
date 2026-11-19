"""Evaluation metrics on pooled out-of-fold predictions.

Per class: support, precision, recall, F1. Per axis: macro-F1, balanced accuracy (the mean of per-class
recall), accuracy and the confusion matrix. Every headline number gets a 95% percentile bootstrap interval
(rows resampled with replacement), because with 18 rows in a class one prediction moves recall by 5.6 points.

Calibration: expected calibration error (ECE, 10 equal-width bins) of the top-class probability, and a single
temperature per axis fitted on the out-of-fold logits by minimising negative log-likelihood. The final model
divides its logits by that temperature so the confidence the platform shows means roughly what it says.
"""
import numpy as np

from .labels import AXES


def confusion(y_true, y_pred, k):
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def from_confusion(m):
    tp = np.diag(m).astype(float)
    support = m.sum(1).astype(float)
    predicted = m.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    return {
        "precision": precision, "recall": recall, "f1": f1, "support": support.astype(int),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "accuracy": float(tp.sum() / max(m.sum(), 1)),
    }


def bootstrap(y_true, y_pred, k, n_boot, seed=0):
    """Percentile 95% intervals for per-class recall/precision/F1, macro-F1 and balanced accuracy."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    stats = {"macro_f1": [], "balanced_accuracy": [], "recall": [], "precision": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = from_confusion(confusion(y_true[idx], y_pred[idx], k))
        for key in stats:
            stats[key].append(r[key])
    out = {}
    for key, vals in stats.items():
        arr = np.array(vals)
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        out[key] = (lo.tolist() if np.ndim(lo) else float(lo), hi.tolist() if np.ndim(hi) else float(hi))
    return out


def evaluate_axis(axis, y_true, y_pred, n_boot=1000, seed=0):
    classes = AXES[axis]
    m = confusion(y_true, y_pred, len(classes))
    r = from_confusion(m)
    ci = bootstrap(y_true, y_pred, len(classes), n_boot, seed)
    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {"support": int(r["support"][i]), "precision": float(r["precision"][i]),
                        "recall": float(r["recall"][i]), "f1": float(r["f1"][i])}
        if ci:
            for key in ("precision", "recall", "f1"):
                per_class[c][f"{key}_ci"] = [ci[key][0][i], ci[key][1][i]]
    res = {"per_class": per_class, "macro_f1": r["macro_f1"], "balanced_accuracy": r["balanced_accuracy"],
           "accuracy": r["accuracy"], "confusion": m.tolist(), "labels": list(classes), "n": int(len(y_true))}
    if ci:
        res["macro_f1_ci"] = list(ci["macro_f1"])
        res["balanced_accuracy_ci"] = list(ci["balanced_accuracy"])
    return res


def _log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def nll(logits, y, t=1.0):
    lp = _log_softmax(np.asarray(logits, dtype=np.float64) / t)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits, y):
    """Scalar temperature in [0.05, 20] minimising NLL (golden-section search on log T)."""
    lo, hi = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(logits, y, np.exp(a)), nll(logits, y, np.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(logits, y, np.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(logits, y, np.exp(b))
    return float(np.exp((lo + hi) / 2))


def ece(probs, y, bins=10):
    probs = np.asarray(probs)
    conf = probs.max(1)
    correct = (probs.argmax(1) == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def word_set(text):
    import re
    return {w for w in re.findall(r"[\w’']+", text.lower()) if len(w) > 1}


def overlap_f1(pred_words, gold_words):
    if not pred_words or not gold_words:
        return 0.0
    inter = len(pred_words & gold_words)
    if not inter:
        return 0.0
    p, r = inter / len(pred_words), inter / len(gold_words)
    return 2 * p * r / (p + r)
