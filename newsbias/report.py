"""Build metrics.json and report.md from a cross-validation run folder.

    python -m newsbias.report results/<run>

Run automatically at the end of `newsbias.cv`; run it by hand to rebuild the report (for example after a run was
interrupted, to see the folds finished so far). It needs only the run folder, not the dataset.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .labels import AXES, NEUTRAL, label_index
from .metrics import ece, evaluate_axis, fit_temperature, nll, overlap_f1, word_set
from .train import softmax


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_preds(run_dir):
    by_seed, histories = {}, {}
    for path in sorted((Path(run_dir) / "preds").glob("seed*_fold*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if "_history" in rec:
                histories[path.stem] = rec
                continue
            by_seed.setdefault(rec["seed"], {})[rec["id"]] = rec
    return by_seed, histories


def _pct(x):
    return f"{100 * x:.1f}"


def _ci(ci):
    return f" ({_pct(ci[0])}–{_pct(ci[1])})" if ci else ""


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def outlet_majority_check(rows, axis, pred_by_id):
    """Accuracy on rows whose label matches / differs from the majority label of *other* rows from the same
    outlet. A model that learned outlet style instead of framing does well on the first group only."""
    by_outlet = {}
    for r in rows:
        by_outlet.setdefault(r["outlet"], []).append(r)
    buckets = {"agrees": [0, 0], "differs": [0, 0]}
    for r in rows:
        if r["id"] not in pred_by_id:
            continue
        others = Counter(o[axis] for o in by_outlet[r["outlet"]] if o["id"] != r["id"])
        if not others:
            continue
        (top, n), *rest = others.most_common()
        if rest and rest[0][1] == n:
            continue
        key = "agrees" if r[axis] == top else "differs"
        buckets[key][0] += int(pred_by_id[r["id"]] == label_index(axis, r[axis]))
        buckets[key][1] += 1
    return {k: {"correct": c, "n": n, "accuracy": (c / n if n else None)} for k, (c, n) in buckets.items()}


def build(run_dir, n_boot=None):
    run_dir = Path(run_dir)
    cfg = _read_json(run_dir / "config.json")
    env = _read_json(run_dir / "env.json")
    rows = _read_json(run_dir / "rows.json")
    folds = _read_json(run_dir / "folds.json")
    base = _read_json(run_dir / "baselines.json")
    n_boot = cfg.get("bootstrap", 1000) if n_boot is None else n_boot
    by_seed, histories = load_preds(run_dir)
    all_ids = [r["id"] for r in rows]
    row_by_id = {r["id"]: r for r in rows}

    metrics = {"run": run_dir.name, "rows": len(rows), "config": cfg, "env": env, "seeds": {}}
    complete = [s for s in cfg["seeds"] if len(by_seed.get(s, {})) == len(all_ids)]
    started = [s for s in cfg["seeds"] if by_seed.get(s)]
    if not started:
        raise SystemExit(f"{run_dir}: no predictions yet")
    primary = complete[0] if complete else started[0]
    partial = not complete
    metrics["primary_seed"] = primary
    metrics["complete"] = not partial

    def truth(ids, axis):
        return np.array([label_index(axis, row_by_id[i][axis]) for i in ids])

    for seed in started:
        preds = by_seed[seed]
        ids = [i for i in all_ids if i in preds]
        seed_res = {"rows": len(ids)}
        for axis in AXES:
            logits = np.array([preds[i]["logits"][axis] for i in ids])
            seed_res[axis] = evaluate_axis(axis, truth(ids, axis), logits.argmax(1),
                                           n_boot if seed == primary else 0, seed=seed)
        metrics["seeds"][str(seed)] = seed_res

    preds = by_seed[primary]
    ids = [i for i in all_ids if i in preds]
    metrics["primary"] = metrics["seeds"][str(primary)]

    # name-masked diagnostic, calibration, per article type, outlet check, baselines
    diag = {"role_masked": {}, "calibration": {}, "article_type": {}, "outlet_majority": {}, "baselines": {},
            "confidence": {}, "errors": {}}
    base_idx = {i: n for n, i in enumerate(base["ids"])}
    for axis in AXES:
        y = truth(ids, axis)
        logits = np.array([preds[i]["logits"][axis] for i in ids])
        masked = np.array([preds[i]["logits_rolemasked"][axis] for i in ids])
        diag["role_masked"][axis] = evaluate_axis(axis, y, masked.argmax(1), 0)
        t = fit_temperature(logits, y)
        diag["calibration"][axis] = {"temperature": t, "nll_before": nll(logits, y), "nll_after": nll(logits, y, t),
                                     "ece_before": ece(softmax(logits), y), "ece_after": ece(softmax(logits, t), y)}
        pred = logits.argmax(1)
        types = {}
        for typ in sorted({row_by_id[i]["article_type"] for i in ids}):
            sel = np.array([row_by_id[i]["article_type"] == typ for i in ids])
            r = evaluate_axis(axis, y[sel], pred[sel], 0)
            types[typ] = {"n": int(sel.sum()), "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"]}
        diag["article_type"][axis] = types
        pred_by_id = dict(zip(ids, pred.tolist()))
        diag["outlet_majority"][axis] = {"model": outlet_majority_check(rows, axis, pred_by_id)}
        for name, bp in base["preds"].items():
            bpred = np.array([bp[axis][base_idx[i]] for i in ids])
            diag["baselines"].setdefault(name, {})[axis] = evaluate_axis(axis, y, bpred, 0)
            if name == "tfidf_logreg":
                diag["outlet_majority"][axis][name] = outlet_majority_check(rows, axis, dict(zip(ids, bpred.tolist())))
        # recall on biased rows split by how sure the annotator was, and every error by id (no text)
        conf = {}
        for i, yt, yp in zip(ids, y, pred):
            c = row_by_id[i]["confidence"][axis]
            bucket = conf.setdefault(c, {"biased_rows": 0, "biased_found": 0, "rows": 0, "correct": 0})
            bucket["rows"] += 1
            bucket["correct"] += int(yt == yp)
            if AXES[axis][yt] != NEUTRAL[axis]:
                bucket["biased_rows"] += 1
                bucket["biased_found"] += int(yt == yp)
        diag["confidence"][axis] = {k: conf[k] for k in ("high", "medium", "low") if k in conf}
        probs = softmax(logits)
        diag["errors"][axis] = [
            {"id": i, "gold": AXES[axis][yt], "predicted": AXES[axis][yp], "p_gold": round(float(probs[n, yt]), 3),
             "confidence": row_by_id[i]["confidence"][axis], "article_type": row_by_id[i]["article_type"]}
            for n, (i, yt, yp) in enumerate(zip(ids, y, pred)) if yt != yp]
    pol = {e["id"]: e for e in diag["errors"]["political"]}
    diag["direction_errors"] = sorted(i for i, e in pol.items() if {e["gold"], e["predicted"]} == {"Left", "Right"})
    metrics["diagnostics"] = diag
    metrics["temperatures"] = {a: diag["calibration"][a]["temperature"] for a in AXES}

    expl = [(i, a, v) for i in ids for a, v in preds[i].get("explanations", {}).items()]
    if expl:
        f1s, rand = [], []
        for _, _, v in expl:
            m, g = set().union(*[word_set(t) for t in v["model"]] or [set()]), set().union(*[word_set(t) for t in v["annotator"]])
            f1s.append(overlap_f1(m, g))
            nw = max(v["n_words"], 1)
            p, r = min(len(g) / nw, 1), min(len(m) / nw, 1)
            rand.append(2 * p * r / (p + r) if p + r else 0.0)
        metrics["explanations"] = {"rows": len(expl), "mean_word_f1": float(np.mean(f1s)),
                                   "random_word_f1": float(np.mean(rand))}

    if len(complete) > 1:
        summ = {}
        for axis in AXES:
            for key in ("macro_f1", "balanced_accuracy"):
                vals = [metrics["seeds"][str(s)][axis][key] for s in complete]
                summ[f"{axis}_{key}"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
            for c in AXES[axis]:
                vals = [metrics["seeds"][str(s)][axis]["per_class"][c]["recall"] for s in complete]
                summ[f"{axis}_recall_{c}"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
        metrics["across_seeds"] = summ

    metrics["training_seconds"] = sum(h.get("seconds", 0) for h in histories.values())
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (run_dir / "report.md").write_text(render(metrics, folds, base, partial), encoding="utf-8")
    return metrics


def render(m, folds, base, partial):
    cfg, env, p, d = m["config"], m["env"], m["primary"], m["diagnostics"]
    L = [f"# Evaluation report: {m['run']}", ""]
    if partial:
        L += [f"> **Partial run.** Only {p['rows']} of {m['rows']} rows have out-of-fold predictions so far. "
              "Numbers below cover those rows only.", ""]
    L += [
        f"Out-of-fold predictions from {cfg['folds']}-fold cross-validation on {m['rows']} rows "
        f"(seed {m['primary_seed']}). Every row was predicted by a model that never saw it. "
        "Ranges in brackets are 95% bootstrap intervals; with 18 rows in a class, one prediction moves recall by "
        "about 5.6 points, so read rare-class numbers as rough.",
        "",
        f"Model `{cfg['model_name']}`, {cfg['epochs']} epochs, batch {cfg['batch_size']}, lr {cfg['lr']} "
        f"(layer decay {cfg['layer_decay']}), head dropout {cfg['head_dropout']}, frozen embeddings "
        f"{cfg['freeze_embeddings']}, class-weight power {cfg['class_weight_power']}, role masking p "
        f"{cfg['role_mask_p']}, Neutral gender swap {cfg['gender_swap_neutral']}, confidence weighting "
        f"{cfg['confidence_weighting']}.",
        f"Data SHA-256 `{env.get('data_sha256', '')[:16]}`, code `{env.get('code_commit', '')}`, torch "
        f"{env.get('torch')}, transformers {env.get('transformers')}, training time "
        f"{m['training_seconds'] / 60:.0f} min.",
        "",
    ]
    for axis in AXES:
        r = p[axis]
        L += [f"## {axis.capitalize()}", "",
              f"Macro-F1 **{_pct(r['macro_f1'])}**{_ci(r.get('macro_f1_ci'))}, balanced accuracy "
              f"**{_pct(r['balanced_accuracy'])}**{_ci(r.get('balanced_accuracy_ci'))}, accuracy "
              f"{_pct(r['accuracy'])}.", ""]
        L.append(_table(["Class", "Rows", "Precision", "Recall", "F1"], [
            [c, v["support"], _pct(v["precision"]) + _ci(v.get("precision_ci")),
             "**" + _pct(v["recall"]) + "**" + _ci(v.get("recall_ci")), _pct(v["f1"])]
            for c, v in r["per_class"].items()]))
        L += ["", "Confusion matrix (rows: true label, columns: predicted):", ""]
        L.append(_table(["true \\ predicted"] + r["labels"],
                        [[r["labels"][i]] + row for i, row in enumerate(r["confusion"])]))
        L += ["", "Against the baselines on the same folds (recall per class, then macro-F1 and balanced accuracy):", ""]
        hdr = ["Model"] + [f"recall {c}" for c in AXES[axis]] + ["macro-F1", "balanced acc."]
        body = []
        for name, label in (("majority", "Majority class"), ("tfidf_logreg", "TF-IDF + logistic regression")):
            b = d["baselines"][name][axis]
            body.append([label] + [_pct(b["per_class"][c]["recall"]) for c in AXES[axis]] +
                        [_pct(b["macro_f1"]), _pct(b["balanced_accuracy"])])
        body.append(["**Fine-tuned model**"] + [_pct(r["per_class"][c]["recall"]) for c in AXES[axis]] +
                    [_pct(r["macro_f1"]), _pct(r["balanced_accuracy"])])
        L += [_table(hdr, body), ""]

    if "across_seeds" in m:
        L += ["## Seed variation", "", "Same folds, different training seeds (mean ± standard deviation):", ""]
        L.append(_table(["Metric", "Mean", "SD"], [[k, _pct(v["mean"]), _pct(v["sd"])]
                                                   for k, v in m["across_seeds"].items()]))
        L.append("")

    L += ["## Shortcut checks", "",
          "These look for signs that the model learned something other than framing.", "",
          "**Names replaced by roles.** The held-out rows predicted again with every listed party, leader and "
          "organisation name replaced by its role (\"an opposition leader\"). Large drops mean the model leans on "
          "names.", ""]
    hdr = ["Axis", "Macro-F1 normal", "Macro-F1 names masked", "Recall change per class (points)"]
    body = []
    for axis in AXES:
        a, b = p[axis], d["role_masked"][axis]
        delta = ", ".join(f"{c} {100 * (b['per_class'][c]['recall'] - a['per_class'][c]['recall']):+.1f}"
                          for c in AXES[axis])
        body.append([axis, _pct(a["macro_f1"]), _pct(b["macro_f1"]), delta])
    L += [_table(hdr, body), ""]
    L += ["**Outlet majority.** Accuracy on rows whose label matches the usual label of the *other* rows from "
          "the same outlet, and on rows where it does not. A big gap means outlet style is doing the work.", ""]
    body = []
    for axis in AXES:
        for who, res in d["outlet_majority"][axis].items():
            body.append([axis, who] + [f"{_pct(res[k]['accuracy'])} ({res[k]['correct']}/{res[k]['n']})"
                                       if res[k]["n"] else "–" for k in ("agrees", "differs")])
    L += [_table(["Axis", "Model", "label = outlet's usual", "label differs"], body), ""]
    L += ["**Top TF-IDF n-grams per class** (logistic regression fitted on all rows). Names here are cues a "
          "model could memorise.", ""]
    for axis, per in base["top_features"].items():
        for cls, feats in per.items():
            L.append(f"- {axis} / {cls}: " + ", ".join(feats))
    L.append("")
    if "explanations" in m:
        e = m["explanations"]
        L += [f"**Highlights against annotator evidence.** On {e['rows']} biased rows with annotator spans, the "
              f"model's top phrases for the gold label share words with the annotator's spans at a mean word-F1 "
              f"of {_pct(e['mean_word_f1'])}, against {_pct(e['random_word_f1'])} for random words.", ""]

    L += ["## Where the errors are", "",
          "**By annotator confidence.** Recall on biased rows (Left/Right, Misogynistic/Misandrist), split by the "
          "confidence the annotator gave the label. If the model misses mostly low-confidence rows, much of the "
          "error is in cases the annotator also found borderline.", ""]
    body = []
    for axis in AXES:
        for c, v in d["confidence"][axis].items():
            rec = f"{_pct(v['biased_found'] / v['biased_rows'])} ({v['biased_found']}/{v['biased_rows']})" \
                if v["biased_rows"] else "–"
            body.append([axis, c, rec, f"{_pct(v['correct'] / v['rows'])} ({v['correct']}/{v['rows']})"])
    L += [_table(["Axis", "Annotator confidence", "Recall on biased rows", "Accuracy on all rows"], body), ""]
    de = d["direction_errors"]
    L += [f"**Direction errors.** {len(de)} political row(s) were predicted on the opposite side (Left as Right or "
          f"Right as Left){': ' + ', '.join(de) if de else ''}. These are worse than calling a biased row Center: "
          "the model saw partisan framing but read its direction wrongly.", "",
          "Every misclassified row is listed by id in `metrics.json` under `diagnostics.errors`. To read them with "
          "their text from your local copy of the dataset: `python -m newsbias.errors " + m["run"] + "`.", ""]
    L += ["## Calibration", "",
          "A single temperature per axis fitted on these out-of-fold logits. The final model applies it, so the "
          "confidence the platform shows is closer to the real hit rate. ECE is the gap between confidence and "
          "accuracy (lower is better).", ""]
    L.append(_table(["Axis", "Temperature", "ECE before", "ECE after", "NLL before", "NLL after"], [
        [a, f"{c['temperature']:.2f}", _pct(c["ece_before"]), _pct(c["ece_after"]), f"{c['nll_before']:.3f}",
         f"{c['nll_after']:.3f}"] for a, c in d["calibration"].items()]))
    L += ["", "## By article type", ""]
    body = []
    for axis in AXES:
        for typ, v in d["article_type"][axis].items():
            body.append([axis, typ, v["n"], _pct(v["accuracy"]), _pct(v["balanced_accuracy"])])
    L += [_table(["Axis", "Type", "Rows", "Accuracy", "Balanced acc."], body), ""]

    L += ["## Folds", "", f"{len(folds['near_duplicate_pairs'])} near-duplicate pair(s) (TF-IDF cosine ≥ "
          f"{folds['group_threshold']}) were kept in the same fold.", ""]
    t = folds["table"]
    cols = [k for k in t[0] if k not in ("fold",)]
    L += [_table(["Fold"] + cols, [[r["fold"]] + [r[c] for c in cols] for r in t]), ""]
    L += ["## Limits", "",
          "- One annotator, so no inter-annotator agreement; the model can at best learn that annotator's "
          "judgement.",
          "- Gender-focused articles were sourced on purpose, so class proportions are not real-world rates and "
          "precision on the platform will be lower than here.",
          "- No independent test set. These are cross-validated estimates on 165 rows.",
          "- The labels describe the headline and lede only, and the model is evaluated on that input only.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--bootstrap", type=int, help="override the number of bootstrap resamples")
    args = ap.parse_args(argv)
    build(args.run_dir, args.bootstrap)
    print(f"wrote {Path(args.run_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
