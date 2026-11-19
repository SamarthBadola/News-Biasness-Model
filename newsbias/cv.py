"""Five-fold cross-validation of the fine-tuned classifier, with baselines and a report.

    python -m newsbias.cv --config configs/base.toml --run base

Writes results/<run>/:
    config.json       the exact settings used
    env.json          versions, CPU, code commit, SHA-256 of the data file
    rows.json         id, labels, outlet and article type of every row (no article text)
    folds.json        fold membership, per-fold class counts and the near-duplicate pairs kept together
    baselines.json    out-of-fold predictions of the majority and TF-IDF baselines, and TF-IDF top n-grams
    preds/            one file per seed and fold: out-of-fold logits, normal and with all names role-masked
    train_log.txt     loss and time per epoch
    metrics.json      everything the report shows, machine-readable (the final model reads temperatures here)
    report.md         the evaluation report

The run resumes: folds whose prediction file exists are skipped, so stopping it (Ctrl+C) and starting the same
command again continues where it left off. Changing the config of an existing run is refused; use a new --run.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import baselines, report
from .augment import RoleMasker
from .attribution import explain
from .config import load_config
from .data import assign_groups, describe, file_sha256, load_examples
from .folds import fold_table, make_folds
from .labels import AXES, NEUTRAL, label_index
from .runinfo import environment
from .train import predict_logits, train_model

RESULTS = Path(__file__).resolve().parent.parent / "results"


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


def prepare_run_dir(run_dir, cfg_dict, overwrite):
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != cfg_dict:
            if not overwrite:
                changed = sorted(k for k in set(old) | set(cfg_dict) if old.get(k) != cfg_dict.get(k))
                sys.exit(f"{run_dir} was made with different settings ({', '.join(changed)}). "
                         f"Use a new --run name, or --overwrite to delete the old results.")
            shutil.rmtree(run_dir)
    (run_dir / "preds").mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")


def explanation_rows(model, tokenizer, test, cfg):
    """For rows with a biased gold label and annotator evidence: the phrases the model credits for that label."""
    out = {}
    for e in test:
        for axis in AXES:
            gold = getattr(e, axis)
            if gold == NEUTRAL[axis] or not e.spans.get(axis):
                continue
            ph = explain(model, tokenizer, e.title, e.lede, axis, label_index(axis, gold), cfg.ig_steps,
                         cfg.max_length, cfg.top_phrases)
            texts = [e.title, e.lede]
            out.setdefault(e.id, {})[axis] = {
                "model": [texts[p["field"]][p["start"]:p["end"]] for p in ph],
                "annotator": e.spans[axis],
                "n_words": len((e.title + " " + e.lede).split()),
            }
    return out


def run(args):
    cfg = load_config(args.config, data_path=args.data, model_name=args.model, epochs=args.epochs)
    if args.seeds:
        cfg.seeds = args.seeds
    run_dir = Path(args.results) / args.run
    prepare_run_dir(run_dir, cfg.to_dict(), args.overwrite)
    log = Logger(run_dir / "train_log.txt")

    examples = load_examples(cfg.data_path)
    if args.limit:
        examples = examples[:args.limit]
    log(f"data: {cfg.data_path}\n{describe(examples)}")

    env = environment()
    env["data_sha256"] = file_sha256(cfg.data_path)
    env["rows"] = len(examples)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    rows = [{"id": e.id, "political": e.political, "gender": e.gender, "outlet": e.outlet,
             "article_type": e.article_type, "confidence": e.confidence} for e in examples]
    (run_dir / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    pairs = assign_groups(examples, cfg.group_threshold)
    folds = make_folds(examples, cfg.folds, cfg.fold_seed)
    (run_dir / "folds.json").write_text(json.dumps(
        {"folds": folds, "table": fold_table(examples, folds), "near_duplicate_pairs": pairs,
         "group_threshold": cfg.group_threshold}, indent=1), encoding="utf-8")
    log(f"folds: {[len(f) for f in folds]}, near-duplicate pairs kept together: {len(pairs)}")

    base_path = run_dir / "baselines.json"
    if not base_path.exists():
        bp = baselines.out_of_fold(examples, folds)
        base_path.write_text(json.dumps({
            "ids": [e.id for e in examples],
            "preds": {m: {a: v.tolist() for a, v in d.items()} for m, d in bp.items()},
            "top_features": baselines.top_features(examples),
        }, indent=1), encoding="utf-8")
        log("baselines done")

    by_id = {e.id: e for e in examples}
    masker = RoleMasker(p=1.0)
    for seed in cfg.seeds:
        for k, test_ids in enumerate(folds):
            out_path = run_dir / "preds" / f"seed{seed}_fold{k}.jsonl"
            if out_path.exists():
                log(f"seed {seed} fold {k}: already done, skipping")
                continue
            test_set = set(test_ids)
            train = [e for e in examples if e.id not in test_set]
            test = [by_id[x] for x in test_ids]
            log(f"seed {seed} fold {k}: train {len(train)}, test {len(test)}")
            t0 = time.time()
            model, tokenizer, history = train_model(train, cfg, seed=seed * 100 + k, log=log)
            logits = predict_logits(model, tokenizer, [(e.title, e.lede) for e in test], cfg)
            masked = predict_logits(model, tokenizer,
                                    [(masker.mask_all(e.title), masker.mask_all(e.lede)) for e in test], cfg)
            expl = explanation_rows(model, tokenizer, test, cfg) if cfg.explain_eval else {}
            tmp = out_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for i, e in enumerate(test):
                    fh.write(json.dumps({
                        "id": e.id, "fold": k, "seed": seed,
                        "logits": {a: logits[a][i].round(5).tolist() for a in AXES},
                        "logits_rolemasked": {a: masked[a][i].round(5).tolist() for a in AXES},
                        "names_masked": masker.count(e.title) + masker.count(e.lede),
                        "explanations": expl.get(e.id, {}),
                    }) + "\n")
                fh.write(json.dumps({"_history": history, "seconds": round(time.time() - t0, 1)}) + "\n")
            tmp.replace(out_path)
            log(f"seed {seed} fold {k}: done in {time.time() - t0:.0f}s")
            del model

    report.build(run_dir)
    log(f"report written to {run_dir / 'report.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.toml")
    ap.add_argument("--run", required=True, help="name of the results folder under results/")
    ap.add_argument("--data", help="override data_path from the config")
    ap.add_argument("--model", help="override model_name (a Hub id or a local folder)")
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--seeds", type=int, nargs="+", help="override seeds")
    ap.add_argument("--limit", type=int, help="use only the first N rows (smoke tests only)")
    ap.add_argument("--results", default=str(RESULTS), help=argparse.SUPPRESS)
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run with different settings")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
