"""Fit the chosen configuration on all rows and save the model the API serves.

    python -m newsbias.final --config configs/base.toml --cv-run base --out models/bias-v1

Use the same config as the cross-validation run you are reporting. The out-of-fold results in
results/<cv-run>/ are the evaluation; this model's training scores are not. ``--cv-run`` copies that run's
calibration temperatures into the model and records which evaluation it belongs to.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import model as model_io
from .config import load_config
from .cv import RESULTS
from .data import describe, file_sha256, load_examples
from .runinfo import environment
from .train import train_model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.toml")
    ap.add_argument("--cv-run", help="cross-validation run whose config and temperatures this model goes with")
    ap.add_argument("--out", required=True, help="folder to write the model to (not committed to git)")
    ap.add_argument("--data", help="override data_path")
    ap.add_argument("--model", help="override model_name")
    ap.add_argument("--epochs", type=int, help="override epochs")
    ap.add_argument("--seed", type=int, help="training seed (default: the config's first seed)")
    ap.add_argument("--results", default=str(RESULTS), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, data_path=args.data, model_name=args.model, epochs=args.epochs)
    temps, cv_info = None, {}
    if args.cv_run:
        run_dir = Path(args.results) / args.cv_run
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        cv_cfg = metrics["config"]
        diff = sorted(k for k in cv_cfg if k not in ("seeds", "bootstrap", "explain_eval")
                      and cv_cfg[k] != cfg.to_dict().get(k))
        if diff:
            sys.exit(f"config differs from cross-validation run {args.cv_run} in: {', '.join(diff)}")
        if not metrics.get("complete"):
            sys.exit(f"cross-validation run {args.cv_run} is not complete")
        temps = metrics["temperatures"]
        p = metrics["primary"]
        cv_info = {"cv_run": args.cv_run, "cv_macro_f1": {a: p[a]["macro_f1"] for a in ("political", "gender")},
                   "cv_balanced_accuracy": {a: p[a]["balanced_accuracy"] for a in ("political", "gender")}}

    examples = load_examples(cfg.data_path)
    print(describe(examples))
    seed = args.seed if args.seed is not None else cfg.seeds[0]
    t0 = time.time()
    model, tokenizer, history = train_model(examples, cfg, seed=seed)
    data_hash = file_sha256(cfg.data_path)
    cfg_hash = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
    version = f"indicbertv2-bias-{time.strftime('%Y%m%d')}-{data_hash[:6]}{cfg_hash[:6]}"
    meta = {"model_version": version, "base_model": cfg.model_name, "max_length": cfg.max_length,
            "temperatures": temps, "training_rows": len(examples), "data_sha256": data_hash, "seed": seed,
            "config": cfg.to_dict(), "history": history, "env": environment(),
            "training_seconds": round(time.time() - t0, 1), **cv_info}
    if temps is None:
        print("warning: no --cv-run given, so confidences are not calibrated (temperature 1)")
    model_io.save(model, tokenizer, args.out, meta)
    print(f"saved {version} to {args.out}")


if __name__ == "__main__":
    main()
