"""Print the rows a run got wrong, with their text, from your local copy of the dataset.

    python -m newsbias.errors base                      # every error on both axes
    python -m newsbias.errors base --axis gender --biased-only

The run folder holds only ids, so the article text comes from the dataset file and is printed, never written into
this repo. Shows the gold label, the annotator's confidence and rationale, the prediction and the probability the
model gave the gold label. Use it to decide whether an error is the model's or the label's.
"""
import argparse
import sys
import json
from pathlib import Path

from .cv import RESULTS
from .data import read_jsonl
from .labels import AXES, NEUTRAL


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name under results/, or a path to a run folder")
    ap.add_argument("--axis", choices=list(AXES))
    ap.add_argument("--biased-only", action="store_true", help="only rows whose gold label is not Center/Neutral")
    ap.add_argument("--data", help="dataset file (default: the run's data_path)")
    ap.add_argument("--chars", type=int, default=400, help="how much of the lede to print")
    args = ap.parse_args(argv)
    # article text has curly quotes and Indic characters; never crash on a console that cannot show them
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    run_dir = Path(args.run) if Path(args.run).is_dir() else RESULTS / args.run
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    errors = metrics["diagnostics"].get("errors")
    if errors is None:
        raise SystemExit(f"{run_dir} has no error list; rebuild it with: python -m newsbias.report {run_dir}")
    data = {r["id"]: r for r in read_jsonl(args.data or metrics["config"]["data_path"])}
    for axis in [args.axis] if args.axis else AXES:
        rows = [e for e in errors[axis] if not args.biased_only or e["gold"] != NEUTRAL[axis]]
        print(f"===== {axis}: {len(rows)} error(s)\n")
        for e in sorted(rows, key=lambda e: (e["gold"], e["predicted"], e["p_gold"])):
            r = data.get(e["id"])
            print(f"{e['id']}  gold {e['gold']} ({e['confidence']})  predicted {e['predicted']}  "
                  f"p(gold) {e['p_gold']:.2f}  {e['article_type']}")
            if r is None:
                print("   (not in the dataset file)\n")
                continue
            lab = r["labels"][axis]
            print(f"   headline:  {r['title']}")
            print(f"   lede:      {r['lede'][:args.chars]}{'...' if len(r['lede']) > args.chars else ''}")
            print(f"   rationale: {lab.get('rationale', '')}")
            spans = [s.get("text", "") for s in lab.get("spans") or []]
            if spans:
                print("   evidence:  " + " | ".join(s[:120] for s in spans))
            print()


if __name__ == "__main__":
    main()
