"""Check the corpus and show what preprocessing does, without training anything.

    python -m newsbias.prepare --config configs/base.toml
    python -m newsbias.prepare --show-swaps 5 --show-masks 5

Prints class counts, the political x gender table, near-duplicate pairs, the planned folds, token lengths (if the
tokenizer can be loaded) and examples of source masking, role masking and Neutral gender swaps, so the
preprocessing can be checked by eye before a long run.
"""
import argparse
import sys
import random

from .augment import RoleMasker, gender_swap
from .config import load_config
from .data import assign_groups, describe, load_examples
from .folds import fold_table, make_folds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.toml")
    ap.add_argument("--data")
    ap.add_argument("--model", help="tokenizer to measure lengths with (default: the config's model)")
    ap.add_argument("--show-swaps", type=int, default=3)
    ap.add_argument("--show-masks", type=int, default=3)
    args = ap.parse_args(argv)
    # article text has curly quotes and Indic characters; never crash on a console that cannot show them
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cfg = load_config(args.config, data_path=args.data)
    examples = load_examples(cfg.data_path)
    print(describe(examples))

    pairs = assign_groups(examples, cfg.group_threshold)
    print(f"\nnear-duplicate pairs (cosine >= {cfg.group_threshold}), kept in one fold: {len(pairs)}")
    for a, b, s in pairs:
        print(f"  {a} ~ {b}  {s}")
    folds = make_folds(examples, cfg.folds, cfg.fold_seed)
    print("\nfolds:")
    for row in fold_table(examples, folds):
        print("  " + ", ".join(f"{k} {v}" for k, v in row.items()))

    try:
        from .model import load_tokenizer
        tok = load_tokenizer(args.model or cfg.model_name)
        lens = sorted(len(tok(e.title, e.lede)["input_ids"]) for e in examples)
        over = sum(n > cfg.max_length for n in lens)
        print(f"\ntokens per row: min {lens[0]}, median {lens[len(lens) // 2]}, max {lens[-1]}; "
              f"{over} row(s) longer than max_length {cfg.max_length}")
    except Exception as e:  # tokenizer download can fail offline; the rest of the report still helps
        print(f"\n(token lengths skipped: {e})")

    masked = [e for e in examples if e.masked_sources]
    print(f"\nsource masking changed {len(masked)} row(s)")
    for e in masked[:args.show_masks]:
        print(f"  {e.id}: {e.title} | {e.lede[:160]}")

    rm = RoleMasker(p=1.0)
    with_names = [e for e in examples if rm.count(e.title + " " + e.lede)]
    print(f"\nrows with party/leader names that role masking can replace: {len(with_names)}")
    rng = random.Random(0)
    for e in rng.sample(with_names, min(args.show_masks, len(with_names))):
        print(f"  {e.id}: {rm.mask_all(e.title)}")

    neutral = [e for e in examples if e.gender == "Neutral"]
    swapped = [(e, gender_swap(e.title), gender_swap(e.lede)) for e in neutral]
    swapped = [s for s in swapped if s[1][1] + s[2][1]]
    print(f"\nNeutral rows that get a gender-swapped copy: {len(swapped)} of {len(neutral)}")
    for e, (t, _), (ld, _) in swapped[:args.show_swaps]:
        print(f"  {e.id}\n    before: {e.title} | {e.lede[:200]}\n    after:  {t} | {ld[:200]}")


if __name__ == "__main__":
    main()
