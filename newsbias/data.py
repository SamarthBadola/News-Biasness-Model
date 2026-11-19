"""Load the annotated corpus and turn it into model-ready examples.

The source file (news-bias-dataset/data/annotated/art_A1.jsonl) is only read, never changed. Preprocessing:

1. Check every row: unique id, ``lede_status`` ok, a valid political and gender label. A bad row stops the run
   with a list of problems instead of being skipped silently.
2. Keep only what the annotator saw (headline and lede) plus bookkeeping fields. The outlet is kept for the
   shortcut diagnostics in the report but never reaches the model. Communal/caste and cultural are dropped.
3. Normalise text (NFC, no control characters, single spaces) and mask outlet and agency names.
4. Find near-duplicate stories (same event, same wire copy) so cross-validation can keep them in one fold.
"""
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .labels import AXES
from .text import mask_sources, normalize

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.8, "low": 0.6}


class DataError(ValueError):
    pass


@dataclass
class Example:
    id: str
    title: str          # normalised and source-masked: this is what the model reads
    lede: str
    political: str
    gender: str
    confidence: dict    # axis -> "high" | "medium" | "low"
    outlet: str = ""
    article_type: str = ""
    spans: dict = field(default_factory=dict)   # axis -> list of highlighted phrases (annotator evidence)
    group: str = ""     # cross-validation group; near-duplicates share one
    masked_sources: int = 0


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataError(f"{path}:{n}: not valid JSON ({e})") from e
    return rows


def file_sha256(path):
    """SHA-256 of the file with line endings normalised, so a Windows checkout (CRLF) and a Linux one (LF) of the
    same data give the same fingerprint."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_rows(rows):
    problems = []
    seen = set()
    for r in rows:
        rid = r.get("id", "<no id>")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        if r.get("lede_status") != "ok":
            problems.append(f"{rid}: lede_status is {r.get('lede_status')!r}, expected 'ok'")
        if not normalize(r.get("title")) or not normalize(r.get("lede")):
            problems.append(f"{rid}: empty title or lede")
        for axis, classes in AXES.items():
            lab = (r.get("labels") or {}).get(axis) or {}
            if lab.get("label") not in classes:
                problems.append(f"{rid}: {axis} label {lab.get('label')!r} is not one of {classes}")
            if lab.get("confidence") not in CONFIDENCE_WEIGHT:
                problems.append(f"{rid}: {axis} confidence {lab.get('confidence')!r} is not high/medium/low")
    return problems


def to_example(row):
    title = mask_sources(normalize(row["title"]), title=True)
    lede = mask_sources(normalize(row["lede"]))
    labels = row["labels"]
    return Example(
        id=row["id"],
        title=title.text,
        lede=lede.text,
        political=labels["political"]["label"],
        gender=labels["gender"]["label"],
        confidence={a: labels[a]["confidence"] for a in AXES},
        outlet=row.get("outlet", ""),
        article_type=row.get("article_type", ""),
        spans={a: [s.get("text", "") for s in labels[a].get("spans") or []] for a in AXES},
        masked_sources=len(title.edits) + len(lede.edits),
    )


def load_examples(path):
    rows = read_jsonl(path)
    if not rows:
        raise DataError(f"{path}: no rows")
    problems = check_rows(rows)
    if problems:
        raise DataError(f"{len(problems)} problem(s) in {path}:\n  " + "\n  ".join(problems[:50]))
    examples = [to_example(r) for r in rows]
    texts = Counter((e.title + " " + e.lede).lower() for e in examples)
    dupes = [t for t, c in texts.items() if c > 1]
    if dupes:
        raise DataError(f"{len(dupes)} exact duplicate text(s) in {path}; remove them in the dataset repo")
    return examples


def subsample(examples, n, seed=0):
    """``n`` rows for a quick smoke run, spread over every political x gender combination.

    Taking the first n rows of the file is not enough: the corpus was collected in batches, so the first 40 rows
    hold one Misogynistic row and no Misandrist ones, and a fold would end up training on a single class. This
    takes rows round-robin from each label combination, so rare classes appear as long as they exist.
    """
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    groups = {}
    for e in examples:
        groups.setdefault((e.political, e.gender), []).append(e)
    queues = [rng.sample(g, len(g)) for _, g in sorted(groups.items())]
    picked = []
    while len(picked) < n:
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop())
    keep = {e.id for e in picked}
    return [e for e in examples if e.id in keep]


def assign_groups(examples, threshold=0.5):
    """Put near-duplicate stories in one group (union-find over TF-IDF cosine similarity >= threshold).

    The dataset has no event ids yet, so this is how "same-event articles stay in one fold" is enforced. Returns
    the list of linked pairs with their similarity, for the report.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [e.title + " " + e.lede for e in examples]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english").fit_transform(docs)
    sim = cosine_similarity(tfidf)
    parent = list(range(len(examples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            if sim[i, j] >= threshold:
                pairs.append((examples[i].id, examples[j].id, round(float(sim[i, j]), 3)))
                parent[find(i)] = find(j)
    for i, e in enumerate(examples):
        e.group = examples[find(i)].id
    return pairs


def describe(examples):
    """Plain-text summary used by `python -m newsbias.prepare` and the report."""
    lines = [f"rows: {len(examples)}"]
    for axis, classes in AXES.items():
        c = Counter(getattr(e, axis) for e in examples)
        lines.append(f"{axis}: " + ", ".join(f"{k} {c[k]}" for k in classes))
    joint = Counter((e.political, e.gender) for e in examples)
    lines.append("political x gender: " + ", ".join(f"{p}/{g} {n}" for (p, g), n in sorted(joint.items())))
    lines.append(f"rows with an outlet or agency name masked: {sum(1 for e in examples if e.masked_sources)}")
    return "\n".join(lines)
