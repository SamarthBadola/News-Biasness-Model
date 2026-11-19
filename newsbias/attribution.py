"""Highlighted phrases with weights, for the explanation step.

Method: Integrated Gradients (Sundararajan et al., 2017) on the input word embeddings, for the logit of the
predicted class of each axis. The baseline replaces every non-special token with [PAD]. Attributions are summed
from sub-word tokens to words, and runs of adjacent high-scoring words become phrases.

Why not attention weights: attention shows where the model looked, not what pushed the decision, and several
studies have shown attention maps can be changed without changing predictions. Integrated Gradients measures
how much each word moved the class score, and its scores add up to the change in that score.

A phrase's ``weight`` is its share of all positive evidence for the class (0 to 1; the weights of the returned
phrases add up to at most 1). Offsets refer to the headline and lede strings the API echoes back.
"""
import re

import torch

from .model import model_inputs

_WORDS = re.compile(r"\S+")
_TRIM = "\"'“”‘’.,;:!?()[]{}—–-"


def _word_spans(text):
    return [(m.start(), m.end()) for m in _WORDS.finditer(text)]


def integrated_gradients(model, tokenizer, title, lede, axis, target, steps, max_length):
    """Per-token attributions. Returns (attr[L] tensor, offsets[L,2], sequence_ids list)."""
    enc = tokenizer(title, lede, truncation="longest_first", max_length=max_length, return_tensors="pt",
                    return_offsets_mapping=True)
    seq_ids = enc.sequence_ids(0)
    offsets = enc.pop("offset_mapping")[0].tolist()
    dev = next(model.parameters()).device
    enc = {k: v.to(dev) for k, v in model_inputs(enc).items()}
    ids = enc["input_ids"]
    special = torch.tensor([s is None for s in seq_ids], device=dev)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.where(special.unsqueeze(0), ids, torch.full_like(ids, pad_id))

    emb = model.encoder.get_input_embeddings()
    with torch.no_grad():
        x = emb(ids)
        x0 = emb(baseline_ids)
    alphas = (torch.arange(steps, device=dev, dtype=x.dtype) + 0.5) / steps
    path = (x0 + alphas.view(-1, 1, 1) * (x - x0)).detach().requires_grad_(True)
    extra = {k: v.expand(steps, -1) for k, v in enc.items() if k != "input_ids"}
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        logits = model(inputs_embeds=path, **extra)[axis]
        grads, = torch.autograd.grad(logits[:, target].sum(), path)
    model.train(was_training)
    attr = ((x - x0)[0] * grads.mean(0)).sum(-1).detach().cpu()
    return attr, offsets, seq_ids


def word_scores(attr, offsets, seq_ids, texts):
    """Sum token attributions into words. Returns {field_index: [(start, end, score), ...]}."""
    out = {}
    for fi, text in enumerate(texts):
        spans = _word_spans(text)
        scores = [0.0] * len(spans)
        j = 0
        for (s, e), sid, a in zip(offsets, seq_ids, attr.tolist()):
            if sid != fi or e <= s:
                continue
            while j < len(spans) and spans[j][1] <= s:
                j += 1
            if j < len(spans) and spans[j][0] <= s < spans[j][1]:
                scores[j] += a
        out[fi] = [(st, en, sc) for (st, en), sc in zip(spans, scores)]
    return out


def phrases(words, texts, min_rel=0.25, top_k=5):
    """Group adjacent words whose score is at least ``min_rel`` of the best word into weighted phrases."""
    positive = [sc for ws in words.values() for _, _, sc in ws if sc > 0]
    if not positive:
        return []
    best, total = max(positive), sum(positive)
    found = []
    for fi, ws in words.items():
        run = []
        for st, en, sc in ws + [(None, None, -1.0)]:
            if sc >= min_rel * best:
                run.append((st, en, sc))
                continue
            if run:
                start, end = run[0][0], run[-1][1]
                text = texts[fi]
                while start < end and text[start] in _TRIM:
                    start += 1
                while end > start and text[end - 1] in _TRIM:
                    end -= 1
                if end > start:
                    found.append({"field": fi, "start": start, "end": end,
                                  "weight": sum(max(s, 0.0) for _, _, s in run) / total})
                run = []
    found.sort(key=lambda p: -p["weight"])
    return found[:top_k]


def explain(model, tokenizer, title, lede, axis, target, steps=16, max_length=256, top_k=5):
    """Phrases for one axis and class, with offsets into ``title`` (field 0) and ``lede`` (field 1)."""
    attr, offsets, seq_ids = integrated_gradients(model, tokenizer, title, lede, axis, target, steps, max_length)
    return phrases(word_scores(attr, offsets, seq_ids, [title, lede]), [title, lede], top_k=top_k)
