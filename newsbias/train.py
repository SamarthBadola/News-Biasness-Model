"""Training loop and prediction.

What happens in one training run (one cross-validation fold, or the final fit on all rows):

1. Build training items from the examples. With ``gender_swap_neutral`` every Neutral row that contains gendered
   words gets a swapped copy that trains only the gender head (its political label is not re-checked, so the
   political loss ignores it).
2. Class weights per axis, ``(N / (K * n_class)) ** class_weight_power``, from the items of this run only. Power
   1 is fully balanced: a mistake on one of the 18 Misandrist rows costs about as much as all the mistakes on
   Neutral rows together. This is the main lever for per-class recall.
3. Optional sample weights from the annotator's confidence (high 1.0, medium 0.8, low 0.6).
4. Each epoch, shuffle, and role-mask party and leader names at random (``role_mask_p``).
5. AdamW with layer-wise learning-rate decay, linear warm-up then decay, gradient clipping at 1.0.
   A fixed number of epochs: there is no early stopping, because picking an epoch on the held-out fold would
   leak it into the estimate.
"""
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .augment import RoleMasker, gender_swap
from .data import CONFIDENCE_WEIGHT
from .labels import AXES, label_index
from .model import BiasClassifier, encode, freeze, load_tokenizer, model_inputs, param_groups


@dataclass
class Item:
    title: str
    lede: str
    targets: dict      # axis -> class index
    mask: dict         # axis -> 1.0 if this item trains that head, else 0.0
    weight: float = 1.0
    source_id: str = ""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_threads(cfg):
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    elif os.cpu_count():
        # physical cores usually beat hyper-threads for this workload
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_items(examples, cfg):
    items = []
    for e in examples:
        w = 1.0
        if cfg.confidence_weighting:
            w = min(CONFIDENCE_WEIGHT[e.confidence[a]] for a in AXES)
        items.append(Item(e.title, e.lede, {a: label_index(a, getattr(e, a)) for a in AXES},
                          {a: 1.0 for a in AXES}, w, e.id))
        if cfg.gender_swap_neutral and e.gender == "Neutral":
            t, n1 = gender_swap(e.title)
            ld, n2 = gender_swap(e.lede)
            if n1 + n2:
                items.append(Item(t, ld, {a: label_index(a, getattr(e, a)) for a in AXES},
                                  {"political": 0.0, "gender": 1.0}, w, e.id + "~swap"))
    return items


def class_weights(items, power):
    out = {}
    for axis, classes in AXES.items():
        counts = np.zeros(len(classes))
        for it in items:
            if it.mask[axis]:
                counts[it.targets[axis]] += 1
        n, k = counts.sum(), len(classes)
        w = (n / (k * np.maximum(counts, 1))) ** power
        w[counts == 0] = 0.0
        out[axis] = torch.tensor(w, dtype=torch.float32)
    return out


def batch_loss(logits, batch_items, weights, cfg, dev):
    total = 0.0
    parts = {}
    sw = torch.tensor([it.weight for it in batch_items], device=dev)
    for axis in AXES:
        y = torch.tensor([it.targets[axis] for it in batch_items], device=dev)
        m = torch.tensor([it.mask[axis] for it in batch_items], device=dev)
        w = weights[axis].to(dev)
        per = F.cross_entropy(logits[axis].float(), y, weight=w, reduction="none")
        denom = (w[y] * m * sw).sum()
        if denom.item() == 0:
            continue
        loss = (per * m * sw).sum() / denom
        parts[axis] = loss.item()
        total = total + cfg.loss_weights.get(axis, 1.0) * loss
    return total, parts


def train_model(examples, cfg, seed, log=print, model_name=None):
    """Fine-tune on ``examples`` and return ``(model, tokenizer, history)``."""
    from transformers import get_linear_schedule_with_warmup

    setup_threads(cfg)
    set_seed(seed)
    name = model_name or cfg.model_name
    dev = device()
    tokenizer = load_tokenizer(name)
    model = BiasClassifier.from_base(name, cfg.head_dropout)
    freeze(model, cfg.freeze_embeddings, cfg.freeze_layers)
    model.to(dev)

    items = build_items(examples, cfg)
    weights = class_weights(items, cfg.class_weight_power)
    masker = RoleMasker(p=cfg.role_mask_p) if cfg.role_mask_p > 0 else None
    rng = random.Random(seed)

    optim = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.head_lr, cfg.layer_decay, cfg.weight_decay))
    steps_per_epoch = math.ceil(len(items) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total_steps), total_steps)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  training on {len(items)} items ({len(examples)} rows), {n_train / 1e6:.1f}M trainable parameters, "
        f"{total_steps} steps, class weights " +
        "; ".join(f"{a}: " + ", ".join(f"{c} {w:.2f}" for c, w in zip(AXES[a], weights[a].tolist())) for a in AXES))

    history = []
    autocast = torch.autocast("cpu", dtype=torch.bfloat16) if cfg.cpu_bf16 and dev.type == "cpu" else None
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        order = items[:]
        rng.shuffle(order)
        sums, n_batches = {a: 0.0 for a in AXES}, 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = order[start:start + cfg.batch_size]
            titles = [masker.mask(it.title, rng) if masker else it.title for it in chunk]
            ledes = [masker.mask(it.lede, rng) if masker else it.lede for it in chunk]
            enc = {k: v.to(dev) for k, v in model_inputs(encode(tokenizer, titles, ledes, cfg.max_length)).items()}
            if autocast:
                with autocast:
                    logits = model(**enc)
            else:
                logits = model(**enc)
            loss, parts = batch_loss(logits, chunk, weights, cfg, dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            for a, v in parts.items():
                sums[a] += v
            n_batches += 1
        rec = {"epoch": epoch + 1, "seconds": round(time.time() - t0, 1)}
        rec.update({f"loss_{a}": round(sums[a] / max(n_batches, 1), 4) for a in AXES})
        history.append(rec)
        log(f"  epoch {epoch + 1}/{cfg.epochs}  " + "  ".join(f"{k} {v}" for k, v in rec.items() if k != "epoch"))
    model.eval()
    return model, tokenizer, history


@torch.inference_mode()
def predict_logits(model, tokenizer, pairs, cfg):
    """``pairs`` is a list of (title, lede). Returns {axis: np.ndarray of shape (n, n_classes)}."""
    model.eval()
    dev = next(model.parameters()).device
    out = {a: [] for a in AXES}
    for start in range(0, len(pairs), cfg.eval_batch_size):
        chunk = pairs[start:start + cfg.eval_batch_size]
        enc = encode(tokenizer, [p[0] for p in chunk], [p[1] for p in chunk], cfg.max_length)
        logits = model(**{k: v.to(dev) for k, v in model_inputs(enc).items()})
        for a in AXES:
            out[a].append(logits[a].float().cpu().numpy())
    return {a: np.concatenate(v) if v else np.zeros((0, len(AXES[a]))) for a, v in out.items()}


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
