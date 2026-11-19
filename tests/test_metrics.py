import numpy as np

from newsbias.metrics import ece, evaluate_axis, fit_temperature, overlap_f1


def test_per_class_metrics():
    y = [0, 0, 1, 1, 2, 2, 2, 2]
    p = [0, 1, 1, 1, 2, 2, 2, 0]
    r = evaluate_axis("gender", y, p, n_boot=100)
    pc = r["per_class"]
    assert pc["Misogynistic"]["recall"] == 0.5 and pc["Misogynistic"]["precision"] == 0.5
    assert pc["Misandrist"]["recall"] == 1.0 and abs(pc["Misandrist"]["precision"] - 2 / 3) < 1e-9
    assert pc["Neutral"]["recall"] == 0.75
    assert abs(r["balanced_accuracy"] - (0.5 + 1 + 0.75) / 3) < 1e-9
    assert r["confusion"] == [[1, 1, 0], [0, 2, 0], [1, 0, 3]]
    lo, hi = r["per_class"]["Neutral"]["recall_ci"]
    assert 0 <= lo <= 0.75 <= hi <= 1


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(3000, 3)) * 2
    y = np.array([rng.choice(3, p=np.exp(z) / np.exp(z).sum()) for z in true_logits])
    t = fit_temperature(true_logits * 3, y)
    assert 2.5 < t < 3.5


def test_ece_perfect_is_zero():
    probs = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert ece(probs, [0, 1]) == 0.0


def test_overlap_f1():
    assert overlap_f1({"a", "b"}, {"b", "c"}) == 0.5
    assert overlap_f1(set(), {"a"}) == 0.0
