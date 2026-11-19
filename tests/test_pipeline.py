"""End to end with the tiny model: cross-validation, report, resume, final fit, prediction and the API."""
import json

import pytest

from newsbias import cv, final
from newsbias.labels import AXES


@pytest.fixture
def cv_run(tmp_path, tiny_config):
    results = tmp_path / "results"
    cv.main(["--config", str(tiny_config), "--run", "t", "--results", str(results)])
    return results


def test_cv_writes_complete_report(cv_run):
    run = cv_run / "t"
    m = json.loads((run / "metrics.json").read_text())
    assert m["complete"] and m["rows"] == 36
    for axis, classes in AXES.items():
        assert set(m["primary"][axis]["per_class"]) == set(classes)
        assert sum(m["primary"][axis]["per_class"][c]["support"] for c in classes) == 36
    assert m["temperatures"]["gender"] > 0
    assert "explanations" in m
    report = (run / "report.md").read_text()
    assert "## Political" in report and "## Gender" in report and "Shortcut checks" in report
    preds = list((run / "preds").glob("*.jsonl"))
    assert len(preds) == 3
    assert "lede" not in preds[0].read_text()   # no article text in committed results


def test_cv_resumes_and_refuses_changed_config(cv_run, tiny_config, capsys):
    cv.main(["--config", str(tiny_config), "--run", "t", "--results", str(cv_run)])
    assert capsys.readouterr().out.count("already done") == 3
    with pytest.raises(SystemExit):
        cv.main(["--config", str(tiny_config), "--run", "t", "--results", str(cv_run), "--epochs", "3"])


def test_final_model_predictor_and_api(cv_run, tiny_config, tmp_path, monkeypatch):
    out = tmp_path / "model"
    final.main(["--config", str(tiny_config), "--cv-run", "t", "--out", str(out), "--results", str(cv_run)])
    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["temperatures"] and meta["cv_run"] == "t"

    from newsbias.predictor import Predictor
    p = Predictor(out, ig_steps=4)
    body = "By Staff\nNEW DELHI: Opposition slams government over failed scheme, says NDTV.\n" \
           "Men are all the same, toxic men again.\nThird paragraph is ignored."
    r = p.classify("Critics say the government betrayed farmers", body)
    assert "Third" not in r["lede"] and "NEW DELHI" not in r["lede"]
    for axis in AXES:
        res = r["axes"][axis]
        assert abs(sum(res["probabilities"].values()) - 1) < 1e-3
        for h in res["highlights"]:
            src = r["headline"] if h["field"] == "headline" else r["lede"]
            assert src[h["start"]:h["end"]] == h["text"]
            assert 0 <= h["weight"] <= 1

    from fastapi.testclient import TestClient
    from newsbias.api import server
    monkeypatch.setenv("NEWSBIAS_API_KEY", "k" * 32)
    monkeypatch.setitem(server._state, "predictor", p)
    with TestClient(server.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        req = {"headline": "Critics say the government betrayed farmers", "body": body}
        assert client.post("/v1/classify", json=req).status_code == 401
        assert client.post("/v1/classify", json=req, headers={"X-API-Key": "wrong"}).status_code == 401
        ok = client.post("/v1/classify", json=req, headers={"X-API-Key": "k" * 32})
        assert ok.status_code == 200, ok.text
        js = ok.json()
        assert js["political"]["label"] in AXES["political"] and js["input"]["scope"] == "headline+lede"
        bad = client.post("/v1/classify", json={**req, "body": "tiny"}, headers={"X-API-Key": "k" * 32})
        assert bad.status_code == 422
        extra = client.post("/v1/classify", json={**req, "lede": "x"}, headers={"X-API-Key": "k" * 32})
        assert extra.status_code == 422


def test_error_listing(cv_run, tiny_config, capsys):
    from newsbias import errors
    m = json.loads((cv_run / "t" / "metrics.json").read_text())
    d = m["diagnostics"]
    assert set(d["confidence"]["gender"]) <= {"high", "medium", "low"}
    n_wrong = sum(1 for e in d["errors"]["gender"])
    assert n_wrong == round((1 - m["primary"]["gender"]["accuracy"]) * 36)
    errors.main([str(cv_run / "t"), "--axis", "gender"])
    out = capsys.readouterr().out
    assert f"gender: {n_wrong} error(s)" in out
    assert "## Where the errors are" in (cv_run / "t" / "report.md").read_text()
