"""Reference server for the classifier API.

    NEWSBIAS_MODEL_DIR=models/bias-v1 NEWSBIAS_API_KEY=<secret> uvicorn newsbias.api.server:app --port 8000

Endpoints:
    POST /v1/classify   needs the header ``X-API-Key``; body and response in newsbias/api/schema.py
    GET  /health        no key; reports whether the model is loaded

The model is loaded once at start-up. Inference runs one request at a time (a lock), because a CPU host gains
nothing from running two 278M-parameter forward passes side by side and would risk running out of memory.
The API key is compared in constant time. The service refuses to start without a key.
"""
import hmac
import os
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from .schema import ClassifyRequest, ClassifyResponse, ErrorResponse, Health

_state = {"predictor": None, "error": None}
_lock = threading.Lock()


def _load():
    from ..predictor import Predictor
    model_dir = os.environ.get("NEWSBIAS_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("NEWSBIAS_MODEL_DIR is not set")
    steps = int(os.environ.get("NEWSBIAS_IG_STEPS", "16"))
    return Predictor(model_dir, ig_steps=steps)


@asynccontextmanager
async def lifespan(app):
    if not os.environ.get("NEWSBIAS_API_KEY"):
        raise RuntimeError("NEWSBIAS_API_KEY is not set; refusing to start without authentication")
    if _state["predictor"] is None:
        try:
            _state["predictor"] = _load()
        except Exception as e:  # keep serving /health so the failure is visible
            _state["error"] = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="News bias classifier", version="1", lifespan=lifespan,
              description="Political and gender bias labels for the headline and lede of a news article.")


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("NEWSBIAS_API_KEY", "")
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or wrong API key")


@app.get("/health", response_model=Health)
def health():
    p = _state["predictor"]
    if p is not None:
        return Health(status="ok", model_version=p.version)
    return Health(status="error" if _state["error"] else "loading")


@app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(require_key)],
          responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def classify(req: ClassifyRequest):
    p = _state["predictor"]
    if p is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        with _lock:
            r = p.classify(req.headline, req.body, explain_axes=req.explain)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return ClassifyResponse(
        model_version=p.version, request_id=req.request_id,
        input={"headline": r["headline"], "lede": r["lede"], "lede_words": r["lede_words"],
               "truncated": r["truncated"]},
        political=r["axes"]["political"], gender=r["axes"]["gender"])
