# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone jailbreak-detection model server.

This is the deployable unit the platform controller manages. It exposes the
HTTP contract the ``nemoguardrails`` jailbreak-detection rail expects, so the
library needs **zero changes** — point
``rails.config.jailbreak_detection.nim_base_url`` at this server and set
``nim_server_endpoint`` to ``/v1/classify``.

Contract:

- ``POST /v1/classify``  body ``{"input": "<prompt>"}``  →
  ``{"jailbreak": <bool>, "score": <float>}``
- ``GET  /v1/health/live``  → ``{"object": "health-response", "message": "live"}``
  (200 whenever the process is up)
- ``GET  /v1/health/ready`` → ``{"object": "health-response", "message": "ready"}``
  (503 with ``"message": "not ready"`` until the model is loaded)

Runs inside the model container with no dependency on ``nemo_helix``; the
classifier is imported relative to this directory so the image can copy just
``model/`` plus its ``requirements.txt``.
"""

from __future__ import annotations

import logging
import os

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

try:  # package import (tests)
    from .classifier import JailbreakClassifier
except ImportError:  # flat import (container: `python server.py` from /app)
    from classifier import JailbreakClassifier

logger = logging.getLogger(__name__)

app = FastAPI(title="NeMo Jailbreak Detect", version="0.1.0")
cli_app = typer.Typer(help="NeMo jailbreak-detection model server.")


def _resolve_model_id() -> str:
    """Resolve the advertised model id, in precedence order.

    1. ``JAILBREAK_MODEL_ID`` — explicit, correctly-named override (this server is
       not a NIM, so it gets its own knob).
    2. ``NIM_SERVED_MODEL_NAME`` — injected by the NeMo Models controller from the
       deployment config's ``model_spec`` (``{model_namespace}/{model_name}``). Honoring
       it means discovery auto-matches the base id the controller expects, so the
       operator declares the identity once in ``model_spec`` and picks the workspace
       (e.g. ``default``) via ``model_namespace`` — no separate server config needed.
    3. The upstream default, for standalone/local runs with neither set.
    """
    return (
        os.environ.get("JAILBREAK_MODEL_ID")
        or os.environ.get("NIM_SERVED_MODEL_NAME")
        or "nvidia/nemoguard-jailbreak-detect"
    )


MODEL_ID = _resolve_model_id()

# Loaded once at startup and reused across requests.
_classifier: JailbreakClassifier | None = None


class ClassifyRequest(BaseModel):
    """Request body for ``/v1/classify``.

    Bounds (``min_length=1``, ``max_length=16777216``) reject empty input (the embedder
    would otherwise burn a forward pass on it) and absurdly large bodies at the API
    layer (the tokenizer truncates to 2048 tokens anyway). FastAPI returns 422 when
    these bounds are violated.
    """

    input: str = Field(min_length=1, max_length=16_777_216)


class ClassifyResponse(BaseModel):
    jailbreak: bool
    score: float


def get_classifier() -> JailbreakClassifier:
    """Return the process-global classifier, loading it on first use."""
    global _classifier
    if _classifier is None:
        _classifier = JailbreakClassifier(device=os.environ.get("JAILBREAK_CHECK_DEVICE"))
    return _classifier


@app.get("/v1/health/live")
def health_live() -> dict[str, str]:
    """Liveness: the process is up and serving HTTP, independent of model load."""
    return {"object": "health-response", "message": "live"}


@app.get("/v1/health/ready")
def health_ready(response: Response) -> dict[str, str]:
    """Readiness: only ready once the classifier is loaded.

    ``start`` loads the model before uvicorn accepts traffic, so in normal operation
    this returns ready as soon as the server is reachable. The 503 branch is a
    defensive guard for the unusual case of running the ASGI app without preloading
    (e.g. ``uvicorn server:app`` directly), so a readiness probe never reports ready
    before the model is actually loaded.
    """
    if _classifier is None:
        response.status_code = 503
        return {"object": "health-response", "message": "not ready"}
    return {"object": "health-response", "message": "ready"}


@app.get("/v1/models")
def list_models() -> dict:
    """OpenAI-style model discovery. Static — this server hosts a single model."""
    owned_by = MODEL_ID.split("/", 1)[0] if "/" in MODEL_ID else "nvidia"
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": owned_by}],
    }


_MALFORMED_INPUT_DETAIL = (
    "Received malformed input. /v1/classify expects JSON with a single, string-valued field named `input`."
)


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    try:
        classification, score = get_classifier()(request.input)
    except ValueError as exc:
        # A malformed prompt that breaks tokenization/inference is a client error
        # (400), not a server fault (500). Log only the exception type: the message can
        # embed the raw prompt, which must not leak into server logs.
        logger.info("%s (%s)", _MALFORMED_INPUT_DETAIL, type(exc).__name__)
        raise HTTPException(status_code=400, detail=_MALFORMED_INPUT_DETAIL) from exc
    return ClassifyResponse(jailbreak=classification, score=score)


@cli_app.callback()
def _main() -> None:
    """NeMo jailbreak-detection model server."""
    # Present so Typer keeps `start` as an explicit subcommand (a single-command
    # Typer app otherwise collapses and rejects the command name).


@cli_app.command()
def start(
    port: int = typer.Option(default=8000, help="Port to listen on."),
    host: str = typer.Option(default="0.0.0.0", help="Host/IP to bind."),
) -> None:
    """Start the model server.

    The model always loads before the server accepts traffic. There is deliberately no
    lazy "load on first request" mode: a readiness-gated orchestrator never routes that
    first request to an unready pod, so a lazy server would never flip to ready
    (deadlock). Loading up front also fail-fasts on a bad model or download instead of
    on the first request.
    """
    # Surface our INFO startup logs (model download/load progress) on the console
    # before uvicorn configures its own logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Weights are HF-cache-aware, so after the first run this is just a load into memory.
    logger.info("Loading model before serving (slow on first run: downloads weights)...")
    get_classifier()
    logger.info("Model loaded.")
    logger.info("Starting HTTP server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli_app()
