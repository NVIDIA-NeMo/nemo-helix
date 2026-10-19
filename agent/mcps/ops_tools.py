"""Meridian Compute ops tools, exposed as an MCP stdio server.

Three tools, in the order the agent uses them:

- route_incident: ask a model which runbook owns the incident and what action to
  take. The model is a setting (ROUTER_MODEL), reached through the NeMo Helix
  inference gateway. This is the one step the experiment swaps between the
  large model, the untuned small model and the fine-tuned decision model.
- get_status: node or service health from the fleet fixture.
- open_ticket / page_team: record the action. Real teams wire their ticketing here.

Settings (environment variables):
  NHX_BASE_URL        Helix URL, default http://localhost:8080
  NHX_WORKSPACE       workspace for the inference gateway, default "default"
  ROUTER_MODEL        gateway model name (`nemo models list`), default
                      nvidia-nemotron-3-ultra-550b-a55b (Nemotron 3 Ultra on build.nvidia.com)
  ROUTER_MODE         "chat" (ask for the letter as text) or "logits" (read option
                      probabilities from a completions call), default chat
  ROUTER_WITH_CONTEXT "1" to include Meridian's routing notes in the prompt (arm A+)
  ROUTER_TEMPERATURE  temperature scaling for logits mode, default 1.0
  MERIDIAN_TAXONOMY   path to taxonomy.json
  MERIDIAN_FLEET      path to the fleet fixture
  MERIDIAN_ACTION_LOG where ticket/page records are appended (JSONL)
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from mcps import prompting

mcp = FastMCP("meridian-ops")

HERE = Path(__file__).resolve().parent
BASE_URL = os.environ.get("NHX_BASE_URL", "http://localhost:8080").rstrip("/")
WORKSPACE = os.environ.get("NHX_WORKSPACE", "default")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "nvidia-nemotron-3-ultra-550b-a55b")
ROUTER_MODE = os.environ.get("ROUTER_MODE", "chat")
ROUTER_WITH_CONTEXT = os.environ.get("ROUTER_WITH_CONTEXT", "0") == "1"
ROUTER_TEMPERATURE = float(os.environ.get("ROUTER_TEMPERATURE", "1.0"))
# Per-question temperatures fitted on the calibration split (eval/analyze_results.py).
ROUTER_TEMPERATURE_RUNBOOK = float(os.environ.get("ROUTER_TEMPERATURE_RUNBOOK", os.environ.get("ROUTER_TEMPERATURE", "1.0")))
ROUTER_TEMPERATURE_ACTION = float(os.environ.get("ROUTER_TEMPERATURE_ACTION", os.environ.get("ROUTER_TEMPERATURE", "1.0")))
# Optional: call a specific OpenAI-compatible base URL instead of the local gateway,
# e.g. a fine-tuned deployment's provider route on a remote Helix:
#   https://<helix>/apis/inference-gateway/v2/workspaces/<ws>/provider/<deployment>/-/v1
# For an authenticated Helix the token comes from `nemo auth token` at call time,
# so no secret is stored in the agent config.
ROUTER_URL = os.environ.get("ROUTER_URL", "").rstrip("/")
FLEET_PATH = Path(os.environ.get("MERIDIAN_FLEET", HERE / "fixtures" / "fleet.json"))
# Outside the agent directory on purpose: everything next to agent.yaml is
# uploaded with the agent, and "artifacts" is a reserved runtime directory.
ACTION_LOG = Path(os.environ.get("MERIDIAN_ACTION_LOG", HERE.parents[1] / "runs" / "actions.jsonl"))

TAXONOMY = prompting.load_taxonomy(os.environ.get("MERIDIAN_TAXONOMY"))
GATEWAY = ROUTER_URL or f"{BASE_URL}/apis/inference-gateway/v2/workspaces/{WORKSPACE}/openai/-/v1"


_TOKEN_CACHE: dict[str, Any] = {"token": None, "at": 0.0}


def _auth_headers() -> dict[str, str]:
    if not ROUTER_URL or ROUTER_URL.startswith(("http://localhost", "http://127.0.0.1")):
        return {}
    if _TOKEN_CACHE["token"] and time.time() - _TOKEN_CACHE["at"] < 300:
        return {"Authorization": f"Bearer {_TOKEN_CACHE['token']}"}
    import subprocess
    from urllib.parse import urlparse
    host = urlparse(ROUTER_URL)
    cli = [os.environ.get("NEMO_CLI", "nemo"), "--base-url", f"{host.scheme}://{host.netloc}", "auth"]

    def _token() -> str:
        return subprocess.run(cli + ["token"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[-1]

    def _expires_soon(tok: str) -> bool:
        import base64
        import json as _json
        try:
            part = tok.split(".")[1]
            claims = _json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
            return claims.get("exp", 0) < time.time() + 300
        except Exception:  # noqa: BLE001
            return False

    try:
        tok = _token()
        # `nemo auth token` returns the stored token even after it expires; refresh it first.
        if _expires_soon(tok):
            subprocess.run(cli + ["refresh"], capture_output=True, text=True, timeout=60)
            tok = _token()
        _TOKEN_CACHE.update(token=tok, at=time.time())
        return {"Authorization": f"Bearer {tok}"}
    except Exception:  # noqa: BLE001
        return {}


# ── routing ─────────────────────────────────────────────────────────────────

def _chat_letter(prompt: str, valid: str) -> tuple[str, float, dict[str, Any]]:
    """Ask the model for a letter as text. Confidence is not available in this mode
    (reasoning models don't return usable logprobs), so it is reported as None."""
    body = {
        "model": ROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{GATEWAY}/chat/completions", json=body, headers=_auth_headers())
        resp.raise_for_status()
    data = resp.json()
    text = (data["choices"][0]["message"].get("content") or "").strip()
    match = re.search(rf"\b([{valid}])\b", text)
    letter = match.group(1) if match else ""
    return letter, None, {"raw": text[:200], "usage": data.get("usage")}


def _logit_letter(prompt: str, valid: str, temperature: float = 1.0) -> tuple[str, float, dict[str, Any]]:
    """One forward pass: read the probability of each option letter from the
    next-token logprobs, apply temperature scaling, pick the max."""
    body = {
        "model": ROUTER_MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 20,
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{GATEWAY}/completions", json=body, headers=_auth_headers())
        resp.raise_for_status()
    data = resp.json()
    top = data["choices"][0]["logprobs"]["top_logprobs"][0]
    scores = {}
    for tok, lp in top.items():
        t = tok.strip()
        if len(t) == 1 and t in valid:
            scores[t] = max(scores.get(t, -1e9), lp)
    if not scores:
        return "", 0.0, {"top_logprobs": top}
    scaled = {k: v / temperature for k, v in scores.items()}
    z = max(scaled.values())
    denom = sum(math.exp(v - z) for v in scaled.values())
    probs = {k: math.exp(v - z) / denom for k, v in scaled.items()}
    letter = max(probs, key=probs.get)
    return letter, probs[letter], {"probs": probs, "usage": data.get("usage")}


def _ask(prompt: str, valid: str, temperature: float = 1.0) -> tuple[str, float | None, dict[str, Any]]:
    return _logit_letter(prompt, valid, temperature) if ROUTER_MODE == "logits" else _chat_letter(prompt, valid)


@mcp.tool()
def route_incident(incident_text: str) -> dict[str, Any]:
    """Decide which runbook owns an incident and what action to take.

    Returns runbook (RB-xx), team, action, confidence (0 to 1, or null when the
    model does not expose probabilities), latency in ms and the model used.
    Always call this first, before gathering evidence or acting.
    """
    t0 = time.perf_counter()
    q1 = prompting.render_runbook_question(incident_text, TAXONOMY, ROUTER_WITH_CONTEXT)
    rb_letter, rb_conf, rb_meta = _ask(q1, prompting.LETTERS, ROUTER_TEMPERATURE_RUNBOOK)
    runbooks = dict(prompting.runbook_options(TAXONOMY))
    rb = runbooks.get(rb_letter)
    if rb is None:
        return {"error": "no runbook chosen", "model": ROUTER_MODEL, "detail": rb_meta}
    q2 = prompting.render_action_question(incident_text, rb["id"], TAXONOMY, ROUTER_WITH_CONTEXT)
    act_letter, act_conf, act_meta = _ask(q2, prompting.ACTION_LETTERS, ROUTER_TEMPERATURE_ACTION)
    actions = dict(prompting.action_options(TAXONOMY))
    act = actions.get(act_letter)
    return {
        "runbook": rb["id"],
        "runbook_name": rb["name"],
        "team": rb["team"],
        "action": act["id"] if act else None,
        "confidence": rb_conf,
        "action_confidence": act_conf,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "model": ROUTER_MODEL,
        "mode": ROUTER_MODE,
    }


# ── evidence ────────────────────────────────────────────────────────────────

def _fleet() -> dict[str, Any]:
    with open(FLEET_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@mcp.tool()
def get_status(target: str) -> dict[str, Any]:
    """Current health of a node (e.g. dgx-07, pod-b) or service (e.g. Vault, Beacon)
    named in the incident. Returns the record, or found=false."""
    fleet = _fleet()
    key = target.strip().lower()
    for kind in ("nodes", "services", "clusters"):
        for name, record in fleet.get(kind, {}).items():
            if name.lower() == key:
                return {"found": True, "kind": kind[:-1], "name": name, **record}
    return {"found": False, "name": target}


# ── actions ─────────────────────────────────────────────────────────────────

def _record(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    ACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"id": f"{kind[:3].upper()}-{uuid.uuid4().hex[:8]}", "kind": kind, "ts": time.time(), **payload}
    with open(ACTION_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


@mcp.tool()
def open_ticket(runbook: str, summary: str, severity: str = "S3") -> dict[str, Any]:
    """Open a ticket for the owning team of a runbook. Use for ticket-business-hours."""
    return _record("ticket", {"runbook": runbook, "summary": summary, "severity": severity})


@mcp.tool()
def page_team(runbook: str, summary: str, escalate: bool = False) -> dict[str, Any]:
    """Page the on-call for a runbook's team. Set escalate=true to also page the
    incident commander (escalate-incident-commander). Use for page-oncall."""
    return _record("page", {"runbook": runbook, "summary": summary, "escalate": escalate})


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
