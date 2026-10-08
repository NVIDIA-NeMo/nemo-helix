# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classified war-game failures + the classifiers the run boundary uses.

Every failure that can affect a run's results is reduced to a :class:`RunFailure` — a stable
``category`` plus an operator-facing ``message`` and ``remediation`` — so :meth:`AgentHardenerRunJob.run`
records the *cause* on every channel the user sees (the run record, the platform job's
``error_details``) instead of a bare "exited with code 1". Failures raise :class:`AgentHardenerRunError`
at their source (subclass of ``RuntimeError`` so existing ``pytest.raises(RuntimeError)`` still hold);
anything else reaching the boundary is classified by :func:`classify_exception`. Subprocess failures
are classified from agent-hardener's own ``run-error.json`` (:func:`read_run_error`), falling back to the
exit code + log tail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Env var pointing agent-hardener's CLI at the path where it should dump a structured failure (run-error.json).
# The plugin sets it for the primary up/run/serve subprocesses and reads the file back on a non-zero exit.
AGENT_HARDENER_ERROR_FILE_ENVVAR = "AGENT_HARDENER_ERROR_FILE"

# Stable failure categories, shared in spirit with agent-hardener's own taxonomy (agent_hardener.errors).
CATEGORY_PROVISIONING = "provisioning"
CATEGORY_MISSING_CREDENTIAL = "missing_credential"
CATEGORY_MANIFEST = "manifest"
CATEGORY_FILESET = "fileset"
CATEGORY_SANDBOX = "sandbox"
CATEGORY_VICTIM_UNAVAILABLE = "victim_unavailable"
CATEGORY_SYNTH_SERVICE = "synth_service"
CATEGORY_HITL_TIMEOUT = "hitl_timeout"
CATEGORY_ATTACKER_FAILED = "attacker_failed"
CATEGORY_NETWORK = "network"
CATEGORY_MODEL_UNAVAILABLE = "model_unavailable"
# The war-game ran the full attack/defend/validate cycle but the round did not pass validation
# (some attacks were not blocked and/or some benign requests failed). agent-hardener exits non-zero and
# writes no structured error, so this is a *result*, not a crash — distinct from a victim/phase failure.
CATEGORY_VALIDATION_FAILED = "validation_failed"
CATEGORY_UNEXPECTED = "unexpected"

# Default operator-facing next step per category; a call site may override with a more specific one.
CATEGORY_REMEDIATION: dict[str, str] = {
    CATEGORY_PROVISIONING: "Run `nemo agent-hardener setup` on the host that executes this job, then retry.",
    CATEGORY_MISSING_CREDENTIAL: "Provide the required secret (e.g. `nemo secrets create`) or set it in the environment.",
    CATEGORY_MANIFEST: "Re-create the manifest or fix the target agent reference, then retry.",
    CATEGORY_FILESET: "Re-upload the file and verify the Files service is reachable, then retry.",
    CATEGORY_SANDBOX: "Check the Docker daemon and the OpenShell gateway on the host, then retry.",
    CATEGORY_VICTIM_UNAVAILABLE: "Inspect the victim agent log; a malformed workflow or policy often stops it loading.",
    CATEGORY_SYNTH_SERVICE: "Check the benign-suite service log (serve.log) on the host, then retry.",
    CATEGORY_HITL_TIMEOUT: "Resubmit the run and respond to the interview/review prompt before it times out.",
    CATEGORY_ATTACKER_FAILED: "The attacker did not finish (often a timeout on a heavy agent); the 0-hit result "
    "is not valid. Re-run, raising the attacker timeout (garak.timeout_s) or lowering attack_intensity.",
    CATEGORY_NETWORK: "Check connectivity to the NeMo Platform control plane, then retry.",
    CATEGORY_MODEL_UNAVAILABLE: "Check the model name, endpoint URL, and API key for the flagged group; "
    "the error lists the models those credentials can reach.",
    CATEGORY_VALIDATION_FAILED: "The war-game completed but the round did not pass validation — some "
    "attacks were not blocked and/or some benign requests failed. Review the scorecard; harden further "
    "or adjust the benign suite.",
    CATEGORY_UNEXPECTED: "See the run log for details; if it persists, file a bug.",
}


@dataclass(frozen=True)
class RunFailure:
    """A classified, user-facing war-game failure. ``stack`` carries diagnostic context (log tail/traceback)."""

    category: str
    message: str
    remediation: str = ""
    stack: str = ""

    def as_error_details(self) -> dict[str, str]:
        """The platform job ``error_details`` payload (matches the automodel/unsloth convention)."""
        return {"message": self.message, "type": self.category, "remediation": self.remediation}


class AgentHardenerRunError(RuntimeError):
    """A war-game failure raised at its source with a known :class:`RunFailure` category.

    Subclasses ``RuntimeError`` so call sites that previously raised ``RuntimeError`` (and the tests
    asserting it) keep working while gaining a classified category the run boundary can surface.
    """

    def __init__(self, category: str, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.remediation = CATEGORY_REMEDIATION.get(category, "") if remediation is None else remediation

    def as_failure(self, *, stack: str = "") -> RunFailure:
        return RunFailure(self.category, str(self), self.remediation, stack)


def classify_exception(exc: BaseException) -> RunFailure:
    """Classify an arbitrary exception that reached the run boundary into a :class:`RunFailure`.

    Typed :class:`AgentHardenerRunError`s carry their own category; an agent-resolution failure is a
    manifest problem; an httpx/transport error is a network problem; everything else is ``unexpected``
    (its ``str`` is shown, its type recorded in ``stack``).
    """
    if isinstance(exc, AgentHardenerRunError):
        return exc.as_failure(stack=_short_repr(exc))

    # Imported lazily to avoid a hard dependency in a module the whole job graph imports.
    from nemo_agent_hardener_plugin.agent_resolver import AgentResolutionError

    if isinstance(exc, AgentResolutionError):
        return _failure(CATEGORY_MANIFEST, str(exc))
    if _is_network_error(exc):
        return _failure(CATEGORY_NETWORK, str(exc) or exc.__class__.__name__)
    return _failure(CATEGORY_UNEXPECTED, str(exc) or exc.__class__.__name__, stack=_short_repr(exc))


def read_run_error(path: Path) -> RunFailure | None:
    """Parse agent-hardener's ``run-error.json`` (written by its CLI boundary) into a :class:`RunFailure`.

    Returns ``None`` when the file is absent or unreadable — the caller then falls back to the exit
    code + log tail. The file is trusted (agent-hardener wrote it), but parsing stays defensive.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    category = raw.get("category")
    category = category if isinstance(category, str) and category else CATEGORY_UNEXPECTED
    message = raw.get("message")
    message = message if isinstance(message, str) and message else "agent-hardener reported a failure"
    remediation = raw.get("remediation")
    remediation = (
        remediation if isinstance(remediation, str) and remediation else CATEGORY_REMEDIATION.get(category, "")
    )
    stack = raw.get("stack") if isinstance(raw.get("stack"), str) else ""
    return RunFailure(category, message, remediation, stack or "")


def classify_subprocess(returncode: int, log_tail: str, run_error: RunFailure | None) -> AgentHardenerRunError:
    """Turn a non-zero ``agent-hardener`` subprocess exit into a classified :class:`AgentHardenerRunError`.

    Prefers agent-hardener's structured ``run-error.json`` (precise category + remediation). Without it,
    falls back to a light heuristic over the log tail, defaulting to ``unexpected`` with the exit code.
    """
    if run_error is not None:
        exc = AgentHardenerRunError(run_error.category, run_error.message, remediation=run_error.remediation)
        return exc
    category = _heuristic_category(log_tail)
    if category == CATEGORY_VALIDATION_FAILED:
        message = "the war-game ran to completion but the round did not pass validation"
    else:
        message = f"agent-hardener exited with code {returncode}"
    return AgentHardenerRunError(category, message)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# Markers proving agent-hardener reached its final summary — i.e. the whole attack/defend/validate cycle
# ran. A non-zero exit *after* this is a round that didn't pass validation, not a crashed phase.
_RUN_COMPLETED_MARKERS: tuple[str, ...] = ("agent hardener final log", "validator results:")

# Cue → category, scanned in order, ONLY for runs that did NOT reach the final summary (a genuine
# mid-run crash). Victim cues are specific failure phrases: bare "victim" appears in healthy logs
# ("victim health ready") and must not trigger a false victim_unavailable.
_HEURISTIC_CUES: tuple[tuple[str, str], ...] = (
    (CATEGORY_ATTACKER_FAILED, "attacker execution failed"),
    (CATEGORY_ATTACKER_FAILED, "attacker agent status: failed"),
    (CATEGORY_VICTIM_UNAVAILABLE, "server disconnected"),
    (CATEGORY_VICTIM_UNAVAILABLE, "victim returned failure"),
    (CATEGORY_VICTIM_UNAVAILABLE, "victim unavailable"),
    (CATEGORY_VICTIM_UNAVAILABLE, "victim unreachable"),
    (CATEGORY_VICTIM_UNAVAILABLE, "openshell victim returned http"),
    (CATEGORY_SANDBOX, "sandbox"),
    (CATEGORY_SANDBOX, "docker"),
    (CATEGORY_SANDBOX, "openshell"),
    (CATEGORY_MISSING_CREDENTIAL, "api key"),
    (CATEGORY_MISSING_CREDENTIAL, "unauthorized"),
    (CATEGORY_NETWORK, "connection refused"),
    (CATEGORY_NETWORK, "timed out"),
)


def _heuristic_category(log_tail: str) -> str:
    """Best-effort category from the log tail when agent-hardener wrote no structured error."""
    lowered = log_tail.lower()
    # A completed run that exits non-zero failed *validation*, not a phase. Decide this first: the cue
    # scan's infra terms ("openshell", "docker") appear in every normal log and would otherwise win.
    if any(marker in lowered for marker in _RUN_COMPLETED_MARKERS):
        return CATEGORY_VALIDATION_FAILED
    for category, cue in _HEURISTIC_CUES:
        if cue in lowered:
            return category
    return CATEGORY_UNEXPECTED


def _failure(category: str, message: str, *, stack: str = "") -> RunFailure:
    return RunFailure(category, message, CATEGORY_REMEDIATION.get(category, ""), stack)


def _is_network_error(exc: BaseException) -> bool:
    """True for transport-level failures (httpx errors, connection/OS socket errors)."""
    try:
        import httpx

        if isinstance(exc, httpx.HTTPError):
            return True
    except ImportError:  # httpx is always present in practice; stay defensive
        pass
    return isinstance(exc, (ConnectionError, TimeoutError))


def _short_repr(exc: BaseException) -> str:
    return f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}"
