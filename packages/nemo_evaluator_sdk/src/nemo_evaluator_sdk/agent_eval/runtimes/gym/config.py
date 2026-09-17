# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run configuration, and the Hydra grammar it is serialized into.

Gym is a Hydra application, so every setting reaches it as an ``++dotted.path=value`` argument.
That grammar is typed and unforgiving — quoting rules live here alongside the config model they
serialize, so the two cannot drift. Redaction lives here too: these values are recorded as run
provenance, and the override map is a free-form escape hatch a caller can put a credential into.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


DEFAULT_REWARD_KEY = "reward"
#: Gym's CLI, expected on PATH. Not configurable: these runner configs become serialized job specs,
#: and a path into somebody's venv is meaningless on the other side of that boundary. Note this name
#: is only ever *resolved*, never executed: :func:`gym_executable` turns it into an absolute path
#: once, and that path is what the subprocesses run — so a child whose PATH differs from ours cannot
#: end up executing a different Gym.
HYDRA_SUBDIR = "gym_hydra"
#: Where Gym writes its per-rollout model-call captures, under the run's work dir.
_CAPTURE_SUBDIR = "model_calls"
#: `gym env start`'s combined output, under the run's work dir. Named here because a *collection*
#: failure often has to point at it: the eval logs show the symptom, this shows the cause.


#: Dict keys Hydra reads back unchanged. Its ``dictKey`` rule accepts no quoting, so a key is
#: whatever the lexer makes of the bare text — this is deliberately narrower than what parses.
_HYDRA_DICT_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
#: Bare words the lexer types rather than reading as text, so they cannot serve as string keys.
_HYDRA_KEY_LITERALS = frozenset({"true", "false", "null", "inf", "nan"})


def _hydra_dict(value: Mapping[str, Any]) -> str:
    """Render a mapping as a Hydra dict container, ``{key:value,...}``.

    Reached for a mapping nested inside a container — ``[{"b": 1}]`` — where there is no dotted path
    to flatten onto, so the dict has to be spelled inline. Values recurse, so the typed spellings
    below hold at any depth.

    Keys are emitted bare, because Hydra's ``dictKey`` rule has no quoted form: ``{'b':1}`` does not
    parse at all. That leaves the key at the mercy of the lexer, which types it — ``{true:1}`` keys
    on the boolean ``True``, ``{1.5:1}`` on a float — and rejects ``:``, ``,``, brackets, and quotes
    outright. Anything outside the conservative shape above therefore raises here rather than
    silently keying the config on something the caller did not write.
    """
    rendered = []
    for key, item in value.items():
        if not isinstance(key, str) or not _HYDRA_DICT_KEY.match(key) or key.casefold() in _HYDRA_KEY_LITERALS:
            raise ValueError(
                f"Gym config override has dict key {key!r}, which Hydra's override grammar cannot "
                "express as a string: keys are unquoted, so only a leading letter or underscore "
                "followed by letters, digits, '_', '.', or '-' survives the round trip. Set this "
                "key through the override path instead of nesting it inside a list."
            )
        rendered.append(f"{key}:{hydra_scalar(item)}")
    return "{" + ",".join(rendered) + "}"


def hydra_scalar(value: Any) -> str:
    """Render a leaf value the way Hydra's override grammar reads it back.

    Hydra's grammar is typed, so an unquoted string is not necessarily a string: ``true`` parses as a
    boolean, ``null`` as ``None``, ``1.5`` as a float, ``a,b`` as a *sweep*, and ``A[B`` fails to
    parse outright. Strings are therefore always single-quoted, which round-trips every one of those
    (verified against ``hydra.core.override_parser``). Interpolations survive quoting — the override
    sets the literal text and OmegaConf resolves it on read — so ``${policy_base_url}`` still works.

    Only ``'`` is escaped. Hydra does **not** decode ``\\\\`` inside a quoted value: escaping
    backslashes doubles them, so they are passed through raw.

    ``None`` and booleans get their own spellings, since ``str()`` would emit ``"None"``/``"True"``
    and Hydra reads those back as text. Containers recurse for the same reason: ``str()`` on a dict
    emits Python's repr, whose quoted keys Hydra rejects outright.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return _hydra_dict(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_scalar(item) for item in value) + "]"
    if isinstance(value, str):
        # A trailing backslash would escape the closing quote and leave the value unterminated, and
        # there is no spelling that avoids it — better to say so than to emit something unparseable.
        if value.endswith("\\"):
            raise ValueError(
                f"Gym config override value {value!r} ends with a backslash, which Hydra's override "
                "grammar cannot express: it escapes the closing quote."
            )
        return "'" + value.replace("'", "\\'") + "'"
    return str(value)


def _flatten_overrides(overrides: Mapping[str, Any], _prefix: str = "") -> list[str]:
    """Flatten a nested override mapping into Hydra ``++dotted.path=value`` arguments.

    Callers describe overrides as structured data — ``{"a": {"b": 1}}`` — rather than as
    pre-serialized Hydra strings, so the config survives being sent somewhere as JSON. Hydra itself
    only speaks the flat form, so the translation happens here, at the point of invocation.

    ``++`` rather than ``+``: it sets a key whether or not it already exists, which is what an
    override means. A bare ``+`` fails on a key the merged config already defines.
    """
    arguments: list[str] = []
    for key, value in overrides.items():
        path = f"{_prefix}{key}"
        # An empty mapping has no leaves to descend to, so recursing would drop the override
        # entirely. It is still a value the caller asked to set: emit it as ``++path={}``, which
        # clears the subtree.
        if isinstance(value, Mapping) and value:
            arguments.extend(_flatten_overrides(value, f"{path}."))
        else:
            arguments.append(f"++{path}={hydra_scalar(value)}")
    return arguments


def selection_args(config: GymRuntimeConfig, work_dir: Path) -> list[str]:
    """The environment/agent/model selection passed to Gym.

    Built once and handed verbatim to both ``gym env validate`` and ``gym env start``, so what is
    validated is exactly what runs — a pre-flight against a different config would be worse than
    none.
    """
    selection = [
        "--config",
        config.agent_config,
        "--model-type",
        config.model_type,
        "--resources-server",
        config.resources_server,
    ]
    if config.bind_resources_server:
        # Composable (Pattern-A) agents leave resources_server.name unbound ('???'); bind it to the
        # env we're running. Assumes the agent config's top-level key equals the agent name (the
        # simple_agent convention) *and* that the resources-server is registered under the
        # environment's own name — not universally true, so self-contained or differently-named
        # servers set bind_resources_server=False and bind themselves via hydra_params.
        selection.append(
            f"+{config.agent}.responses_api_agents.{config.agent}.resources_server.name={config.resources_server}"
        )
    selection.extend(_flatten_overrides(config.hydra_params))
    # Gym is a Hydra app, so each invocation writes a timestamped run directory — by default
    # `outputs/<date>/<time>/` under the *current* directory. Since the subprocesses inherit this
    # process's cwd (so Gym can find env.yaml), the default would litter whatever directory the
    # caller happened to run from. Redirect it under the run's work dir, with the rest of the run's
    # artifacts. Applies to every Gym entry point, `gym list` included.
    #
    # Quoted like any other override value: this one is a *path*, and Hydra's grammar reads `,` as a
    # sweep separator and `[` as a list opener, so an unquoted work dir containing either is
    # silently misread or rejected outright. Verified against Hydra's own parser — unquoted,
    # `/tmp/a,b` becomes a ChoiceSweep and `/tmp/x[1]` raises OverrideParseException.
    selection.append(f"hydra.run.dir={hydra_scalar(str(work_dir / HYDRA_SUBDIR))}")
    # Off (Gym's default) a rollout reports one usage block summed over the turn and no timing at
    # all; on, each model exchange is captured with its own tokens and latency. Asserted over
    # `hydra_params` rather than defaulted, because turning it off degrades every trace silently.
    # ``++`` for the same reason `_flatten_overrides` uses it: these keys are absent from the merged
    # config until something sets them, and a bare `+` fails once they are present.
    selection.append(f"++observability_enabled={hydra_scalar(True)}")
    selection.append(f"++model_call_capture_dir={hydra_scalar(str(model_call_capture_dir(work_dir)))}")
    return selection


def model_call_capture_dir(work_dir: Path) -> Path:
    """Where Gym writes one ``<rollout_id>.capture.jsonl`` per rollout. Gym requires an absolute path."""
    return (work_dir / _CAPTURE_SUBDIR).resolve()


class GymRuntimeConfig(BaseModel):
    """Declarative config for running an existing Gym environment via the ``gym`` CLI.

    Holds only plain fields; the two-step invocation is built from them at run
    time. The dataset itself is recovered from the tasks (stamped by
    :func:`discover_gym_tasks`), mirroring the Harbor runner.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(description="Agent name to collect rollouts with, e.g. 'simple_agent'.")
    agent_config: str = Field(description="Repo-relative agent config passed to `gym env start` (--config).")
    resources_server: str = Field(description="Resources-server (environment) name, e.g. 'mcqa' (--resources-server).")
    model_type: str = Field(
        default="inference_provider",
        description="Model-type config (--model-type). `inference_provider` speaks OpenAI-compatible chat; "
        "`openai_model` uses the OpenAI Responses API (500s against chat-only endpoints).",
    )
    bind_resources_server: bool = Field(
        default=True,
        description="Auto-bind the agent's `resources_server.name` to `resources_server` via a Hydra override "
        "(the composable/Pattern-A agent case, e.g. simple_agent whose config leaves it '???'). Set False for "
        "self-contained agents that already bind their own resources-server.",
    )
    hydra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters merged into Gym's Hydra config, applied after the auto-derived "
        "resources-server binding. Structured rather than pre-serialized Hydra strings so the config "
        "travels as JSON — `{'a': {'b': 1}}` becomes `++a.b=1` at invocation. This is the escape "
        "hatch for what Gym does not standardize: an environment whose resources-server is registered "
        "under a different name, or which references a model server no shipped config defines. "
        "Distinct from `env_vars`: these configure the Gym *environment*, not the OS environment.",
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables set on the `gym` invocation, merged over the ones this "
        "process already has. Some Gym environments are configurable only this way — `wmt_translation` "
        "reads `WMT_TRANSLATION_COMET_PY_CACHE` for its model-cache root, defaulting to a path that "
        "exists only inside NVIDIA's container image — and requiring the caller to export those turns "
        "a property of the environment into a property of whoever launched the run. Redacted from "
        "recorded provenance on the same rules as `hydra_params`.",
    )
    num_repeats: int = Field(default=1, ge=1, description="Attempts per row; each attempt becomes one trial.")
    concurrency: int = Field(
        default=4,
        ge=1,
        description="Concurrent rollouts for `gym eval run` (the collection-phase knob, tuned to the model "
        "endpoint's limits). Distinct from AgentEvalRunConfig.parallelism, which bounds concurrent scoring.",
    )
    startup_timeout_s: float = Field(default=240.0, gt=0, description="Max wait for `gym env start` readiness.")
    collection_timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Max wait for `gym eval run` collection; None = unbounded (scales with dataset x num_repeats x "
        "model latency, so no safe fixed default). Set it to bound a hung/slow model endpoint.",
    )
    shutdown_grace_s: float = Field(
        default=30.0,
        gt=0,
        description="Grace period for a Gym subprocess *group* to exit on SIGTERM (letting Ray shut down cleanly) "
        "before escalating to SIGKILL.",
    )
    reward_key: str = Field(default=DEFAULT_REWARD_KEY, description="Key read from each rollout record.")
