# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General utilities for the nemo-agents plugin."""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import yaml
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.entities import parse_qualified_name
from nemo_platform_plugin.virtual_models.client import VirtualModelsClient

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")

# LLM ``_type`` values that resolve through the platform Inference Gateway —
# i.e. their ``model_name`` should correspond to a VirtualModel registered in
# the workspace.  Other types (e.g. direct cloud SDKs) bypass IGW and are not
# validatable through the platform VirtualModels API.
_IGW_LLM_TYPES = frozenset({"openai", "nim"})

# Fabric model providers that speak through Platform's OpenAI-compatible IGW
# endpoint. Native providers such as Anthropic should keep their own adapter
# defaults unless the config explicitly supplies a base_url.
_FABRIC_IGW_MODEL_PROVIDERS = frozenset({"openai", "nvidia", "openai-compatible"})

# Regex used to skip LLM ``model_name`` values that still contain unexpanded
# ``$VAR`` / ``${VAR}`` placeholders.  These were left in place by
# :func:`expand_env_vars` because the corresponding env var was unset; the
# validator can't meaningfully look them up, and ``expand_env_vars`` already
# logged a warning, so we skip silently rather than raising.
_UNEXPANDED_ENV_VAR_RE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")

# Intake reads are workspace-scoped, so a config's literal workspace must be
# rebound to the deployment target or its traces are unreachable.
_INTAKE_WORKSPACE_RE = re.compile(r"(/apis/intake/v\d+/workspaces/)[^/]+(/)")


def expand_env_vars(value: Any, vars_dict: dict[str, Any]) -> Any:
    """Recursively expand ``$VAR`` / ``${VAR}`` references in *value* using vars.

    Strings are scanned for ``$VAR`` and ``${VAR}`` tokens; each token is
    replaced by the value of the corresponding environment variable.  When a
    variable is not set, the token is left untouched and a warning is logged
    so the user can tell why, e.g., a missing ``api_key`` still looks literal.

    ``dict`` and ``list`` containers are traversed; other types are returned
    as-is.  ``$$`` escapes a literal dollar sign.
    """
    if isinstance(value, str):
        # Honor the POSIX-style "$$" escape for a literal "$" before expansion.
        placeholder = "\0DOLLAR\0"
        protected = value.replace("$$", placeholder)

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            if name in vars_dict:
                return vars_dict[name]
            logger.warning("Env var %r referenced in config is not set; leaving as-is", name)
            return match.group(0)

        expanded = _ENV_VAR_PATTERN.sub(_replace, protected)
        return expanded.replace(placeholder, "$")
    if isinstance(value, dict):
        return {k: expand_env_vars(v, vars_dict) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v, vars_dict) for v in value]
    return value


def _normalize_output_dir(user_dir: str) -> Path:
    """Return a clean relative :class:`~pathlib.Path` from a user-supplied string.

    Leading/trailing whitespace and slashes are stripped; duplicate slashes are
    collapsed by splitting on ``/`` and filtering empty parts. Parent directory
    references (``..``) are rejected to prevent directory traversal attacks.

    This function expects relative paths. While it will strip leading slashes
    from absolute-looking paths (``/foo/bar`` → ``foo/bar``), the result is
    always interpreted as relative to the base directory supplied by the caller.

    Examples::

        _normalize_output_dir("eval/my-agent")    # Path("eval/my-agent")
        _normalize_output_dir("/eval/my-agent/")  # Path("eval/my-agent") - leading slash stripped
        _normalize_output_dir("//eval//my-agent") # Path("eval/my-agent")

    Args:
        user_dir: A user-supplied output directory string, expected to be relative.

    Returns:
        A normalized relative Path with no empty components or parent references.

    Raises:
        ValueError: If the path contains ``..`` directory traversal components.
    """
    parts = [p for p in user_dir.strip().split("/") if p]
    if ".." in parts:
        raise ValueError(f"Output path '{user_dir}' contains directory traversal components (..), which is not allowed")
    return Path(*parts) if parts else Path("output")


def output_dir_override(eval_config: Path, output_base: Path) -> list[str]:
    """Return ``--override`` args that repoint ``output_dir`` to an absolute path.

    Reads ``eval.general.output_dir`` from *eval_config*, normalises it, and
    returns a ``["--override", "eval.general.output_dir", "<abs_path>"]`` list
    ready to append to a ``nat eval`` command.  Returns an empty list if
    ``output_dir`` is absent from the config.

    No files are written — the override is passed directly on the command line.

    Args:
        eval_config: Path to the user-authored NAT evaluation YAML.
        output_base: Absolute base directory supplied by the runner backend.
    """
    config = yaml.safe_load(eval_config.read_text(encoding="utf-8"))

    try:
        user_dir: str = config["eval"]["general"]["output_dir"]
    except KeyError:
        return []

    normalised = _normalize_output_dir(user_dir)
    return ["--override", "eval.general.output_dir", str(output_base / normalised)]


def rebase_optimize_outputs(config: dict[str, Any], output_base: Path) -> dict[str, Any]:
    """Return a copy of *config* with optimize output paths rebased to *output_base*.

    Rewrites ``eval.general.output_dir`` and ``optimizer.output_path`` when
    present, preserving their relative suffixes while moving the actual write
    location under the supplied absolute base directory.

    Non-string values (None, int, etc.) in output path fields are silently
    skipped to avoid breaking configs with unusual structures.

    Raises:
        ValueError: If any output path contains directory traversal attempts or
            would escape the base directory.
    """
    rebased = copy.deepcopy(config)

    with contextlib.suppress(KeyError):
        eval_output_dir = rebased["eval"]["general"]["output_dir"]
        if isinstance(eval_output_dir, str):
            normalized = _normalize_output_dir(eval_output_dir)
            resolved = (output_base / normalized).resolve()
            if not resolved.is_relative_to(output_base.resolve()):
                raise ValueError(f"Eval output path '{eval_output_dir}' would escape base directory")
            rebased["eval"]["general"]["output_dir"] = str(resolved)

    with contextlib.suppress(KeyError):
        optimizer_output_path = rebased["optimizer"]["output_path"]
        if isinstance(optimizer_output_path, str):
            normalized = _normalize_output_dir(optimizer_output_path)
            resolved = (output_base / normalized).resolve()
            if not resolved.is_relative_to(output_base.resolve()):
                raise ValueError(f"Optimizer output path '{optimizer_output_path}' would escape base directory")
            rebased["optimizer"]["output_path"] = str(resolved)

    return rebased


def get_base_url() -> str:
    """Return the base URL for the platform from the environment."""
    from nemo_platform_plugin.config import get_platform_config

    # `or` chain so get_platform_config() stays lazy.
    return (os.environ.get("NEMO_BASE_URL") or os.environ.get("NMP_BASE_URL") or get_platform_config().base_url).rstrip(
        "/"
    )


def get_internal_base_url() -> str | None:
    """Return the in-cluster platform base URL reachable from inside agent pods, or None.

    This is the API Service DNS used to reach the platform from a deployed agent
    when :func:`get_base_url` is not routable from inside the container. Read from
    ``NEMO_INTERNAL_BASE_URL``, then ``NMP_INTERNAL_BASE_URL``.
    """
    internal = os.environ.get("NEMO_INTERNAL_BASE_URL") or os.environ.get("NMP_INTERNAL_BASE_URL")
    return internal.rstrip("/") if internal else None


def get_default_model() -> str | None:
    """Return the default model for the platform from the SDK context."""
    from nemo_platform_ext.config import get_context

    return get_context().default_model


def inject_gateway_url(
    config: dict[str, Any],
    workspace: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Deep-copy *config* and inject the Inference Gateway URL into OpenAI/NIM LLMs.

    For each LLM in ``config["llms"]`` whose ``_type`` is ``"openai"`` or
    ``"nim"``, sets ``base_url`` and ``api_key`` using ``setdefault`` so
    existing explicit values are preserved.

    The base URL is resolved in priority order:

    1. *base_url* argument (when supplied explicitly, e.g. from a CLI flag).
    2. ``NEMO_BASE_URL`` environment variable (only consulted when *base_url*
       is ``None``).
    3. Platform configuration (``PlatformConfig.base_url``), which defaults to
       ``http://localhost:8080`` and can be overridden via ``NMP_BASE_URL``
       (only consulted when *base_url* is ``None`` and ``NEMO_BASE_URL`` is
       unset).

    Args:
        config: NAT workflow or eval config dict (not mutated).
        workspace: Workspace name used to construct the IGW path.
        base_url: Optional explicit base URL.  When ``None``, the platform
            configuration is consulted.

    Returns:
        A deep copy of *config* with IGW URLs injected.
    """
    if base_url is None:
        base_url = get_base_url()
    base_url = base_url.rstrip("/")

    gateway_url = f"{base_url}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1"

    config = copy.deepcopy(config)
    for llm_cfg in config.get("llms", {}).values():
        if isinstance(llm_cfg, dict) and llm_cfg.get("_type") in ("openai", "nim"):
            llm_cfg.setdefault("base_url", gateway_url)
            llm_cfg.setdefault("api_key", "not-used")

    return config


def inject_fabric_gateway_url(
    config: dict[str, Any],
    workspace: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Deep-copy a Platform agent config and bind its models to the Inference Gateway."""
    if base_url is None:
        base_url = get_base_url()
    gateway_url = f"{base_url.rstrip('/')}/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1"

    resolved = copy.deepcopy(config)
    model_configs: list[Any] = []

    models = resolved.get("models")
    if isinstance(models, dict):
        model_configs.extend(models.values())

    harnesses = resolved.get("harnesses")
    if isinstance(harnesses, dict):
        for harness in harnesses.values():
            if isinstance(harness, dict) and "model" in harness:
                model_configs.append(harness["model"])

    for model_config in model_configs:
        if not isinstance(model_config, dict):
            continue
        if model_config.get("base_url") is not None:
            continue
        settings = model_config.get("settings")
        if isinstance(settings, dict) and isinstance(settings.get("base_url"), str):
            model_config["base_url"] = settings.pop("base_url")
            continue
        provider = model_config.get("provider")
        if not isinstance(provider, str) or provider.lower() not in _FABRIC_IGW_MODEL_PROVIDERS:
            continue
        model_config["base_url"] = gateway_url

    rebind_intake_ingest_workspace(resolved, workspace)

    return resolved


def rebind_intake_ingest_workspace(config: dict[str, Any], workspace: str) -> dict[str, Any]:
    """Point ATIF HTTP storage endpoints at *workspace* instead of the config's literal.

    Mutates *config* in place and returns it. Endpoints that are not Intake ingest
    URLs are left alone.
    """
    telemetry = config.get("telemetry")
    atif = telemetry.get("atif") if isinstance(telemetry, dict) else None
    storage_configs = atif.get("storage", []) if isinstance(atif, dict) else []
    if not isinstance(storage_configs, list):
        return config
    for storage_config in storage_configs:
        if not isinstance(storage_config, dict) or storage_config.get("type") != "http":
            continue
        endpoint = storage_config.get("endpoint")
        if not isinstance(endpoint, str):
            continue
        # Rewrite the path only; a query or fragment can carry the same text.
        parts = urlsplit(endpoint)
        path = _INTAKE_WORKSPACE_RE.sub(
            lambda match: f"{match.group(1)}{workspace}{match.group(2)}",
            parts.path,
            count=1,
        )
        if path != parts.path:
            storage_config["endpoint"] = parts._replace(path=path).geturl()
    return config


def inject_fabric_trace_agent_name(
    config: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    """Stamp the Platform-registered *agent_name* onto a Fabric agent spec's telemetry."""
    telemetry = config.get("telemetry")
    if not agent_name or not isinstance(telemetry, dict):
        return config
    telemetry.setdefault("agent_name", agent_name)
    return config


def inject_nemo_trace_fields(
    config: dict[str, Any],
    workspace: str,
    agent_name: str,
) -> dict[str, Any]:
    """Inject *workspace* and *agent_name* into any ``nemo_files`` telemetry exporters.

    Mutates *config* in place (call after :func:`inject_gateway_url` which
    already deep-copies) and returns it for chaining convenience.
    """
    tracers = config.get("general", {}).get("telemetry", {}).get("tracing", {})
    for tracer_cfg in tracers.values():
        if isinstance(tracer_cfg, dict) and tracer_cfg.get("_type") == "nemo_files":
            tracer_cfg.setdefault("workspace", workspace)
            tracer_cfg.setdefault("agent_name", agent_name)
    return config


def inject_default_model(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Replace ``${NEMO_DEFAULT_MODEL}`` placeholders in *config* with the resolved model."""
    default_model = get_default_model()
    if default_model is None:
        return config
    config = expand_env_vars(config, vars_dict={"NEMO_DEFAULT_MODEL": default_model})
    return config


def validate_llm_models(
    config: dict[str, Any],
    *,
    workspace: str,
    sdk: NeMoPlatform,
) -> None:
    """Pre-flight check that every IGW-routed LLM in *config* exists as a VirtualModel.

    Iterates ``config["llms"]`` and, for each block whose ``_type`` is in
    :data:`_IGW_LLM_TYPES`, strips a leading ``{workspace}/`` qualifier from
    ``model_name`` (mirroring IGW's OpenAI proxy — the VM route takes
    workspace as its own path segment, so the bare name is what it expects)
    then calls the typed VirtualModels client.
    Names are deduplicated *after* stripping so the same model declared under
    multiple LLM keys (e.g. agent + judge) costs one network call.

    LLM blocks whose ``model_name`` still contains an unexpanded ``$VAR`` /
    ``${VAR}`` placeholder are skipped — :func:`expand_env_vars` already
    warned about the missing env var, and the literal placeholder string
    isn't a routable model name to look up.

    Raises:
        ValueError: If any unique ``model_name`` is absent from *workspace*.
            The message names every missing model and tells the user how to
            recover (set ``NEMO_DEFAULT_MODEL``, register the model, or edit
            the YAML).

    Soft-fails on any other SDK exception (network, auth, 5xx, etc.) by
    logging a warning and returning normally — the validator is meant to
    *help* users catch typos and missing models early, not gate runs when
    the platform is transiently unreachable.  The actual eval/optimize call
    will surface those errors itself.

    Args:
        config: A NAT workflow / eval / optimize config dict, post env-var
            expansion.  Not mutated.
        workspace: Workspace name passed to the VirtualModels SDK call.
        sdk: Sync platform SDK handle.
    """
    llms = config.get("llms")
    if not isinstance(llms, dict):
        return

    # Dedup by resolved (workspace, name) so the same model under multiple LLM
    # keys costs one retrieve; keep the first llm_key for error locations.
    to_check: dict[tuple[str, str], str] = {}
    for llm_key, llm_cfg in llms.items():
        if not isinstance(llm_cfg, dict):
            continue
        if llm_cfg.get("_type") not in _IGW_LLM_TYPES:
            continue
        model_name = llm_cfg.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            continue
        if _UNEXPANDED_ENV_VAR_RE.search(model_name):
            logger.debug(
                "Skipping LLM %r model validation: model_name %r still contains an unexpanded "
                "env-var placeholder (set the corresponding env var or edit the YAML).",
                llm_key,
                model_name,
            )
            continue
        # A model_name may be workspace-qualified ("prod/foo"); the VM route
        # takes workspace as its own path segment, so resolve it here.
        target_ws, target_name = parse_qualified_name(model_name, default_workspace=workspace)
        if not target_name:
            continue
        to_check.setdefault((target_ws, target_name), llm_key)

    if not to_check:
        return

    virtual_models_client = client_from_platform(sdk, VirtualModelsClient)
    missing: list[tuple[str, str]] = []  # (qualified_name, llm_key)
    for (target_ws, target_name), llm_key in to_check.items():
        try:
            virtual_models_client.get_virtual_model(name=target_name, workspace=target_ws)
        except ClientNotFoundError:
            missing.append((f"{target_ws}/{target_name}", llm_key))
        except Exception as exc:  # pragma: no cover - defensive soft-fail
            # Soft-fail: a transient platform outage or auth blip shouldn't gate
            # the run; the subprocess surfaces the real error. exc_info keeps the
            # traceback.
            logger.warning(
                "Could not validate LLM %r (model=%r) against workspace %r: %s. "
                "Continuing anyway; the underlying eval/optimize call will surface any "
                "real error.",
                llm_key,
                target_name,
                target_ws,
                exc,
                exc_info=exc,
            )

    if missing:
        details = ", ".join(f"{name!r} (llms.{key}.model_name)" for name, key in missing)
        raise ValueError(
            f"The following LLM model(s) are not registered as VirtualModels in workspace "
            f"{workspace!r}: {details}. "
            f"Fix by (a) setting NEMO_DEFAULT_MODEL and using ${{NEMO_DEFAULT_MODEL}} in the "
            f"config, (b) registering the model with `nemo inference virtual-models create`, "
            f"or (c) editing the YAML to reference an existing VirtualModel "
            f"(`nemo inference virtual-models list --workspace {workspace}`)."
        )


def preflight_validate_llm_models(
    config_path: Path,
    *,
    workspace: str,
    sdk: NeMoPlatform | None,
    agent_config: dict[str, Any] | None = None,
) -> None:
    """Load *config_path*, expand env vars, optionally merge an agent config, and validate.

    Thin wrapper around :func:`validate_llm_models` that mirrors the first
    two passes :func:`temp_injected_config` does (YAML load + env-var
    expansion, plus :func:`merge_agent_config` when *agent_config* is
    given) so the dict we validate matches what ``nat eval`` /
    ``nat optimize`` will eventually see — without subprocess-spawning
    side effects.

    Used as a pre-flight in ``EvaluateAgentJob.run`` and
    ``OptimizeJob.run`` (via ``nemo-optimization``) to surface a missing-VirtualModel error
    before the subprocess starts.

    No-op when *sdk* is ``None`` (local-only paths that don't have a
    platform handle have nothing to look up against — letting the run
    proceed is the right behaviour, and matches how the call sites
    treated the original inline pre-flight).

    Args:
        config_path: Path to the eval or optimize NAT YAML config.
        workspace: Workspace passed to :func:`validate_llm_models`.
        sdk: Sync platform SDK handle.  ``None`` is a no-op.
        agent_config: Optional agent NAT config dict to merge under the
            YAML's contents before validation.  Used by the optimize job
            so an agent-fetched LLM gets validated alongside the
            optimize-side judge.

    Raises:
        ValueError: If any IGW-routed LLM in the merged, expanded config
            isn't a registered VirtualModel in *workspace*.  See
            :func:`validate_llm_models` for the message shape.
    """
    if sdk is None:
        return
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expanded_config = expand_env_vars(raw_config, vars_dict=dict(os.environ))
    if agent_config is not None:
        expanded_config = merge_agent_config(agent_config, expanded_config)
    validate_llm_models(expanded_config, workspace=workspace, sdk=sdk)


# NAT config sections that are dict-of-typed-components (e.g. ``llms.llm:
# {_type: openai, ...}``).  When merging an agent config with an optimize
# config we want a *shallow* per-key merge for these — same component name on
# both sides means the optimize-side definition replaces the agent's, so
# tuning metadata (``optimizable_params`` / ``search_space``) lands on the
# correct object.  Doing a deep merge here would interleave fields from two
# unrelated component definitions and produce nonsense (e.g. an LLM with
# ``_type`` from one side and ``model_name`` from the other).
_COMPONENT_DICT_SECTIONS = frozenset(
    {
        "functions",
        "function_groups",
        "llms",
        "embedders",
        "memory",
        "object_stores",
        "retrievers",
        "ttc_strategies",
        "authentication",
        "trainers",
        "trainer_adapters",
        "trajectory_builders",
    }
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge with *override* winning at every leaf.

    Returns a new dict; neither input is mutated.  Non-dict values from
    *override* replace whatever's at the same key in *base*.  Lists are
    treated as opaque values and replaced wholesale — list-merge semantics
    are intentionally not invented here because NAT configs use lists as
    ordered references (``tool_names``, ``optimizable_params``) where
    concatenation would be semantically wrong.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_agent_config(
    agent_config: dict[str, Any],
    optimize_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge an agent's stored NAT config with an optimize-config dict.

    Used when composing platform agent definitions with optimize-config YAML
    before dispatch (e.g. ``nemo customization optimize --agent ...``).
    The agent supplies the workflow shape (``workflow``, ``functions``, telemetry, the LLMs
    actually invoked at runtime); the optimize config supplies eval, the
    optimizer block, judge LLMs, and any tuning overrides on shared LLM
    keys.

    Merge rules:

    * **Component dict sections** (``llms``, ``functions``, ``embedders``,
      ``memory``, ``retrievers``, etc., enumerated in
      ``_COMPONENT_DICT_SECTIONS``): per-key shallow merge.  If the same
      component name appears on both sides (e.g. both define ``llms.llm``)
      the optimize-side definition replaces the agent's wholesale — that's
      where ``optimizable_params`` and ``search_space`` need to end up
      attached, alongside the rest of the LLM spec, for the optimizer to
      pick them up.  Component names that appear on only one side are
      passed through unchanged.
    * **``workflow``**: agent always wins.  Optimize configs are not
      expected to declare a workflow when an agent is supplied — the
      agent's stored workflow is the whole point of the merge.  If the
      optimize side does declare one, the agent's still wins (and a
      warning is emitted) so the resulting trial config matches what the
      deployed agent actually runs.
    * **All other keys** (``general``, ``eval``, ``optimizer``, etc.):
      deep merge with optimize winning at the leaves.  This preserves
      ``general.telemetry`` from the agent while letting the optimize side
      add or override sibling keys.

    Args:
        agent_config: The YAML-equivalent dict stored on the platform's
            ``Agent`` entity (i.e. the contents of ``react-agent.yml``).
        optimize_config: The user-authored optimize config dict (i.e. the
            contents of ``react-optimize.yml``).

    Returns:
        A new merged dict suitable for passing to ``nat optimize
        --config_file`` to run the agent's workflow locally with the
        tuning knobs and eval/optimizer settings from the optimize side.
    """
    merged: dict[str, Any] = copy.deepcopy(agent_config)

    if "workflow" in optimize_config and optimize_config["workflow"] != agent_config.get("workflow"):
        logger.warning(
            "Optimize config declares a 'workflow' that differs from the agent's stored "
            "workflow; ignoring the optimize-side workflow so trials run the same shape "
            "as the deployed agent."
        )

    for key, opt_value in optimize_config.items():
        if key == "workflow":
            continue
        if key in _COMPONENT_DICT_SECTIONS and isinstance(opt_value, dict):
            base_section = merged.get(key) or {}
            if not isinstance(base_section, dict):
                merged[key] = copy.deepcopy(opt_value)
                continue
            section = copy.deepcopy(base_section)
            for comp_name, comp_value in opt_value.items():
                section[comp_name] = copy.deepcopy(comp_value)
            merged[key] = section
            continue

        if isinstance(opt_value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], opt_value)
        else:
            merged[key] = copy.deepcopy(opt_value)

    return merged


@contextlib.contextmanager
def temp_injected_config(
    config_path: Path,
    workspace: str,
    base_url: str | None = None,
    defaults: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
    output_base: Path | None = None,
) -> Iterator[Path]:
    """Context manager that writes a URL-injected copy of a NAT config to a temp file.

    Loads the YAML at *config_path*, expands any ``$VAR`` / ``${VAR}``
    references in string values via :func:`expand_env_vars`, applies
    :func:`inject_gateway_url`, and writes the result to a temporary file in
    the **same directory** as the source.  Writing to the same directory
    ensures that relative paths inside the config (e.g.
    ``dataset.file_path: my-data.json``) remain valid when the subprocess
    sets ``cwd`` to that directory.

    Yields:
        Absolute :class:`~pathlib.Path` to the temp file.  The file is deleted
        when the context exits, whether or not an exception was raised.

    Args:
        config_path: Path to the original NAT YAML config file.
        workspace: Workspace name forwarded to :func:`inject_gateway_url`.
        base_url: Optional explicit base URL forwarded to
            :func:`inject_gateway_url`.
        defaults: Optional dict of key-value pairs to ``setdefault`` into the
            config after gateway URL injection.
        extra_config: Optional NAT config dict to merge *under* the YAML's
            contents before gateway URL injection — typically used to fold a
            stored agent definition into an optimize config.  The on-disk YAML
            wins on shared keys via :func:`merge_agent_config` semantics, so
            user-authored overrides remain authoritative.
        output_base: Optional base directory for rebasing output paths in
            optimize configs. When provided, :func:`rebase_optimize_outputs`
            is applied to rewrite ``eval.general.output_dir`` and
            ``optimizer.output_path`` to write under *output_base*.

    Raises:
        yaml.YAMLError: If the config file contains invalid YAML syntax.
        ValueError: If output path rebasing fails due to directory traversal
            attempts or other path validation errors.
        RuntimeError: If config processing fails for other reasons (wraps
            underlying exceptions with context).
    """
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Failed to parse config file {config_path}: {exc}") from exc
    except (OSError, IOError) as exc:
        raise RuntimeError(f"Failed to read config file {config_path}: {exc}") from exc

    try:
        config = expand_env_vars(config, vars_dict=dict(os.environ))

        if extra_config is not None:
            config = merge_agent_config(extra_config, config)

        injected = inject_gateway_url(config, workspace, base_url=base_url)

        if defaults:
            for key, value in defaults.items():
                injected.setdefault(key, value)

        if output_base is not None:
            injected = rebase_optimize_outputs(injected, output_base)
    except ValueError:
        # Re-raise ValueError from rebase_optimize_outputs as-is
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to process config {config_path}: {exc}") from exc

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f".injected-{config_path.stem}-",
        dir=config_path.parent,
        delete=False,
    )
    try:
        try:
            yaml.safe_dump(injected, tmp)
            tmp.flush()
            tmp.close()
        except yaml.YAMLError as exc:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(f"Failed to serialize processed config: {exc}") from exc
        except Exception as exc:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(f"Failed to write temporary config file: {exc}") from exc

        # Yield after successful write - exceptions from the caller should propagate
        yield Path(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
