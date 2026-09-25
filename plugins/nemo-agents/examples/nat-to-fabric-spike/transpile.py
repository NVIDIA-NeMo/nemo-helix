#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NAT -> Fabric transpile spike (Stage 1).

Reads a NAT (NVIDIA NeMo Agent Toolkit) workflow config and emits a Fabric-native
`agent.yaml` targeting the LangChain Deep Agents harness, plus a migration report.

Scope note: this Stage 1 spike reads the NAT YAML directly so it runs with only
PyYAML. A production version would resolve the config through NAT's
`WorkflowBuilder.from_config()` to also recover default prompts (which live in
NAT's Python, not the YAML) and each tool's resolved input/output schema. Points
where that matters are flagged in the report rather than guessed.

What it demonstrates:
  * A non-trivial NAT topology (reasoning wrapper -> orchestrator -> sub-agents)
    maps onto Deep Agents subagents. Every agent in the tree keeps its identity;
    nested agents are emitted as their own subagents with a `delegates_to` list,
    not flattened into their parent.
  * NAT MCP servers (function_groups of _type mcp_client) carry across to Fabric's
    `mcp.servers` one-to-one, including stdio and streamable-http.
  * Credentials are never guessed. Env-var URLs carry across; NAT OAuth2 / custom
    header auth is flagged as a Fabric adapter gap. Dangling refs, non-MCP groups,
    and missing models are reported as errors, not silently mishandled.
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys
from collections import deque
from pathlib import Path

import yaml

# NAT agent archetypes and the field each uses to reference its children.
AGENT_CHILD_FIELD = {
    "react_agent": "tool_names",
    "tool_calling_agent": "tool_names",
    "rewoo_agent": "tool_names",
    "router_agent": "branches",
    "sequential_executor": "tool_list",
    "reasoning_agent": "augmented_fn",  # single ref, not a list
}
AGENT_TYPES = set(AGENT_CHILD_FIELD)
MCP_TYPES = {"mcp_client", "per_user_mcp_client"}
ENV_REF = re.compile(r"\$\{(\w+)(?::[-=]?[^}]*)?\}")
RESOLVE = "[RESOLVE]"

# NAT top-level sections this transpiler does not carry; each needs a home elsewhere.
UNMAPPED_SECTIONS = ["middleware", "memory", "retrievers", "embedders", "object_stores", "ttc_strategies", "optimizer"]
FEATURE_HINTS = {
    "middleware": "NAT middleware (e.g. NASSE) maps to NeMo Relay",
    "memory": "needs a Fabric or Platform memory equivalent",
    "retrievers": "needs a retrieval equivalent (Platform or an MCP server)",
    "embedders": "map to Fabric models or a retrieval service",
    "object_stores": "needs a Platform storage equivalent",
    "ttc_strategies": "test-time-compute strategies need a Fabric or Relay equivalent",
    "optimizer": "config optimization is a separate Platform or Fabric concern",
}


class Report:
    """Collects human-facing findings emitted alongside the config."""

    def __init__(self) -> None:
        self.carried: list[str] = []
        self.env_vars: set[str] = set()
        self.auth_gaps: list[str] = []
        self.builtins: set[str] = set()
        self.errors: list[str] = []
        self.features: list[str] = []
        self.notes: list[str] = []
        self.main_tools: list[str] = []
        self.agent_types: dict[str, str] = {}  # agent name -> NAT _type, for analysis


def child_refs(entry: dict) -> list[str]:
    """Return the names an agent entry points at, normalized to a list.

    Returns [] for a custom/unknown _type (we don't know its child field). Callers
    that need to flag custom types check AGENT_TYPES separately.
    """
    field = AGENT_CHILD_FIELD.get(entry.get("_type") or "")
    if field is None:
        return []
    value = entry.get(field)
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def scan_env(text: object, report: Report) -> list[str]:
    """Record every ${VAR} reference in a string; return the names found."""
    found = ENV_REF.findall(str(text))
    report.env_vars.update(found)
    return found


def classify(ref: str, functions: dict, groups: dict) -> str:
    """Bucket a referenced name: agent | mcp | builtin | group | dangling."""
    if ref in groups:
        return "mcp" if groups[ref].get("_type") in MCP_TYPES else "group"
    if ref in functions:
        return "agent" if functions[ref].get("_type") in AGENT_TYPES else "builtin"
    return "dangling"


def llm_to_model(llm: dict, report: Report, alias: str) -> dict:
    """Map a NAT LLM entry to a Fabric ModelConfig. Only `nim` maps to NVIDIA."""
    ltype = llm.get("_type")
    model_name = llm.get("model_name")
    if not model_name:
        model_name = RESOLVE
        report.errors.append(f"LLM '{alias}' has no model_name; emitted {RESOLVE}.")

    if ltype == "nim":
        provider, api_key_env = "nvidia", "NVIDIA_API_KEY"
    else:
        provider, api_key_env = (ltype or RESOLVE), None
        report.notes.append(
            f"LLM '{alias}' is _type '{ltype}', not nim. Set provider '{provider}' "
            f"and its api key env var manually."
        )

    reserved = {"_type", "model_name", "temperature"}
    settings = {k: v for k, v in llm.items() if k not in reserved}
    model: dict = {"provider": provider, "model": model_name, "temperature": llm.get("temperature", 0.0)}
    if api_key_env:
        model["api_key_env"] = api_key_env
    if settings:
        model["settings"] = settings
    return model


def mcp_server_to_fabric(name: str, group: dict, auth_block: dict, report: Report) -> dict:
    """Map one NAT mcp_client function group to a Fabric McpServerConfig and record findings."""
    server = group.get("server", {})
    transport = str(server.get("transport", "streamable-http"))

    if transport == "stdio":
        command = server.get("command")
        args = server.get("args", []) or []
        if command:
            url = shlex.join([str(command), *[str(a) for a in args]])
            report.carried.append(f"{name}: stdio, no credentials")
        elif server.get("url"):
            url = str(server["url"])
            scan_env(url, report)
            report.carried.append(f"{name}: stdio (command supplied via url)")
        else:
            url = RESOLVE
            report.errors.append(f"{name}: stdio server has no command; emitted {RESOLVE}.")
    else:
        url = str(server.get("url", ""))
        if not url:
            report.errors.append(f"{name}: {transport} server has no url.")
        found = scan_env(url, report)
        report.carried.append(f"{name}: {transport}, {'url via env var' if found else 'static url'}")

    fabric = {"transport": transport, "url": url, "exposure": "harness_native"}

    # Auth the Deep Agents adapter cannot carry today (it forwards only transport +
    # url; a ${ENV} in the url is the one credential path that reaches the server).
    provider_ref = server.get("auth_provider")
    if provider_ref:
        provider = auth_block.get(provider_ref, {})
        report.auth_gaps.append(
            f"{name}: NAT used auth_provider '{provider_ref}' (_type "
            f"{provider.get('_type', 'unknown')}). Deep Agents carries only ${{ENV}} "
            f"URLs, so this needs a token-in-URL gateway or Fabric adapter OAuth2 support."
        )
    if server.get("custom_headers"):
        report.auth_gaps.append(
            f"{name}: NAT used custom_headers, which the Deep Agents adapter ignores. "
            f"Move the credential into the url via ${{ENV}} or extend the adapter."
        )
    return fabric


def transpile(config: dict, report: Report, name_override: str | None = None, default_name: str | None = None) -> dict:
    llms = config.get("llms", {})
    functions = config.get("functions", {})
    groups = config.get("function_groups", {})
    auth_block = config.get("authentication", {})
    workflow = config["workflow"]

    # NAT sections this transpiler does not carry; each needs a home in Fabric/Platform/Relay.
    for section in UNMAPPED_SECTIONS:
        if config.get(section):
            report.features.append(f"{section}: {FEATURE_HINTS.get(section, 'needs a Fabric or Platform equivalent')}")

    # MCP auth providers carry env-var URLs and redirect URIs the user must set.
    for provider in auth_block.values():
        if isinstance(provider, dict):
            for value in provider.values():
                if isinstance(value, str):
                    scan_env(value, report)

    # Unwrap reasoning_agent wrappers down to the executing agent (loop-safe).
    entry = workflow
    main_key: str | None = None
    seen_wrap: set[str] = set()
    while entry.get("_type") == "reasoning_agent":
        refs = child_refs(entry)
        if not refs:
            break
        wrapped = refs[0]
        if wrapped not in functions:
            report.errors.append(f"reasoning_agent augmented_fn '{wrapped}' is not defined in functions.")
            break
        if wrapped in seen_wrap:
            report.errors.append(f"reasoning_agent chain loops at '{wrapped}'; stopped unwrapping.")
            break
        seen_wrap.add(wrapped)
        main_key = wrapped
        report.notes.append(f"Unwrapped reasoning_agent onto '{wrapped}' as the main Deep Agent.")
        entry = functions[wrapped]

    if entry.get("_type") not in AGENT_TYPES:
        report.errors.append(
            f"Top-level agent type '{entry.get('_type')}' is a custom NAT type, not a stock "
            f"archetype. The spike maps stock agents; resolve custom registered types via NAT "
            f"WorkflowBuilder before transpiling."
        )

    used_groups: list[str] = []

    def leaf_and_children(agent_entry: dict) -> tuple[list[str], list[str]]:
        """Split an agent's direct children into leaf tools and sub-agent names."""
        leaf: list[str] = []
        children: list[str] = []
        for ref in child_refs(agent_entry):
            kind = classify(ref, functions, groups)
            if kind == "agent":
                children.append(ref)
            elif kind == "mcp":
                leaf.append(ref)
                used_groups.append(ref)
            elif kind == "builtin":
                leaf.append(ref)
                report.builtins.add(ref)
            elif kind == "group":
                leaf.append(ref)
                report.errors.append(
                    f"'{ref}': function_group _type '{groups[ref].get('_type')}' is not MCP; not carried as a server."
                )
            else:  # dangling
                report.errors.append(f"'{ref}' is referenced but not defined in functions or function_groups.")
        return leaf, children

    # Main agent's own tools, then a cycle-safe walk of the sub-agent graph.
    main_tools, main_children = leaf_and_children(entry)
    subagents: list[dict] = []
    seen_agents: set[str] = set()
    if main_key:
        seen_agents.add(main_key)
    queue: deque[str] = deque(main_children)
    while queue:
        ref = queue.popleft()
        if ref in seen_agents:
            continue  # cycle or shared sub-agent; emit once
        seen_agents.add(ref)
        agent_entry = functions[ref]
        report.agent_types[ref] = agent_entry.get("_type", "")
        leaf, children = leaf_and_children(agent_entry)
        sub: dict = {"name": ref, "description": agent_entry.get("description", f"{ref} sub-agent"), "tools": leaf}
        if children:
            sub["delegates_to"] = children  # preserve nested topology instead of flattening
        subagents.append(sub)
        queue.extend(children)

    # Prompt resolution: NAT archetypes carry a default prompt in Python when the
    # config leaves system_prompt unset. Flag rather than invent it.
    system_prompt = entry.get("system_prompt")
    if system_prompt is None:
        system_prompt = f"{RESOLVE} Default {entry['_type']} prompt (recover via NAT WorkflowBuilder)."
        report.notes.append(
            f"Main agent '{entry['_type']}' had no explicit system_prompt; its default "
            f"lives in NAT Python and must be resolved via WorkflowBuilder."
        )

    # Models: main agent's llm becomes default; every NAT llm is also a named alias.
    main_llm = entry.get("llm_name")
    if main_llm and main_llm in llms:
        models = {"default": llm_to_model(llms[main_llm], report, "default")}
    else:
        if main_llm:
            report.errors.append(f"Main agent references llm '{main_llm}' not defined in llms.")
        else:
            report.errors.append(f"Main agent '{entry['_type']}' has no llm_name; emitted {RESOLVE} model.")
        models = {"default": {"provider": RESOLVE, "model": RESOLVE, "temperature": 0.0}}
    for name, llm in llms.items():
        if name != main_llm:
            models[name] = llm_to_model(llm, report, name)

    # MCP servers: every group actually referenced by the agent tree.
    mcp_servers = {}
    for name in dict.fromkeys(used_groups):  # dedupe, preserve order
        mcp_servers[name] = mcp_server_to_fabric(name, groups[name], auth_block, report)

    harness_settings: dict = {"system_prompt": system_prompt}
    if subagents:
        harness_settings["deepagents"] = {"subagents": subagents}

    agent_name = name_override or main_key or default_name or entry["_type"].replace("_", "-")
    description = f"Migrated from a NAT {workflow['_type']}"
    if main_key:
        description += f" wrapping {entry['_type']} '{main_key}'"
    description += "."

    report.main_tools = main_tools
    report.agent_types[agent_name] = entry.get("_type", "")

    fabric: dict = {
        "schema_version": "fabric.agent/v1alpha1",
        "metadata": {"name": agent_name, "description": description},
        "harness": {
            "adapter_id": "nvidia.fabric.langchain.deepagents",
            "resolution": "preinstalled",
            "settings": harness_settings,
        },
        "models": models,
        "runtime": {"input_schema": "chat", "output_schema": "message"},
    }
    if mcp_servers:
        fabric["mcp"] = {"servers": mcp_servers}
    if main_tools:
        report.notes.append(f"Main-agent direct tools (not sub-agents): {', '.join(main_tools)}.")
    return fabric


def render_report(report: Report) -> str:
    lines = ["# NAT -> Fabric migration report", ""]

    status = "ready to run"
    if report.errors:
        status = f"blocked: {len(report.errors)} issue(s)"
    elif report.auth_gaps or report.builtins or report.env_vars:
        manual = len(report.auth_gaps) + len(report.builtins) + (1 if report.env_vars else 0)
        status = f"ready after {manual} manual step(s)"
    lines += [f"**Status: {status}**", ""]

    lines.append("## MCP servers carried across")
    lines += [f"- {item}" for item in report.carried] or ["- None."]

    if report.env_vars:
        lines += ["", "## Environment variables to set before running"]
        lines += [f"- `{env}`" for env in sorted(report.env_vars)]

    lines += ["", "## Auth requiring action (Fabric adapter gap)"]
    lines += [f"- {gap}" for gap in report.auth_gaps] or ["- None."]

    lines += ["", "## Builtin tools requiring an MCP equivalent"]
    if report.builtins:
        lines += [
            f"- `{name}`: NAT in-process tool. Needs a prebuilt MCP server equivalent before it runs under Deep Agents."
            for name in sorted(report.builtins)
        ]
    else:
        lines.append("- None. All tools are MCP servers.")

    lines += ["", "## Errors (must resolve)"]
    lines += [f"- {err}" for err in report.errors] or ["- None."]

    lines += ["", "## Features not carried (need a Fabric, Platform, or Relay home)"]
    lines += [f"- {feat}" for feat in report.features] or ["- None."]

    lines += ["", "## Notes"]
    lines += [f"- {note}" for note in report.notes] or ["- None."]
    lines.append("")
    return "\n".join(lines)


def render_analysis(fabric: dict, report: Report) -> str:
    """Read-only "what this NAT agent is and how it's composed" report.

    Same introspection as the transpile pass, without emitting a Fabric config. This
    is the analyzer surface: install NeMo Platform, hand it a NAT workflow, and it
    tells you the agent's topology, models, tools, and what would need work to move.
    """
    meta = fabric["metadata"]
    name = meta["name"]
    settings = fabric["harness"]["settings"]
    subagents = settings.get("deepagents", {}).get("subagents", [])
    models = fabric.get("models", {})
    servers = fabric.get("mcp", {}).get("servers", {})

    lines = [f"# NAT agent analysis: {name}", "", meta.get("description", ""), ""]

    lines += ["## Composition", ""]
    lines.append(f"- **{name}** ({report.agent_types.get(name, '?')}, main agent)")
    for tool in report.main_tools:
        lines.append(f"  - tool: {tool}")
    for sub in subagents:
        lines.append(f"  - **{sub['name']}** ({report.agent_types.get(sub['name'], '?')}, sub-agent)")
        for tool in sub.get("tools", []):
            lines.append(f"    - tool: {tool}")
        for delegate in sub.get("delegates_to", []):
            lines.append(f"    - delegates to: {delegate}")

    lines += ["", "## Models", ""]
    for alias, model in models.items():
        lines.append(f"- `{alias}`: {model.get('provider')} / {model.get('model')}")

    lines += ["", "## Tools", ""]
    if servers:
        lines.append("MCP servers:")
        lines += [f"- `{sname}` ({spec.get('transport')})" for sname, spec in servers.items()]
    if report.builtins:
        lines.append("NAT builtins (would need an MCP equivalent to run under Deep Agents):")
        lines += [f"- `{b}`" for b in sorted(report.builtins)]
    if not servers and not report.builtins:
        lines.append("- No tools resolved.")

    if report.features:
        lines += ["", "## Features needing another home", ""]
        lines += [f"- {feat}" for feat in report.features]
    if report.errors:
        lines += ["", "## Unresolved (custom types or missing refs)", ""]
        lines += [f"- {err}" for err in report.errors]

    lines += ["", "## Summary", ""]
    lines.append(
        f"{1 + len(subagents)} agent(s), {len(servers)} MCP server(s), {len(report.builtins)} NAT "
        f"builtin(s), {len(report.features)} feature(s) needing another home, {len(report.errors)} "
        f"unresolved item(s)."
    )
    lines.append("")
    return "\n".join(lines)


def structural_check(fabric: dict) -> None:
    """Minimal, self-contained validation of the Fabric contract.

    Does not embed the Fabric JSON Schema (NeMo-Fabric is a separate repo). For full
    validation, point --schema at a local Fabric checkout's schemas/agent.schema.json.
    """
    for key in ("schema_version", "metadata", "harness", "runtime"):
        if key not in fabric:
            raise ValueError(f"missing required top-level key: {key}")
    if "name" not in fabric["metadata"]:
        raise ValueError("metadata.name is required")
    if "adapter_id" not in fabric["harness"]:
        raise ValueError("harness.adapter_id is required")
    for name, model in fabric.get("models", {}).items():
        for key in ("provider", "model"):
            if key not in model:
                raise ValueError(f"models.{name}.{key} is required")
    for name, server in fabric.get("mcp", {}).get("servers", {}).items():
        for key in ("transport", "url", "exposure"):
            if key not in server:
                raise ValueError(f"mcp.servers.{name}.{key} is required")


def validate(fabric: dict, schema_path: Path | None) -> str:
    # Full JSON Schema validation only when a local Fabric schema is supplied.
    # Otherwise fall back to the self-contained structural check so the spike
    # carries no private Fabric artifact.
    if schema_path and schema_path.is_file():
        try:
            import json

            import jsonschema

            schema = json.loads(schema_path.read_text())
            jsonschema.validate(instance=fabric, schema=schema)
            return f"Valid against {schema_path.name} (FabricConfig, fabric.agent/v1alpha1)."
        except ImportError:
            pass
    structural_check(fabric)
    return "Passed structural check (Fabric contract: required keys + mcp server shape)."


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Transpile a NAT agent config to a Fabric agent.yaml")
    parser.add_argument("--in", dest="src", default=str(here / "nat_agent" / "config.yml"))
    parser.add_argument("--out", dest="out", default=str(here / "fabric" / "agent.yaml"))
    parser.add_argument("--report", dest="report", default=str(here / "MIGRATION_REPORT.md"))
    parser.add_argument("--name", dest="name", default=None, help="Override the emitted metadata.name.")
    parser.add_argument(
        "--schema",
        dest="schema",
        default="",
        help="Optional path to a local NeMo-Fabric schemas/agent.schema.json for full JSON Schema validation.",
    )
    parser.add_argument("--analyze", dest="analyze", action="store_true", help="Analyze the NAT agent and write ANALYSIS.md instead of emitting a Fabric config.")
    parser.add_argument("--analyze-out", dest="analyze_out", default=str(here / "ANALYSIS.md"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.src).read_text())
    report = Report()
    fabric = transpile(config, report, name_override=args.name, default_name=Path(args.src).stem)

    if args.analyze:
        apath = Path(args.analyze_out)
        apath.parent.mkdir(parents=True, exist_ok=True)
        apath.write_text(render_analysis(fabric, report))
        print(f"Wrote {apath}")
        print(f"Agents: {1 + len(fabric['harness']['settings'].get('deepagents', {}).get('subagents', []))}")
        print(f"MCP servers: {len(fabric.get('mcp', {}).get('servers', {}))}")
        print(f"Builtins: {len(report.builtins)}  Features needing another home: {len(report.features)}  Unresolved: {len(report.errors)}")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by transpile.py from nat_agent/config.yml. Do not edit by hand.\n"
        "# schema_version: fabric.agent/v1alpha1 (NeMo Fabric).\n"
    )
    out_path.write_text(header + yaml.safe_dump(fabric, sort_keys=False, default_flow_style=False))
    Path(args.report).write_text(render_report(report))

    result = validate(fabric, Path(args.schema) if args.schema else None)
    print(f"Wrote {out_path}")
    print(f"Wrote {args.report}")
    print(f"Schema check: {result}")
    print(f"MCP servers carried: {len(fabric.get('mcp', {}).get('servers', {}))}")
    print(f"Sub-agents preserved: {len(fabric['harness']['settings'].get('deepagents', {}).get('subagents', []))}")
    print(f"Auth gaps flagged: {len(report.auth_gaps)}")
    print(f"Errors: {len(report.errors)}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
