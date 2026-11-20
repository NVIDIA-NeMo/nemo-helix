# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom scenarios: your own requests in place of UserSim's, in one of two kinds.

- **Topics** for the ``general_open_ended`` probe: ``topics.yaml`` holds a ``set_id`` and a ``topics`` list,
  each topic a ``type`` and a ``description``. Each simulated user gets one topic and asks about it in their
  own words over several turns.
- **Tool-use requests** for the ``safety_agentic`` probe: ``bank.yaml``, an action-request bank in UserSim's
  schema (``usersim/engine/assets/safety_agentic/SCHEMA.md``), with two optional files:

  - ``actions.yaml``: action-taxonomy entries for the bank's tools, added to UserSim's own taxonomy.
  - ``outcomes.yaml``: what Studio and the report call a conversation that called the tool it should
    (``should_call``) or the one it shouldn't (``should_not_call``). UserSim's schema has no place for it.

UserSim reads the bank and the taxonomy from two environment variables, and the topics from an asset root
on ``USERSIM_ASSET_PATH``. Each step that draws, runs or scores a conversation fetches the set and points
those variables at it before UserSim loads them.
"""

import asyncio
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from pydantic import BaseModel
from usersim.engine.core import action_taxonomy, agentic_bank
from usersim.engine.core._assets import ASSET_PATH_ENV_VAR, packaged_assets_dir

BANK = "bank.yaml"
ACTIONS = "actions.yaml"
OUTCOMES = "outcomes.yaml"
TOPICS = "topics.yaml"
# Every set the CLI uploads lands in a fileset named for it, so this prefix finds them all.
FILESET_PREFIX = "user-sim-scenarios-"
_FILES = (BANK, ACTIONS, OUTCOMES, TOPICS)
_TOPIC_PROBE = "general_open_ended"
_OUTCOME_KEYS = ("should_call", "should_not_call")
_BANK_ENV = "USERSIM_SAFETY_AGENTIC_BANK"
_TAXONOMY_ENV = "USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY"
# Where the service leaves a fetched set for the preview subprocess, which installs it itself: installing
# sets process-wide variables, which the service mustn't carry into other requests.
FETCHED_ENV = "NEMO_USER_SIM_SCENARIOS_DIR"
# What ``install`` writes: the bank as given, and UserSim's taxonomy with the set's actions merged in.
_TAXONOMY = "action_taxonomy.yaml"


class ScenarioError(ValueError):
    pass


def merged_taxonomy(actions: str | None) -> dict:
    """UserSim's action taxonomy with ``actions``' entries added, each replacing a shipped one of its name."""
    shipped = yaml.safe_load(action_taxonomy._default_action_taxonomy_path().read_text())
    if not actions:
        return shipped
    extra = yaml.safe_load(actions)
    entries = extra.get("entries") if isinstance(extra, dict) else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) and e.get("name") for e in entries):
        raise ScenarioError(f"{ACTIONS} needs a top-level `entries` list of actions, each with a `name`.")
    names = {entry["name"] for entry in entries}
    kept = [entry for entry in shipped["entries"] if entry.get("name") not in names]
    return {**shipped, "entries": kept + entries}


def install(files: Mapping[str, bytes], directory: Path) -> dict[str, str]:
    """Write a scenario set to ``directory``, check UserSim accepts it and return the variables that select it.

    ``files`` maps file names to contents and must hold ``bank.yaml`` or ``topics.yaml``. The variables also
    take effect in this process, so call it before UserSim first loads the set.
    """
    if BANK not in files and TOPICS not in files:
        raise ScenarioError(f"A scenario set needs {BANK} or {TOPICS}; found {sorted(files) or 'no files'}.")
    outcome_labels(files)  # rejected here, before a run, rather than when it's scored
    directory.mkdir(parents=True, exist_ok=True)
    variables = _install_topics(files[TOPICS], directory) if TOPICS in files else {}
    if BANK in files:
        variables |= _install_bank(files, directory)
    return variables


def _topics(content: bytes) -> list[dict[str, str]]:
    document = yaml.safe_load(content)
    topics = document.get("topics") if isinstance(document, dict) else None
    if not topics or not isinstance(topics, list):
        raise ScenarioError(f"{TOPICS} needs a non-empty `topics` list.")
    for topic in topics:
        if not (
            isinstance(topic, dict) and all(isinstance(topic.get(k), str) and topic[k] for k in ("type", "description"))
        ):
            raise ScenarioError(f"Each topic in {TOPICS} needs a `type` and a `description`.")
    return [{"type": topic["type"], "description": topic["description"]} for topic in topics]


def _install_topics(content: bytes, directory: Path) -> dict[str, str]:
    """An asset root holding UserSim's general_open_ended probe with the set's topics as its only ones.

    UserSim takes a probe's assets from the first root that has the probe, so the root carries the probe's
    shipped prompts too. With no locale folders, every locale reads these topics.
    """
    topics = _topics(content)
    root = directory / "assets"
    probe = root / _TOPIC_PROBE
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True)
    for item in (packaged_assets_dir() / _TOPIC_PROBE).iterdir():
        if item.name != "topics":
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, probe / item.name)
    (probe / "topics" / "base").mkdir(parents=True)
    (probe / "topics" / "base" / "topics.json").write_text(json.dumps(topics, ensure_ascii=False, indent=2))
    _bind_users_to_topics(probe / "prompts.yaml")
    variables = {ASSET_PATH_ENV_VAR: str(root)}
    os.environ.update(variables)
    return variables


_BOUND = "Your persona shapes how you ask, not what you ask about."


def _bind_users_to_topics(prompts: Path) -> None:
    """Make the set's topic decide what a simulated user asks.

    UserSim's prompt asks for a request that reflects the persona's interests, which suits its own broad
    topics. With a custom set, users drift into their persona's job or hobbies (a warehouse clerk given
    an email to triage asks about pallet weights), so the persona keeps the voice and the topic sets the request.
    """
    document = yaml.safe_load(prompts.read_text())
    prompt = document["user_agent_system_prompt"]
    prompt = prompt.replace(
        "- Your question should reflect your education level, background, and interests.",
        f"- Your request must be about the topic below. {_BOUND} Your wording reflects your education and background.",
    )
    prompt = prompt.replace(
        "Your question or request should be about this topic:",
        "Your request must be about this topic, from your first message to your last:",
    )
    if _BOUND not in prompt:  # UserSim reworded its prompt; still bind the user
        prompt = prompt.replace(
            "<INSTRUCTIONS>", f"<INSTRUCTIONS>\n- Your request must be about the topic below. {_BOUND}", 1
        )
    document["user_agent_system_prompt"] = prompt
    prompts.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=10_000))


def _install_bank(files: Mapping[str, bytes], directory: Path) -> dict[str, str]:
    bank, taxonomy = directory / BANK, directory / _TAXONOMY
    bank.write_bytes(files[BANK])
    actions = files.get(ACTIONS)
    taxonomy.write_text(yaml.safe_dump(merged_taxonomy(actions.decode() if actions else None), sort_keys=False))
    variables = {_BANK_ENV: str(bank), _TAXONOMY_ENV: str(taxonomy)}
    os.environ.update(variables)
    try:
        # Both loaders cache by path, so a set installed at a new path is read fresh.
        action_taxonomy.load_action_taxonomy_default()
        agentic_bank.load_agentic_bank_default()
    except Exception as exc:
        raise ScenarioError(f"UserSim rejected the scenario set: {exc}") from exc
    return variables


def outcome_labels(files: Mapping[str, bytes]) -> dict[str, str]:
    """The set's names for its outcomes, from ``outcomes.yaml``; empty when it has none."""
    if OUTCOMES not in files:
        return {}
    labels = yaml.safe_load(files[OUTCOMES]) or {}
    if not isinstance(labels, dict) or set(labels) - set(_OUTCOME_KEYS):
        raise ScenarioError(f"{OUTCOMES} takes only {' and '.join(_OUTCOME_KEYS)}, each a short phrase.")
    if not all(isinstance(label, str) and label.strip() for label in labels.values()):
        raise ScenarioError(f"Each label in {OUTCOMES} must be a short phrase.")
    return {key: label.strip() for key, label in labels.items()}


def read_local(path: Path) -> dict[str, bytes]:
    """A scenario set from a folder of its files, or from a single bank or topics file."""
    if path.is_file():
        return {TOPICS if path.name == TOPICS else BANK: path.read_bytes()}
    return {name: (path / name).read_bytes() for name in _FILES if (path / name).is_file()}


def probes(files: Mapping[str, bytes]) -> list[str]:
    """The probes a set's files feed: topics run in general_open_ended, a bank in safety_agentic."""
    return sorted({*([_TOPIC_PROBE] if TOPICS in files else []), *(["safety_agentic"] if BANK in files else [])})


def set_id(files: Mapping[str, bytes]) -> str:
    """The set's name, which names the fileset the CLI uploads it to: the bank's ``bank_id`` or the topics' ``set_id``."""
    source, key = (BANK, "bank_id") if BANK in files else (TOPICS, "set_id")
    document = yaml.safe_load(files[source])
    identifier = document.get(key) if isinstance(document, dict) else None
    if not isinstance(identifier, str) or not identifier:
        raise ScenarioError(f"{source} needs a `{key}`.")
    return identifier


def fetch(files: FilesClient, workspace: str, fileset: str) -> dict[str, bytes]:
    """A scenario set from its fileset."""
    found = {}
    for name in _FILES:
        try:
            found[name] = files.download_file(workspace=workspace, name=fileset, path=name).read()
        except NotFoundError:
            pass
    return _complete(found, fileset)


def _complete(found: dict[str, bytes], fileset: str) -> dict[str, bytes]:
    if BANK not in found and TOPICS not in found:
        raise ScenarioError(f"Fileset {fileset} has neither {BANK} nor {TOPICS}.")
    return found


async def fetch_async(files: AsyncFilesClient, workspace: str, fileset: str) -> dict[str, bytes]:
    """``fetch`` for the service."""
    found = {}
    for name in _FILES:
        try:
            found[name] = await (await files.download_file(workspace=workspace, name=fileset, path=name)).read()
        except NotFoundError:
            pass
    return _complete(found, fileset)


def save_raw(files: Mapping[str, bytes], directory: Path) -> Path:
    """Write a fetched set as is, for a subprocess to ``install`` with ``read_local``."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_bytes(content)
    return directory


class ScenarioSet(BaseModel):
    """An uploaded set, as Studio offers it for a new run."""

    fileset: str
    kind: Literal["topics", "tool requests"]
    probes: list[str]
    # How many topics a topic set holds. A bank's requests aren't counted.
    size: int | None = None


async def list_sets(files: AsyncFilesClient, workspace: str) -> list[ScenarioSet]:
    """The sets uploaded to ``workspace``, by fileset name. A fileset without a bank or topics is left out."""
    names = [fs.name async for fs in (await files.list_filesets(workspace=workspace)).items()]
    names = sorted(name for name in names if name.startswith(FILESET_PREFIX))
    fetched = await asyncio.gather(*(fetch_async(files, workspace, name) for name in names), return_exceptions=True)
    sets = []
    for name, found in zip(names, fetched, strict=True):
        if isinstance(found, ScenarioError):
            continue
        if isinstance(found, BaseException):
            raise found
        topics = TOPICS in found
        sets.append(
            ScenarioSet(
                fileset=name,
                kind="topics" if topics else "tool requests",
                probes=probes(found),
                size=len(_topics(found[TOPICS])) if topics else None,
            )
        )
    return sets
