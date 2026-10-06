#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline conversion of one Gym Responses record. Never fetch embedded paths/URLs.

Prefer --source-atif for Harbor-backed records. Output is private evidence, not
publication approval, proof, or a claim of lossless round-trip conversion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_ATIF_BYTES = 25 * 1024 * 1024
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
SUFFIX_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
RECEIPT_SCHEMA = "nemo.eval_author.gym_to_atif.v1"
ATIF_VERSION = re.compile(r"ATIF-v1\.[0-7]")


def _validate_message(message: Any, *, location: str) -> tuple[int, bool]:
    """Same message boundary eval-author-trace-environment's prepare enforces."""
    if isinstance(message, str):
        return 0, bool(message.strip())
    if not isinstance(message, list):
        raise ConversionError(f"{location} must be text or a list of ATIF content parts")
    image_count = 0
    has_text = False
    for index, part in enumerate(message):
        if not isinstance(part, dict):
            raise ConversionError(f"{location}[{index}] must be an object")
        part_type = part.get("type")
        text = part.get("text")
        if part_type == "text" and isinstance(text, str):
            has_text = has_text or bool(text.strip())
        elif part_type == "image" and isinstance(part.get("source"), dict):
            image_count += 1
        else:
            raise ConversionError(f"{location}[{index}] is not a supported text or image content part")
    return image_count, has_text


def _validate_trajectory(payload: Any, *, location: str = "trajectory", depth: int = 0) -> dict[str, Any]:
    """Same structural boundary eval-author-trace-environment's prepare enforces.

    Kept as a local copy so this skill stands alone when copied without its
    siblings; dev tests additionally validate projected ATIF against Harbor's
    own models.
    """
    if depth > 8:
        raise ConversionError("embedded ATIF trajectories exceed the depth limit of 8")
    if not isinstance(payload, dict):
        raise ConversionError(f"{location} must be an object")
    version = payload.get("schema_version")
    if not isinstance(version, str) or ATIF_VERSION.fullmatch(version) is None:
        raise ConversionError(f"{location}.schema_version must identify ATIF v1.x")
    agent = payload.get("agent")
    if not isinstance(agent, dict) or not isinstance(agent.get("name"), str) or not agent["name"].strip():
        raise ConversionError(f"{location}.agent.name must be nonempty text")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ConversionError(f"{location}.steps must be a nonempty list")

    image_count = 0
    image_only_user_steps: list[int] = []
    user_step_count = 0
    for index, step in enumerate(steps, start=1):
        step_location = f"{location}.steps[{index - 1}]"
        if not isinstance(step, dict):
            raise ConversionError(f"{step_location} must be an object")
        if step.get("step_id") != index:
            raise ConversionError(f"{step_location}.step_id must be sequential from 1")
        source = step.get("source")
        if source not in {"user", "agent", "system"}:
            raise ConversionError(f"{step_location}.source must be user, agent, or system")
        if source == "user":
            user_step_count += 1
        step_images, has_text = _validate_message(step.get("message", ""), location=f"{step_location}.message")
        image_count += step_images
        if source == "user" and step_images and not has_text:
            image_only_user_steps.append(index)

        calls = step.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ConversionError(f"{step_location}.tool_calls must be a list")
        call_ids: set[str] = set()
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ConversionError(f"{step_location}.tool_calls[{call_index}] must be an object")
            call_id = call.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                raise ConversionError(f"{step_location} has a missing or duplicate tool_call_id")
            call_ids.add(call_id)
        observation = step.get("observation")
        if observation is not None:
            if not isinstance(observation, dict):
                raise ConversionError(f"{step_location}.observation must be an object")
            results = observation.get("results", [])
            if not isinstance(results, list):
                raise ConversionError(f"{step_location}.observation.results must be a list")
            for result in results:
                if not isinstance(result, dict):
                    raise ConversionError(f"{step_location}.observation results must be objects")
                source_call_id = result.get("source_call_id")
                if source_call_id is not None and source_call_id not in call_ids:
                    raise ConversionError(
                        f"{step_location} observation references unknown tool call {source_call_id!r}"
                    )
                result_images, _ = _validate_message(result.get("content", ""), location="observation content")
                image_count += result_images

    subagents = payload.get("subagent_trajectories") or []
    if not isinstance(subagents, list):
        raise ConversionError(f"{location}.subagent_trajectories must be a list")
    for index, subagent in enumerate(subagents):
        child = _validate_trajectory(subagent, location=f"{location}.subagent_trajectories[{index}]", depth=depth + 1)
        image_count += child["image_count"]
        image_only_user_steps.extend(child["image_only_user_steps"])
    if depth == 0 and user_step_count == 0:
        raise ConversionError("trajectory must contain at least one root user step")
    return {"image_count": image_count, "image_only_user_steps": image_only_user_steps}


class ConversionError(ValueError):
    """Content-free conversion failure."""


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError("duplicate JSON keys are unsupported")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise ConversionError("non-finite JSON numbers are unsupported")


def decode(data: bytes | str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ConversionError("source must be complete, bounded UTF-8 JSON; select --row for JSONL") from error


def encode(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_source(path: Path, row: int | None = None) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ConversionError("source must be an explicitly supplied regular file")
    if row is None:
        if path.suffix.lower() == ".jsonl":
            raise ConversionError("JSONL requires an explicit one-based --row; episodes are never merged")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ConversionError("source exceeds the raw-byte limit")
        data = path.read_bytes()
    else:
        if row < 1:
            raise ConversionError("--row must be positive")
        with path.open("rb") as stream:
            for _ in range(row):
                data = stream.readline(MAX_SOURCE_BYTES + 1)
                if not data:
                    raise ConversionError("selected JSONL row does not exist")
                if len(data) > MAX_SOURCE_BYTES:
                    raise ConversionError("JSONL record exceeds the raw-byte limit")
    if len(data) > MAX_SOURCE_BYTES or not data.strip():
        raise ConversionError("source is empty or exceeds the raw-byte limit")
    return data


def envelope(record: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(record, dict):
        raise ConversionError("one Gym rollout/response object is required")
    if "response" in record:
        response = record["response"]
        request = record.get("responses_create_params")
        if not isinstance(request, dict):
            raise ConversionError("rollout requires responses_create_params")
    else:
        response, request = record, {}
        if record.get("object") != "response":
            raise ConversionError("expected a Gym rollout envelope or standalone Responses object")
    if not isinstance(response, dict) or not isinstance(response.get("output"), list):
        raise ConversionError("Gym response.output must be an array")
    return request, response


def _signature(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    # Normalize only message wrapper defaults, never tool kinds or fuzzy text.
    if item.get("type", "message") != "message" or "role" not in item:
        return item
    return {
        **{k: v for k, v in item.items() if k not in {"id", "status"} and not (k == "phase" and v is None)},
        "type": "message",
    }


class Projection:
    def __init__(self, record: dict[str, Any], source_digest: str):
        self.record = record
        self.source_digest = source_digest
        self.steps: list[dict[str, Any]] = []
        self.calls: dict[str, dict[str, Any]] = {}
        self.results: set[str] = set()
        self.uncertainties = [
            "Agent implementation/version and per-item timing are not inferred from model names or response timestamps.",
            "ATIF step segmentation follows Gym item order, not an inferred original LLM-call grouping.",
            "ATIF session_id is derived from the source record digest, not a native session identifier.",
        ]
        self.losses = ["Gym-specific fields not projected into ATIF remain in the exact private Gym source record."]
        self.operations: list[str] = []
        self.bridge = isinstance(record.get("atif_conversion"), dict)
        self.last_type: str | None = None

    def image(self, url: Any, path: str) -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise ConversionError(f"image URL unavailable at {path}; supply the original ATIF")
        if url.startswith("data:"):
            media_type = url[5:].split(";", 1)[0].split(",", 1)[0]
        else:
            media_type = SUFFIX_TYPES.get(Path(urlsplit(url).path).suffix.lower())
            if media_type:
                self.uncertainties.append(f"Image MIME inferred from path suffix, not fetched or verified: {path}")
        if media_type not in IMAGE_TYPES:
            raise ConversionError(f"image MIME is not established at {path}; supply source ATIF or a typed data URI")
        self.uncertainties.append(
            f"Image bytes/references are retained without fetching or validating the media: {path}"
        )
        return {"type": "image", "source": {"media_type": media_type, "path": url}}

    def serialized_parts(self, text: str, path: str) -> list[dict[str, Any]] | None:
        if not self.bridge or not text.lstrip().startswith("["):
            return None
        try:
            value = decode(text)
        except ConversionError:
            return None
        if not isinstance(value, list) or not any(isinstance(p, dict) and p.get("type") == "image" for p in value):
            return None
        result = []
        for part in value:
            if not isinstance(part, dict):
                raise ConversionError(f"malformed serialized ATIF content at {path}")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                result.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image" and isinstance(part.get("source"), dict):
                source = part["source"]
                if (
                    not isinstance(source.get("media_type"), str)
                    or source["media_type"] not in IMAGE_TYPES
                    or not isinstance(source.get("path"), str)
                    or not source["path"].strip()
                ):
                    raise ConversionError(f"serialized image source is unsupported at {path}; use original ATIF")
                result.append({"type": "image", "source": {"media_type": source["media_type"], "path": source["path"]}})
            else:
                raise ConversionError(f"unsupported serialized content at {path}")
        self.operations.append(f"Recovered serialized ATIF image parts: {path}")
        self.uncertainties.append(
            f"ATIF-shaped text was interpreted as serialized content using Harbor-bridge metadata: {path}"
        )
        return result

    def content(self, value: Any, path: str) -> str | list[dict[str, Any]]:
        if isinstance(value, str):
            return self.serialized_parts(value, path) or value
        if not isinstance(value, list):
            raise ConversionError(f"unsupported message/tool content at {path}")
        parts = []
        for index, part in enumerate(value):
            pointer = f"{path}[{index}]"
            if not isinstance(part, dict):
                raise ConversionError(f"content part must be an object at {pointer}")
            kind = part.get("type")
            if not isinstance(kind, str):
                raise ConversionError(f"content part type must be text at {pointer}")
            if kind in {"input_text", "output_text", "text"} and isinstance(part.get("text"), str):
                recovered = self.serialized_parts(part["text"], pointer)
                parts.extend(recovered or [{"type": "text", "text": part["text"]}])
            elif kind == "input_image":
                parts.append(self.image(part.get("image_url"), pointer))
            elif kind == "refusal" and isinstance(part.get("refusal"), str):
                parts.append({"type": "text", "text": part["refusal"]})
                self.operations.append(f"Refusal retained as text; not a success label: {pointer}")
            else:
                raise ConversionError(
                    f"unsupported content part at {pointer}; original ATIF or an explicit adapter is required"
                )
        return parts

    def step(self, source: str, message: Any, item: dict[str, Any], path: str) -> dict[str, Any]:
        metadata = {"item_path": path}
        for key in ("id", "role", "phase", "status", "namespace"):
            if key in item:
                metadata[key] = item[key]
        step = {"step_id": len(self.steps) + 1, "source": source, "message": message, "extra": {"gym": metadata}}
        self.steps.append(step)
        return step

    def item(self, item: Any, path: str) -> None:
        if not isinstance(item, dict):
            raise ConversionError(f"Gym item must be an object at {path}")
        kind = item.get("type", "message" if "role" in item else None)
        if not isinstance(kind, str):
            raise ConversionError(f"Gym item type must be text at {path}")
        if kind == "message":
            role = item.get("role")
            if not isinstance(role, str) or role not in {"user", "assistant", "system", "developer"}:
                raise ConversionError(f"unsupported message role at {path}")
            source = "agent" if role == "assistant" else "system" if role == "developer" else role
            if role == "developer":
                self.losses.append(f"Developer role projected to system with original role retained in extra: {path}")
            self.step(source, self.content(item.get("content"), path + ".content"), item, path)
        elif kind == "function_call":
            call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
            if not isinstance(call_id, str) or not call_id or call_id in self.calls:
                raise ConversionError(f"missing or duplicate call_id at {path}")
            if not isinstance(name, str) or not name or not isinstance(arguments, str):
                raise ConversionError(f"invalid function call at {path}")
            if item.get("namespace") is not None:
                raise ConversionError(f"namespaced function calls need an explicit identity adapter at {path}")
            arguments = decode(arguments)
            if not isinstance(arguments, dict):
                raise ConversionError(f"function arguments must encode an object at {path}")
            step = self.steps[-1] if self.last_type == "function_call" else self.step("agent", "", item, path)
            call = {
                "tool_call_id": call_id,
                "function_name": name,
                "arguments": arguments,
                "extra": {"gym_item_path": path},
            }
            step.setdefault("tool_calls", []).append(call)
            self.calls[call_id] = step
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or call_id not in self.calls or call_id in self.results:
                raise ConversionError(f"unmatched or duplicate tool result at {path}; no tool call will be invented")
            result = {
                "source_call_id": call_id,
                "content": self.content(item.get("output"), path + ".output"),
                "extra": {"gym_item_path": path, "gym_status": item.get("status")},
            }
            if self.steps[-1] is not self.calls[call_id]:
                self.uncertainties.append(
                    f"Tool result follows later items; ATIF attachment changes presentation order: {path}"
                )
            self.calls[call_id].setdefault("observation", {"results": []})["results"].append(result)
            self.results.add(call_id)
            self.operations.append(
                f"Tool observation attached to its recorded call; original item position retained: {path}"
            )
        elif kind == "reasoning":
            summary = item.get("summary", [])
            if not isinstance(summary, list) or any(
                not isinstance(p, dict) or p.get("type") != "summary_text" or not isinstance(p.get("text"), str)
                for p in summary
            ):
                raise ConversionError(f"unsupported reasoning summary at {path}")
            step = self.step("agent", "", item, path)
            step["extra"]["gym"]["reasoning_summary"] = copy.deepcopy(summary)
            self.uncertainties.append(f"Recorded reasoning summary is not treated as verbatim model reasoning: {path}")
            if item.get("encrypted_content") or item.get("content"):
                self.losses.append(f"Additional/encrypted reasoning retained only in original Gym record: {path}")
        else:
            raise ConversionError(f"unsupported Gym item type at {path}; supply original ATIF or an explicit adapter")
        self.last_type = kind

    def convert(self, output_scope: str = "auto") -> dict[str, Any]:
        if output_scope not in {"auto", "full", "generated"}:
            raise ConversionError("unsupported output scope")
        request, response = envelope(self.record)
        if request.get("previous_response_id") or request.get("conversation"):
            raise ConversionError(
                "remote conversation references are not resolved; supply self-contained input or original ATIF"
            )
        incoming = request.get("input", [])
        if isinstance(incoming, str):
            incoming = [{"type": "message", "role": "user", "content": incoming}]
        if not isinstance(incoming, list):
            raise ConversionError("Gym request.input must be text or an item array")
        outgoing = response["output"]
        if output_scope == "auto":
            prefix = (
                bool(incoming)
                and len(outgoing) >= len(incoming)
                and all(_signature(a) == _signature(b) for a, b in zip(incoming, outgoing))
            )
            prompt_items = any(
                isinstance(p, dict) and isinstance(p.get("role"), str) and p["role"] in {"user", "system", "developer"}
                for p in outgoing
            )
            if prefix or not incoming:
                output_scope = "full"
                if prefix:
                    self.operations.append("Exact request prefix already in response.output; not duplicated")
            elif prompt_items:
                raise ConversionError(
                    "ambiguous prompt/history overlap; choose --output-scope full or generated explicitly"
                )
            else:
                output_scope = "generated"
        self.operations.append("Gym response output scope: " + output_scope)
        items = [(p, f"$.response.output[{i}]") for i, p in enumerate(outgoing)]
        if output_scope == "generated":
            items = [(p, f"$.responses_create_params.input[{i}]") for i, p in enumerate(incoming)] + items
        instructions = request.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ConversionError("request instructions must be text")
        if instructions:
            instruction = {"role": "system", "content": instructions}
            if not items or _signature(items[0][0]) != _signature(instruction):
                items.insert(0, (instruction, "$.responses_create_params.instructions"))
                self.operations.append("Request instructions projected as a leading system step")
        for item, path in items:
            self.item(item, path)
        if not any(s["source"] == "user" for s in self.steps):
            raise ConversionError("Gym record contains no root user instruction; it will not be invented")
        if self.calls.keys() - self.results:
            self.uncertainties.append("Some recorded calls have no recorded result; no observations were invented")
        model = response.get("model") or request.get("model")
        agent = {"name": "unknown-gym-agent", "version": "unknown"}
        if isinstance(model, str) and model:
            agent["model_name"] = model
        tools = request.get("tools", response.get("tools", []))
        if tools is not None:
            if not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools):
                raise ConversionError("tool definitions must be an array of objects")
            agent["tool_definitions"] = copy.deepcopy(tools)
        source = {
            "record_sha256": self.source_digest,
            "response_id": response.get("id"),
            "response_status": response.get("status"),
            "adapter": "gym_to_atif.v1",
        }
        if "reward" in self.record:
            source["reported_reward"] = self.record["reward"]
        if response.get("usage") is not None:
            source["aggregate_usage"] = copy.deepcopy(response["usage"])
        if response.get("status") not in (None, "completed"):
            self.uncertainties.append("Gym response is not completed; status/reward do not establish task proof")
        if self.bridge:
            self.losses.append(
                "Upstream ATIF-to-Gym conversion can omit metadata/subagent/tool-definition evidence; consult original ATIF"
            )
        return {
            "schema_version": "ATIF-v1.7",
            "session_id": "gym-" + self.source_digest.removeprefix("sha256:"),
            "agent": agent,
            "steps": self.steps,
            "extra": {
                "gym_source": source,
                "normalization": {
                    "uncertainties": self.uncertainties,
                    "losses": self.losses,
                    "gym_operations": self.operations,
                },
            },
        }


def normalize(
    record_bytes: bytes, source_atif: bytes | None = None, *, allow_projection: bool = False, output_scope: str = "auto"
) -> tuple[bytes, dict[str, Any]]:
    if len(record_bytes) > MAX_SOURCE_BYTES or (source_atif is not None and len(source_atif) > MAX_SOURCE_BYTES):
        raise ConversionError("source exceeds the raw-byte limit")
    record = decode(record_bytes)
    envelope(record)
    bridge = record.get("atif_conversion")
    if bridge is not None:
        if not isinstance(bridge, dict):
            raise ConversionError("atif_conversion metadata must be an object")
        for field in ("trajectories", "source_trajectory_paths"):
            if field in bridge and not isinstance(bridge[field], list):
                raise ConversionError("atif_conversion trajectory metadata must be arrays")
    # No path from atif_conversion is ever dereferenced. Only a caller-supplied
    # --source-atif can authorize reading an original trajectory.
    if source_atif is not None:
        payload = decode(source_atif)
        if not isinstance(payload, dict):
            raise ConversionError("supplied source ATIF must be an object")
        descriptions = bridge.get("trajectories", []) if isinstance(bridge, dict) else []
        known_ids = {
            d.get("session_id") for d in descriptions if isinstance(d, dict) and isinstance(d.get("session_id"), str)
        }
        if known_ids and payload.get("session_id") not in known_ids:
            raise ConversionError("supplied ATIF session does not match advertised Gym trajectory metadata")
        canonical = source_atif
        basis = "provided_atif"
        uncertainties = [
            "Gym/ATIF association is operator-supplied; session metadata is checked when available.",
            "Referenced media are not copied, fetched or validated; retain the original media alongside the ATIF.",
        ]
        if len(descriptions) > 1:
            uncertainties.append(
                "One explicitly selected ATIF from a multi-step Harbor run is retained; Gym aggregate reward is not transferred."
            )
        losses = []
    else:
        if bridge is not None and not allow_projection:
            raise ConversionError(
                "Gym advertises source ATIF; supply --source-atif or explicitly opt into --allow-projection"
            )
        if isinstance(bridge, dict) and (
            len(bridge.get("trajectories", [])) > 1 or len(bridge.get("source_trajectory_paths", [])) > 1
        ):
            raise ConversionError("multi-step Harbor projection is ambiguous; explicitly supply one original ATIF")
        projection = Projection(record, digest(record_bytes))
        payload = projection.convert(output_scope)
        canonical = encode(payload)
        if len(canonical) > MAX_ATIF_BYTES:
            raise ConversionError("projected ATIF exceeds the canonical byte limit")
        basis, uncertainties, losses = "gym_projection", projection.uncertainties, projection.losses
    # Same structural boundary prepare enforces; dev tests also validate against Harbor.
    try:
        _validate_trajectory(payload)
    except (ValueError, TypeError, RecursionError):
        raise ConversionError("ATIF structural validation failed; inspect the private source evidence") from None
    return canonical, {
        "schema": RECEIPT_SCHEMA,
        "basis": basis,
        "source_kind": "atif" if source_atif is not None else "gym",
        "gym_sha256": digest(record_bytes),
        "atif_sha256": digest(canonical),
        "source_atif_sha256": digest(source_atif) if source_atif is not None else None,
        "atif_bytes_preserved": source_atif is not None,
        "lossless_round_trip_claimed": False,
        "uncertainties": uncertainties,
        "losses": losses,
        "media_fetched": False,
    }


def write_output(output: Path, raw: bytes, atif: bytes, receipt: dict[str, Any]) -> None:
    if output.exists() or output.is_symlink():
        raise ConversionError("output directory already exists; use a new private directory")
    old_umask = os.umask(0o077)
    written: list[Path] = []
    created = False
    try:
        output.mkdir(parents=True, mode=0o700)
        created = True
        for name, data in (("source.gym.json", raw), ("trace.atif.json", atif), ("conversion.json", encode(receipt))):
            path = output / name
            with path.open("xb") as stream:
                written.append(path)
                stream.write(data)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        if created:
            output.rmdir()
        raise
    finally:
        os.umask(old_umask)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--row", type=int, help="One-based physical JSONL line; never merges records")
    parser.add_argument(
        "--source-atif", type=Path, help="Explicit original trajectory; never read from embedded path metadata"
    )
    parser.add_argument(
        "--allow-projection",
        action="store_true",
        help="Allow a lossy Gym projection when a source ATIF is advertised but unavailable",
    )
    parser.add_argument("--output-scope", choices=("auto", "full", "generated"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = read_source(args.input, args.row)
        original = read_source(args.source_atif) if args.source_atif else None
        atif, receipt = normalize(raw, original, allow_projection=args.allow_projection, output_scope=args.output_scope)
        receipt["selected_line"] = args.row
        write_output(args.output_dir, raw, atif, receipt)
        print(
            json.dumps(
                {
                    "status": "normalized",
                    "basis": receipt["basis"],
                    "source_kind": receipt["source_kind"],
                    "output_dir": str(args.output_dir),
                    "uncertainty_count": len(receipt["uncertainties"]),
                    "loss_count": len(receipt["losses"]),
                }
            )
        )
        return 0
    except (ConversionError, OSError, ValueError, TypeError, RecursionError) as error:
        # Source values, token strings, URLs, and decoded provider errors stay private.
        message = str(error) if isinstance(error, ConversionError) else "normalization failed structural or file checks"
        print(json.dumps({"status": "error", "message": message}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
