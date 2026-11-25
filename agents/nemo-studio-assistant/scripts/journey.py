#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the Studio Assistant user journey through Studio's assistant API.

Runs the steps in tests/user_journey.md against a local platform with the
assistant deployed, answers pickers with their first option, declines every
mutation approval, and prints one PASS/FAIL line per step. Exits non-zero when
any step fails.

    python3 agents/nemo-studio-assistant/scripts/journey.py [--base-url URL] [--workspace NAME]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

REFUSAL_PREFIX = "I can only help with NeMo Helix and NeMo Studio"
DECLINE_REASON = "Declined by the user journey script."
SMOKE_WORKSPACE = "journey-smoke"
TURN_TIMEOUT_SECONDS = 240


@dataclass
class Turn:
    text: str = ""
    tools: list[str] = field(default_factory=list)
    activity: list[str] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def refused(self) -> bool:
        return self.text.startswith(REFUSAL_PREFIX)


@dataclass
class Step:
    name: str
    message: str
    pathname: str
    check: Callable[[Turn], str | None]


class Journey:
    def __init__(self, base_url: str, workspace: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.workspace = workspace
        self.api = f"{self.base_url}/apis/studio/v2/assistant"
        query = urlencode({"workspace": workspace})
        self.session_id = self._post(f"{self.api}/sessions?{query}", {})["session_id"]

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def _events(self, message: str, pathname: str) -> Iterator[tuple[str, str]]:
        body = {
            "message": message,
            "workspace": self.workspace,
            "studio_base_url": f"{self.base_url}/studio",
            "studio_pathname": pathname,
        }
        request = urllib.request.Request(
            f"{self.api}/sessions/{self.session_id}/messages",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=TURN_TIMEOUT_SECONDS) as response:
            event, data = "message", []
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data:
                        yield event, "\n".join(data)
                    event, data = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))

    def _answer_permission(self, payload: dict[str, Any], turn: Turn) -> None:
        tool_input = payload.get("input") or {}
        if payload.get("tool_name") == "AskUserQuestion":
            answers = []
            for question in tool_input.get("questions") or []:
                options = question.get("options") or [""]
                first = options[0]
                label = first if isinstance(first, str) else first.get("label", "")
                turn.questions.append(f"{question.get('question')} -> {label}")
                answers.append(f'"{question.get("question")}"="{label}"')
            reason = (
                f"Your questions have been answered: {', '.join(answers)}. "
                "You can now continue with these answers in mind."
            )
            decision = {"approved": False, "reason": reason}
        else:
            turn.approvals.append(f"{tool_input.get('action')} {tool_input.get('resource')}")
            decision = {"approved": False, "reason": DECLINE_REASON}
        self._post(f"{self.api}/sessions/{self.session_id}/permissions/{payload['request_id']}", decision)

    def send(self, message: str, pathname: str) -> Turn:
        turn = Turn()
        started = time.monotonic()
        completed = False
        for event, data in self._events(message, pathname):
            if event == "done":
                completed = True
                break
            payload: Any = json.loads(data) if data.startswith(("{", "[")) else data
            if event == "error":
                turn.errors.append(str(payload))
            elif event == "permission_request":
                self._answer_permission(payload, turn)
            elif event == "input_request":
                turn.inputs.append(str(payload.get("tool_name")))
                self._post(f"{self.api}/sessions/{self.session_id}/inputs/{payload['request_id']}", {"skipped": True})
            elif isinstance(payload, dict) and payload.get("type") == "assistant":
                for block in payload.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        turn.text += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "")
                        turn.tools.append(name)
                        summary = (block.get("input") or {}).get("summary")
                        if summary:
                            turn.activity.append(f"{name}: {summary}")
        if not completed and not turn.errors:
            # An empty or cut-off stream must not pass as an answer.
            turn.errors.append("stream ended without a done event")
        turn.seconds = time.monotonic() - started
        return turn

    def workspace_exists(self, name: str) -> bool:
        try:
            urllib.request.urlopen(f"{self.base_url}/apis/entities/v2/workspaces/{name}", timeout=30)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            raise
        return True


def _require(condition: bool, failure: str) -> str | None:
    return None if condition else failure


def build_steps(journey: Journey) -> list[Step]:
    ws = journey.workspace
    dashboard = f"/studio/workspaces/{ws}/dashboard"
    return [
        Step(
            "greeting",
            "Hi! What can you help me with here?",
            dashboard,
            lambda t: _require(bool(t.text.strip()) and not t.refused, "expected a non-refusal answer"),
        ),
        Step(
            "page context",
            "How do I add a guardrail here?",
            f"/studio/workspaces/{ws}/virtual-models",
            lambda t: _require(
                not t.refused and "virtual" in t.text.lower() and "guardrail" in t.text.lower(),
                "expected an answer about guardrails on the Virtual Models page",
            ),
        ),
        Step(
            "docs search",
            "How does skill optimization work?",
            dashboard,
            lambda t: _require(not t.refused and "search_docs" in t.tools, "expected a search_docs step"),
        ),
        Step(
            "platform read",
            "List my workspaces.",
            dashboard,
            lambda t: _require(
                not t.refused and "nemo_api" in t.tools and ws in t.text, f"expected a nemo_api call naming {ws!r}"
            ),
        ),
        Step(
            "short follow-up",
            "yes",
            dashboard,
            lambda t: _require(not t.refused, "a short follow-up must not be refused"),
        ),
        Step(
            "off-topic refusal",
            "How do I bake a cheesecake?",
            dashboard,
            lambda t: _require(t.refused and not t.tools, "expected the fixed refusal with no agent run"),
        ),
        Step(
            "declined mutation",
            f"Create a new workspace named {SMOKE_WORKSPACE}.",
            f"/studio/workspaces/{ws}/dashboard",
            lambda t: _require(
                bool(t.approvals) and not journey.workspace_exists(SMOKE_WORKSPACE),
                f"expected an approval prompt and no {SMOKE_WORKSPACE!r} workspace after declining",
            ),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--verbose", "-v", action="store_true", help="print each answer in full")
    args = parser.parse_args()

    journey = Journey(args.base_url, args.workspace)
    print(f"session {journey.session_id}")
    failures = 0
    for index, step in enumerate(build_steps(journey), start=1):
        try:
            turn = journey.send(step.message, step.pathname)
            failure = step.check(turn) or ("; ".join(turn.errors) if turn.errors else None)
        except Exception as error:  # noqa: BLE001 - report and continue with the next step
            turn, failure = Turn(), f"{type(error).__name__}: {error}"
        failures += failure is not None
        status = "PASS" if failure is None else "FAIL"
        print(f"{index}. {status} {step.name} ({turn.seconds:.1f}s) tools={turn.tools}")
        for detail in (*turn.activity, *turn.questions, *(f"declined {a}" for a in turn.approvals)):
            print(f"     {detail}")
        if failure:
            print(f"     reason: {failure}")
        text = " ".join(turn.text.split())
        print(f"     answer: {text if args.verbose else text[:240]}")
    print(f"{failures} of {index} steps failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
