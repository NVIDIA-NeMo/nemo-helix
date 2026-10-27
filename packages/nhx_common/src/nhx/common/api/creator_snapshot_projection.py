# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project registered creator fields from serialized HTTP responses."""

import json
from dataclasses import dataclass
from re import Pattern

from fastapi import FastAPI
from fastapi.routing import iter_route_contexts
from nemo_helix_plugin.api.creator_snapshot import CreatorSnapshotRoute
from nhx.common.auth.creator_snapshot import CreatorSnapshotCapabilities
from nhx.common.auth.dependencies import auth_client_context
from starlette.routing import compile_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send


@dataclass(frozen=True)
class CreatorSnapshotResponseRoute:
    path_regex: Pattern[str]
    methods: frozenset[str]
    paths: tuple[str, ...]


def _remove_path(value: object, parts: tuple[str, ...]) -> bool:
    if not parts:
        return False
    part, *remaining = parts
    if part == "[]":
        if not isinstance(value, list):
            return False
        changed = False
        for item in value:
            changed = _remove_path(item, tuple(remaining)) or changed
        return changed
    if not isinstance(value, dict):
        return False
    if part.endswith("[]"):
        items = value.get(part[:-2])
        if not isinstance(items, list):
            return False
        changed = False
        for item in items:
            changed = _remove_path(item, tuple(remaining)) or changed
        return changed
    if not remaining:
        if part in value:
            del value[part]
            return True
        return False
    return _remove_path(value.get(part), tuple(remaining))


def _project_json_body(body: bytes, paths: tuple[str, ...]) -> bytes:
    content = json.loads(body)
    changed = False
    for path in paths:
        changed = _remove_path(content, tuple(path.split("."))) or changed
    if not changed:
        return body
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def collect_creator_snapshot_routes(app: FastAPI) -> list[CreatorSnapshotResponseRoute]:
    """Resolve route declarations after all router prefixes have been applied."""
    registrations = []
    for route_context in iter_route_contexts(app.router.routes):
        route = route_context.original_route
        if not isinstance(route, CreatorSnapshotRoute):
            continue
        paths = route.response_paths.get(route.response_model)
        if paths is None or route_context.path is None or route_context.methods is None:
            continue
        path_regex, _, _ = compile_path(route_context.path)
        registrations.append(CreatorSnapshotResponseRoute(path_regex, frozenset(route_context.methods), paths))
    return registrations


class CreatorSnapshotProjectionMiddleware:
    """Project registered JSON response paths under the request's auth context."""

    def __init__(self, app: ASGIApp, registrations: list[CreatorSnapshotResponseRoute]) -> None:
        self.app = app
        self.registrations = registrations

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        registration = next(
            (
                item
                for item in self.registrations
                if scope["method"] in item.methods and item.path_regex.fullmatch(scope["path"])
            ),
            None,
        )
        if registration is None:
            await self.app(scope, receive, send)
            return

        auth_client = auth_client_context.get()
        if auth_client is not None and CreatorSnapshotCapabilities.from_auth_client(auth_client).can_read:
            await self.app(scope, receive, send)
            return

        response_start: Message | None = None
        body_parts: list[bytes] = []

        async def send_projected(message: Message) -> None:
            nonlocal response_start
            if message["type"] == "http.response.start":
                status = message["status"]
                content_type = next(
                    (value for name, value in message["headers"] if name.lower() == b"content-type"),
                    b"",
                )
                if 200 <= status < 300 and content_type.split(b";", 1)[0].lower() == b"application/json":
                    response_start = message
                else:
                    await send(message)
                return
            if message["type"] != "http.response.body" or response_start is None:
                await send(message)
                return
            body_parts.append(message.get("body", b""))
            if message.get("more_body", False):
                return
            original_body = b"".join(body_parts)
            projected_body = _project_json_body(original_body, registration.paths)
            if projected_body != original_body:
                response_start["headers"] = [
                    (name, value)
                    for name, value in response_start["headers"]
                    if name.lower() not in {b"content-length", b"etag", b"content-md5"}
                ]
                response_start["headers"].append((b"content-length", str(len(projected_body)).encode()))
            await send(response_start)
            await send({"type": "http.response.body", "body": projected_body})

        await self.app(scope, receive, send_projected)
