# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone trusted plugin host, executed with the revision's Python.

Keep this file independent of nhx-platform-runner: it runs in a separate venv
containing only the plugin, nemo-helix-plugin, FastAPI, and Uvicorn.
The Unix socket is private transport, not a public authentication boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import entry_points

import uvicorn
from fastapi import FastAPI
from nemo_helix_plugin.service import NemoService


def create_app(name: str) -> FastAPI:
    matches = [ep for ep in entry_points(group="nemo.services") if ep.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one nemo.services entry point for {name!r}; found {len(matches)}")
    service_class = matches[0].load()
    if not isinstance(service_class, type) or not issubclass(service_class, NemoService):
        raise TypeError("Service entry point must be a NemoService subclass")
    service = service_class()
    if service.name != name:
        raise ValueError("Service name does not match the registered plugin name")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            await service.on_startup()
            yield
        finally:
            await service.on_shutdown()

    app = FastAPI(title=f"Community plugin: {name}", lifespan=lifespan)
    for router in service.get_routers():
        app.include_router(router.router, prefix=f"/apis/{name}{router.prefix}", tags=[router.tag or name])
    for exception_type, handler in service.get_exception_handlers().items():
        app.add_exception_handler(exception_type, handler)

    @app.get("/_helix/ready", include_in_schema=False)
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/_helix/descriptor", include_in_schema=False)
    async def descriptor() -> dict:
        return {"protocol_version": 1, "name": name, "openapi": app.openapi()}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(args.name), uds=args.socket, access_log=False)


if __name__ == "__main__":
    main()
