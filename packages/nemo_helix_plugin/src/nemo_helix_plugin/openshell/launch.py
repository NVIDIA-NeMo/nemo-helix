# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a deployed agent after its local Helix auth proxy is ready.

The proxy adds the deployment's service and creator identity to Helix requests.
OpenShell's network policy controls which requests the sandbox can make.
"""

import argparse
import asyncio
import signal
from contextlib import suppress

import uvicorn
from nemo_helix_plugin.client.auth_proxy import build_auth_proxy_app


async def run(command: list[str], *, upstream: str, principal: str, on_behalf_of: str, port: int) -> int:
    app = build_auth_proxy_app(
        base_url=upstream,
        headers={"X-NHX-Principal-Id": f"service:{principal}", "X-NHX-Principal-On-Behalf-Of": on_behalf_of},
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=3)
    )

    # Uvicorn does not install signal handlers on a worker thread. This parent
    # owns shutdown of both the proxy and the agent.
    def serve() -> None:
        try:
            server.run()
        except SystemExit as exc:
            raise RuntimeError("Platform auth proxy failed to start.") from exc

    proxy = asyncio.create_task(asyncio.to_thread(serve))
    process = None
    process_wait = None
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    stopped = asyncio.create_task(stopping.wait())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if stopping.is_set():
                    return 1
                if proxy.done():
                    await proxy
                    raise RuntimeError("Platform auth proxy failed to start.")
                await asyncio.sleep(0.02)
        process = await asyncio.create_subprocess_exec(*command)
        process_wait = asyncio.create_task(process.wait())
        await asyncio.wait({proxy, process_wait, stopped}, return_when=asyncio.FIRST_COMPLETED)
        return process.returncode if process.returncode is not None else 1
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                async with asyncio.timeout(5):
                    await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()
        server.should_exit = True
        try:
            async with asyncio.timeout(5):
                await proxy
        except (TimeoutError, RuntimeError):
            proxy.cancel()
        stopped.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(proxy, stopped, *([process_wait] if process_wait else []), return_exceptions=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--principal", required=True)
    parser.add_argument("--on-behalf-of", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("An agent command is required.")
    raise SystemExit(
        asyncio.run(
            run(
                command,
                upstream=args.upstream,
                principal=args.principal,
                on_behalf_of=args.on_behalf_of,
                port=args.port,
            )
        )
    )


if __name__ == "__main__":
    main()
