# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "nemo-relay==0.9.3",
#   "langchain==1.3.17",
#   "langchain-core==1.6.0",
#   "opentelemetry-proto==1.43.0",
#   "protobuf==6.33.6",
# ]
# ///

"""Reproduce the synthetic, provider-free Relay 0.9.3 full-export scenario.

Run with uv's script environment, not the workspace's newer Relay dependency.
See README.md beside this script for provenance and expected nondeterminism.
"""

import argparse
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import nemo_relay
from google.protobuf.json_format import MessageToDict
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict
from nemo_relay.integrations.langchain._serialization import model_response_to_json
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


async def record_calls() -> None:
    async def model(request):
        return model_response_to_json(
            ModelResponse(
                result=[
                    AIMessage(
                        content="hello back",
                        id="msg-1",
                        response_metadata={"model_name": "recorded-model", "model_provider": "recorded-provider"},
                        usage_metadata={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                    )
                ]
            ),
            nemo_relay.typed.BestEffortAnyCodec(),
        )

    content = json.loads(
        json.dumps({"model": "recorded-model", "messages": messages_to_dict([HumanMessage(content="hello")])})
    )
    await nemo_relay.llm.execute(
        "recorded-model",
        nemo_relay.LLMRequest({}, content),
        model,
        model_name="recorded-model",
    )
    codec = nemo_relay.typed.BestEffortAnyCodec()
    await nemo_relay.typed.tool_execute(
        name="lookup",
        args={"query": "hello"},
        func=lambda args: nemo_relay.ToolExecutionResult({"answer": "world"}),
        args_codec=codec,
        result_codec=codec,
    )

    def fail(args):
        raise ValueError("recorded failure")

    try:
        await nemo_relay.typed.tool_execute(name="broken", args={}, func=fail, args_codec=codec, result_codec=codec)
    except Exception as error:
        if "recorded failure" not in str(error):
            raise
    else:
        raise AssertionError("the synthetic failing tool unexpectedly succeeded")


def capture(output: Path) -> None:
    bodies: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    subscriber = nemo_relay.OpenTelemetrySubscriber(
        nemo_relay.OpenTelemetryConfig("full", f"http://127.0.0.1:{server.server_port}")
    )
    subscriber.register("test-full")
    try:
        agent = nemo_relay.scope.push("outer-agent", nemo_relay.ScopeType.Agent, input={"instruction": "hello"})
        nested = nemo_relay.scope.push("nested-agent", nemo_relay.ScopeType.Agent, input={"text": "hello"})
        asyncio.run(record_calls())
        nemo_relay.scope.pop(nested, output={"answer": "hello back"})
        nemo_relay.scope.pop(agent, output="hello back")
        nemo_relay.subscribers.flush()
        subscriber.force_flush()
    finally:
        subscriber.deregister("test-full")
        subscriber.shutdown()
        server.shutdown()
        thread.join()
        server.server_close()
    if not bodies:
        raise AssertionError("Relay exported no OTLP requests")
    requests = [MessageToDict(ExportTraceServiceRequest.FromString(body)) for body in bodies]
    output.write_text(json.dumps(requests, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="new capture JSON path (use /tmp to inspect before replacing a fixture)"
    )
    capture(parser.parse_args().output)
