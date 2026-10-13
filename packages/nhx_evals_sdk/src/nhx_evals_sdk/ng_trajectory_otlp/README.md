<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ng_trajectory_otlp

Converts NeMo-Gym rollout records into OTLP spans.

Gym writes one JSONL record per rollout: an OpenAI Responses API request/response pair plus the
reward and the indices that join it back to the input dataset. That is a result, not a trace — it
carries no span tree and no timings. This subpackage projects what the record *does* evidence onto
the OTLP span vocabulary: an AGENT span for the rollout, an LLM span for the model call, and a TOOL
span per function call the model made.

```python
from nhx_evals_sdk.ng_trajectory_otlp import rollout_to_resource_spans

# A record is one line of the rollouts JSONL Gym wrote; this is the smallest one that converts.
record = {"response": {"output_text": "Answer."}}
resource_spans = rollout_to_resource_spans(record, rollout_id="task-1:0", task_id="task-1")
```

It returns generated `ResourceSpans` messages.

```python
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

request = ExportTraceServiceRequest(resource_spans=resource_spans)
```

The Gym runtime renders the message to JSON only to store it, because an `EvidenceDescriptor` holds
JSON; `OTLPTraceHandle` parses it straight back.

## Keep it liftable

It lives here because the Gym runtime is its only caller, not because it belongs to this SDK. It
imports nothing from the rest of `nhx_evals_sdk` and depends only on `opentelemetry-proto`, so
it can be extracted into its own distribution the day a producer outside the evaluator wants it. Its
tests import only this subpackage, which is what keeps that honest. Adding an SDK import here would
quietly take it away.

`rollout_id` is load-bearing. Span ids are derived from it so that converting a record twice yields
the same ids and a re-publish replaces spans rather than duplicating them — which means two rollouts
sharing an id collide, and a consumer keyed on span identity keeps only the last. A task id is not
enough: one task can be attempted repeatedly, and Gym does not always record which attempt a record
belongs to.

## What it does not do

* **Invent timings.** Gym records none, so spans carry no start time. A consumer that needs one
  supplies it (the evaluator's publish path fills in the run's start time).
* **Reconstruct concurrency.** One rollout is one linear exchange as recorded; parallel tool calls
  are not distinguishable in the source.
* **Report usage Gym did not state.** Absent token counts stay absent rather than becoming zero.
