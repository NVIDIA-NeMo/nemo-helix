<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Intake

## EvaluatorResults

Types:

```python
from nemo_platform.types.intake import (
    EvaluatorResult,
    EvaluatorResultDataType,
    EvaluatorResultFilter,
    EvaluatorResultParam,
    EvaluatorResultSortField,
    EvaluatorResultsPage,
    FloatFilter,
)
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/evaluator-results">client.intake.evaluator_results.<a href="./src/nemo_platform/resources/intake/evaluator_results.py">create</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/evaluator_result_create_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/evaluator_result.py">EvaluatorResult</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/evaluator-results/{evaluator_result_id}">client.intake.evaluator_results.<a href="./src/nemo_platform/resources/intake/evaluator_results.py">retrieve</a>(evaluator_result_id, \*, workspace) -> <a href="./src/nemo_platform/types/intake/evaluator_result.py">EvaluatorResult</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/evaluator-results">client.intake.evaluator_results.<a href="./src/nemo_platform/resources/intake/evaluator_results.py">list</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/evaluator_result_list_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/evaluator_result.py">SyncDefaultPagination[EvaluatorResult]</a></code>

## Annotations

Types:

```python
from nemo_platform.types.intake import (
    Annotation,
    AnnotationFilter,
    AnnotationKind,
    AnnotationParam,
    AnnotationSortField,
    AnnotationsPage,
    FeedbackAnnotation,
    FeedbackAnnotationParam,
    LabelAnnotation,
    LabelAnnotationParam,
    MetadataAnnotation,
    MetadataAnnotationParam,
    NoteAnnotation,
    NoteAnnotationParam,
    NumericFilter,
)
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/annotations">client.intake.annotations.<a href="./src/nemo_platform/resources/intake/annotations.py">create</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/annotation_create_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/annotation.py">Annotation</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}">client.intake.annotations.<a href="./src/nemo_platform/resources/intake/annotations.py">retrieve</a>(annotation_id, \*, workspace) -> <a href="./src/nemo_platform/types/intake/annotation.py">Annotation</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/annotations">client.intake.annotations.<a href="./src/nemo_platform/resources/intake/annotations.py">list</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/annotation_list_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/annotation.py">SyncDefaultPagination[Annotation]</a></code>
- <code title="delete /apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}">client.intake.annotations.<a href="./src/nemo_platform/resources/intake/annotations.py">delete</a>(annotation_id, \*, workspace) -> None</code>

## Ingest

Types:

```python
from nemo_platform.types.intake import EvaluationContext
```

### Atif

Types:

```python
from nemo_platform.types.intake.ingest import (
    AtifAgent,
    AtifContentPart,
    AtifContentPartImage,
    AtifContentPartText,
    AtifFinalMetrics,
    AtifImageSource,
    AtifIngestRequest,
    AtifMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifStepAgent,
    AtifStepSystem,
    AtifStepUser,
    AtifSubagentTrajectoryRef,
    AtifToolCall,
    AtifTrajectory,
)
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/ingest/atif">client.intake.ingest.atif.<a href="./src/nemo_platform/resources/intake/ingest/atif.py">create</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/ingest/atif_create_params.py">params</a>) -> None</code>

### ChatCompletions

Types:

```python
from nemo_platform.types.intake.ingest import (
    CapturedChatCompletionsRequest,
    CapturedChatCompletionsResponse,
    CapturedChatMessage,
    ChatCompletionsIngestParam,
    ChatCompletionsIngestResponse,
    ChatMessageRole,
)
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/ingest/chat-completions">client.intake.ingest.chat_completions.<a href="./src/nemo_platform/resources/intake/ingest/chat_completions.py">create</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/ingest/chat_completion_create_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/ingest/chat_completions_ingest_response.py">ChatCompletionsIngestResponse</a></code>

### Spans

Types:

```python
from nemo_platform.types.intake.ingest import DirectSpanInput, DirectSpansIngestParam, JsonValue
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/ingest/spans">client.intake.ingest.spans.<a href="./src/nemo_platform/resources/intake/ingest/spans.py">create</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/ingest/span_create_params.py">params</a>) -> None</code>

### Otlp

#### V1

##### Traces

Types:

```python
from nemo_platform.types.intake.ingest.otlp.v1 import IngestResponse
```

Methods:

- <code title="post /apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces">client.intake.ingest.otlp.v1.traces.<a href="./src/nemo_platform/resources/intake/ingest/otlp/v1/traces.py">create</a>(body, \*, workspace, \*\*<a href="src/nemo_platform/types/intake/ingest/otlp/v1/trace_create_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/ingest/otlp/v1/ingest_response.py">IngestResponse</a></code>

## Spans

Types:

```python
from nemo_platform.types.intake import (
    Span,
    SpanEvaluationContext,
    SpanFilter,
    SpanKind,
    SpanSortField,
    SpanStatus,
    SpansPage,
)
```

Methods:

- <code title="get /apis/intake/v2/workspaces/{workspace}/spans/{span_id}">client.intake.spans.<a href="./src/nemo_platform/resources/intake/spans/spans.py">retrieve</a>(span_id, \*, workspace) -> <a href="./src/nemo_platform/types/intake/span.py">Span</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/spans">client.intake.spans.<a href="./src/nemo_platform/resources/intake/spans/spans.py">list</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/span_list_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/span.py">SyncDefaultPagination[Span]</a></code>

### Groups

Types:

```python
from nemo_platform.types.intake.spans import (
    SpanGroup,
    SpanGroupBy,
    SpanGroupSortField,
    SpanGroupsPage,
)
```

Methods:

- <code title="get /apis/intake/v2/workspaces/{workspace}/spans/groups">client.intake.spans.groups.<a href="./src/nemo_platform/resources/intake/spans/groups.py">list</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/spans/group_list_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/spans/span_group.py">SyncDefaultPagination[SpanGroup]</a></code>

### EvaluatorResults

Types:

```python
from nemo_platform.types.intake.spans import EvaluatorResultListResponse
```

Methods:

- <code title="get /apis/intake/v2/workspaces/{workspace}/spans/{span_id}/evaluator-results">client.intake.spans.evaluator_results.<a href="./src/nemo_platform/resources/intake/spans/evaluator_results.py">list</a>(span_id, \*, workspace) -> <a href="./src/nemo_platform/types/intake/spans/evaluator_result_list_response.py">EvaluatorResultListResponse</a></code>

## Sessions

Types:

```python
from nemo_platform.types.intake import Session
```

Methods:

- <code title="get /apis/intake/v2/workspaces/{workspace}/sessions/{id}">client.intake.sessions.<a href="./src/nemo_platform/resources/intake/sessions.py">retrieve</a>(id, \*, workspace) -> <a href="./src/nemo_platform/types/intake/session.py">Session</a></code>

## Traces

Types:

```python
from nemo_platform.types.intake import (
    CostRollupResponse,
    LatencyRollupResponse,
    TokenRollupResponse,
    Trace,
    TraceFilter,
    TraceMetricBucketParam,
    TraceMetricPointResponse,
    TraceMetrics,
    TraceSortField,
    TracesPage,
)
```

Methods:

- <code title="get /apis/intake/v2/workspaces/{workspace}/traces/{id}">client.intake.traces.<a href="./src/nemo_platform/resources/intake/traces.py">retrieve</a>(id, \*, workspace, \*\*<a href="src/nemo_platform/types/intake/trace_retrieve_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/trace.py">Trace</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/traces">client.intake.traces.<a href="./src/nemo_platform/resources/intake/traces.py">list</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/trace_list_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/trace.py">SyncDefaultPagination[Trace]</a></code>
- <code title="get /apis/intake/v2/workspaces/{workspace}/traces/metrics">client.intake.traces.<a href="./src/nemo_platform/resources/intake/traces.py">get_metrics</a>(\*, workspace, \*\*<a href="src/nemo_platform/types/intake/trace_get_metrics_params.py">params</a>) -> <a href="./src/nemo_platform/types/intake/trace_metrics.py">TraceMetrics</a></code>
