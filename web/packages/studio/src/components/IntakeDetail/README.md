<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Intake Detail

Session, trace, and span detail UI: persistent session chrome, trajectory explorer, per-span bodies, kind templates, and shared atoms. List views (traces/spans tables) live in `IntakeLists/`.

Span templates provide specialized view for each known KIND template, with a fallback for unknown kinds. To review the output of these templates, run the seed script at:
services/intake/scripts/spans/seed_span_type_showcase.py.

## Architecture

```mermaid
flowchart TB
  subgraph routes [Routes]
    SR[IntakeSessionDetailRoute]
    ER[EvaluationSessionDetailRoute]
  end

  subgraph session [Session page]
    SDV[SessionDetailView]
    SSH[SessionSummaryHeader]
  end

  subgraph trace [Trace page]
    TDV[TraceDetailView]
    TSA[TraceSpanAccordions]
    STV[SpanTreeView]
    SGV[TraceSpanGraphView]
    SLV[SpanListView]
    TST[TraceDetailSpanTree]
    TAC[TraceSpanAccordionContent]
  end

  subgraph spanBody [Shared span body]
    SMA[SpanMetadataAccordions]
    REG[SpanTemplates/registry]
    SKV[spanKeyValues / traceKeyValues]
    IAP[AnnotationsPanel]
    RJD[RawJsonDebug]
  end

  SR --> SDV
  ER --> SDV
  SDV --> SSH
  SDV --> TDV
  TDV --> TSA
  TSA -->|tree| STV
  TSA -->|graph| SGV
  TSA -->|list| SLV
  STV --> TST
  STV --> TAC
  SGV --> TAC
  SLV --> TAC
  TAC --> SMA
  SMA --> REG
  SMA --> SKV
  SMA --> IAP
  SMA --> RJD
  TDV --> SKV
  TDV --> RJD
```

| Layer                         | Role                                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Routes**                    | Intake and Evaluation routes supply context to the same session detail view.                                        |
| **SessionDetailView**         | Owns session queries, the persistent header/sidebar, URL selection, and the session trajectory model.               |
| **TraceDetailView**           | Hydrates the selected trace and renders its Attributes / Evaluation Context accordions and raw JSON debug.          |
| **TraceSpanAccordions**       | Uses session span summaries and fetches trace annotations; Tree/Graph/List toggle; toolbar; row headers + feedback. |
| **Tree / Graph / List views** | Three trace layouts that share span selection, detail content, annotations, and URL state.                          |
| **TraceSpanAccordionContent** | Lazy `useGetSpan` when a span body is shown; merges list summary with full detail via `mergeSpanDetails`.           |
| **SpanMetadataAccordions**    | Single source of truth for span body inside the trace explorer.                                                     |
| **SpanTemplates/**            | Per-`SpanKind` descriptors + content components; registered in `registry.ts`.                                       |
| **traceSpanShared.ts**        | Note-focus nonces and accordion DOM ids shared by the explorer views.                                               |
| **IntakeComponents/**         | Shared UI: key/value grids, payloads, status badges, feedback controls, `spanKeyValues` / `traceKeyValues`.         |

## Trace layouts

`TraceViewToolbar` keeps the view control in the same slot for session and trace-selected bodies. Graph is available after selecting a trace. All three views share URL selection and span details.

|               | **Tree** (`SpanTreeView`)                               | **Graph** (`TraceSpanGraphView`)                                                                                       | **List** (`SpanListView`)                |
| ------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Layout**    | Trajectory tree + selected span panel                   | Interactive DAG + selected span panel                                                                                  | Flat `IntakeAccordion` of every span     |
| **Selection** | Click tree node to show its span                        | Click graph node to show its span or choose a call from a grouped node                                                 | Expand accordion row inline              |
| **Modes**     | Session and trace hierarchy                             | Grouped combines repeated operations; All spans shows every parent link and can highlight the path with the most spans | Trace spans in start order               |
| **Data**      | First summary page rendered as an always-open hierarchy | Intake span summaries from the selected trace                                                                          | First summary page, hierarchy in trigger |
| **Lazy load** | `useGetSpan` when a span is selected                    | `useGetSpan` when a graph node or grouped call is selected                                                             | `useGetSpan` only when a row is open     |

The graph and span details use a draggable divider. The divider is keyboard accessible with Left, Right, Home and End keys.

The Intake detail route is `/intake/sessions/:sessionId`; the Evaluation detail route is `/experiment/:experimentName/:evaluationName/sessions/:sessionId`. Both accept `traceId` and `spanId` query parameters for progressively deeper links and feed the same `SessionDetailView`. The sidebar loads the session's first summary-only page of up to 1,000 spans, combines each trace with its spans and tree in a `SessionTrajectory`, and renders every available hierarchy open and non-collapsible. Selecting a span fetches its full payload for the detail body. Clicking **Session** clears trace/span selection. Direct trace/span links remain available when the span lies outside the summary page.

Annotations for the whole trace are fetched once (`useListAnnotations` filtered by `session_id`) so each row can show feedback sentiment and annotation counts without per-span queries.

## Span templates

A template is two files plus a registry entry:

1. **`*SpanTemplate.ts`** — descriptor: `sections`, `defaultOpen`, `attributeNamespaces`, optional `headerTitle` / `headerBadge`, optional `customSections`
2. **`*SpanContent.tsx`** — elevated kind body (rendered above accordions when `sections` includes `'kind'`)
3. **`registry.ts`** — `SPAN_TEMPLATES[kind] = …`; unknown kinds fall back to `defaultSpanTemplate`

Registered kinds today: LLM, TOOL, RETRIEVER, EMBEDDING, AGENT, RERANKER, EVALUATOR, GUARDRAIL, CHAIN, UNKNOWN.

### Data sources

| Source                  | Used for                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Typed `Span` fields     | Model, tokens, cost, input/output, status, errors, ids, timestamps                       |
| `raw_attributes` (JSON) | Kind-specific telemetry not promoted to typed fields (e.g. `retrieval.documents.*`)      |
| `useGetSpan`            | Full payload when a span body is shown (merged with list summary via `mergeSpanDetails`) |
| `useListAnnotations`    | Per-span feedback and annotation counts in trace row headers                             |

Templates read `raw_attributes` through `SpanTemplates/rawAttributes.ts` (`parseRawAttribute`, `collectIndexedEntries`, kind-specific extractors). Display helpers live in `templateFields.tsx` (`TemplateKeyValues`, `RankedDocumentList`).

### How `SpanMetadataAccordions` renders

1. **Error banner** — failed spans (`status === error`)
2. **Kind body** — `template.Content` when `sections` includes `'kind'`
3. **Section accordions** — driven by `template.sections` (subset of `kind`, `llm`, `input`, `output`, `metadata`, `annotations`):

   **Annotations leads** the accordion group when present, so reviewers see feedback before payloads. **`customSections`** (retriever query/documents, reranker ranked list) render next, then the remaining generic sections.

| Section            | Accordion label | Body                                                                     |
| ------------------ | --------------- | ------------------------------------------------------------------------ |
| `llm`              | Usage           | Token/cost grid (`buildSpanLlmEntries`, minus model params in kind body) |
| `input` / `output` | Input / Output  | `SpanPayloadView` + `raw`/`md`/`json` toggle on the trigger              |
| `metadata`         | Metadata        | `buildSpanSummaryEntries` via `KeyValueRows`                             |
| `annotations`      | Annotations     | `AnnotationsPanel` (+ count badge on trigger)                            |
| _(custom)_         | _(per kind)_    | `template.customSections(span)` — open by default                        |

4. **Raw JSON debug** — collapsible `RawJsonDebug` dump of the full span object

Expand/collapse-all from the trace toolbar drives section state via `expandToken` / `collapseToken` props (tree view). "Add note" on a row opens the Annotations section and focuses its note field via `focusNoteNonce`.

### Payload formats

Every payload renders through `SpanPayloadView` in one of three formats: `raw` (verbatim text), `md` (rendered markdown), or `json` (pretty-printed and syntax-highlighted). A payload opens in `json` when it parses as JSON and `raw` otherwise, so the common case needs no click.

Input and Output pair the view with `SpanPayloadFormatToggle` on the section trigger. The two share state through `useSpanPayloadFormat`, called in `SpanMetadataAccordions` because the toggle renders in `slotEnd` while the payload renders in `slotContent`. The control hides itself when the span has no payload and disables `json` (with a tooltip) for payloads that are not JSON. Selecting a format on a collapsed section also opens it. The choice is scoped to the payload text it was made for, so selecting a span with a different payload re-derives the default rather than keeping a view that payload cannot satisfy. A span whose payload is byte-identical keeps the selection, since either one renders the same text the same way. The trigger is a `<summary>`, so each button suppresses the row toggle.

Payloads at or above 20,000 characters paint a spinner for one frame before mounting the renderer, and skip Shiki highlighting so the full text always appears. Kind-specific payloads (e.g. the retriever query) use `SpanPayloadView` without a toggle and take the same default.

### Metadata catchall

Metadata is **maintenance-free**: whatever is not shown elsewhere.

`buildSpanSummaryEntries` concatenates:

1. **Catalogued fields** — `SPAN_SUMMARY_DESCRIPTORS` (minus keys already in the row header or error banner)
2. **Unmapped typed fields** — any `Span` property not handled by descriptors, Usage section, or templates
3. **Unclaimed `raw_attributes`** — dotted keys **not** under `template.attributeNamespaces`

Claiming a namespace (e.g. `retrieval` for RETRIEVER) removes those keys from Metadata so the kind body or `customSections` are the single source of truth. New telemetry in `raw_attributes` appears in Metadata automatically until a template claims it.

### Adding a kind

```text
SpanTemplates/MyKindSpanContent.tsx    # elevated UI; optional customSections builder
SpanTemplates/MyKindSpanTemplate.ts    # sections, attributeNamespaces, optional header overrides
registry.ts                            # register under SpanKind
```

Omit sections the kind does not need (e.g. RETRIEVER has no `input`/`output`/`llm` — query and documents live in `customSections` instead). Use `customSections` when kind-specific data deserves its own accordion (see `RetrieverSpanContent.tsx`). Omit `'kind'` only for generic fallback behavior (`DefaultSpanTemplate`).
