<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Insight suite authoring

> **WARNING:** Use Experimentalist only with a trusted repository.
> An agent import executes module top-level code.

Experimentalist turns an Insight and its production trace references into
evaluator dataset changes before optimization begins. The
`nemo_experimentalist_plugin.eval_author` package implements this step using the
plugin's evaluator, staging, and trace helpers.

## Current Files

- `agent.py` defines the canonical `EvalAuthor` agent.
- `materialization.py` stages, validates, and persists Insight suites locally.
- `models.py` defines the lightweight `EvalAuthorConfig` and `EvalAuthorResult` models.
- `REFERENCE.md` documents the Python return contract.
- `run.py` defines `run_eval_author(...)`, a reusable orchestration function for
  Python callers.
- `config.yaml` is a default run preset for future CLI or job wiring.

## Configuration

`config.yaml` is not loaded automatically yet. A future entrypoint should read
the file, overlay caller-provided values, and validate the nested `eval_author`
block with `EvalAuthorConfig`.

The preset includes the run inputs needed by `run_eval_author(...)`:

- `insight`: local Insight file path or platform Insight id.
- `agent`: optional agent source override. If omitted, the Insight's agent is used.
- `train_dataset`: evaluator training dataset URI.
- `validation_dataset`: evaluator validation dataset URI.
- `task_template`: local or `fileset://` evaluator task template URI for production traces.
- `experiment_dir`: local suite-authoring working directory.
- `workspace`, `base_url`, `mode`, and `evaluator_type`: platform and evaluator routing.
- `eval_author.max_summary_tokens` and `eval_author.max_traces`: agent tuning parameters.

## Materialized Insight Suite

For an Insight with trace references, Experimentalist copies and fills one local Harbor
task template per trace beneath its experiment working directory:

```text
eval-and-optimize/eval_author/<insight-slug>/insight-suite/
```

Each authoring invocation fills a fresh candidate suite from the current template
and traces. The complete suite is Harbor-validated locally, promoted to the
experiment-local working copy with backup-and-restore failure handling, and
analyzed for the Insight's root cause. Experimentalist then adds normalized
Insight-specific verifier metric keys to every task in the staged train,
validation, and generated Insight datasets while preserving existing task
metrics. All three datasets must pass static Harbor validation before they are
returned.

After authoring and validation, Experimentalist hashes every task file and verifier
file and persists deterministic suite and scorer identities in the local suite's
manifest. The returned Insight dataset points at the experiment-local suite.
The authoring step does not split, merge, or evaluate that suite. Experimentalist
currently leaves it unconsumed for downstream integration by its owning team.

Task-template inputs may be local paths, `file://` URIs, or NeMo Helix
`fileset://<workspace>/<fileset>` references. Fileset-backed templates are
downloaded into the experiment-local staging directory before Harbor parses
them. The staged template is refreshed on every invocation rather than reused.

The returned Python contract is documented in the
[Suite authoring Python reference](REFERENCE.md#evalauthorresult).

## Intended Python Invocation

`run_eval_author(...)` remains available to Python callers:

```python
import asyncio
from pathlib import Path

from nemo_experimentalist_plugin.eval_author.models import EvalAuthorConfig
from nemo_experimentalist_plugin.eval_author.run import run_eval_author
from nemo_experimentalist_plugin.entities import DatasetRef

async def main() -> None:
    result = await run_eval_author(
        insight="insight-id",
        train_dataset=DatasetRef(uri="path/to/train"),
        validation_dataset=DatasetRef(uri="path/to/validation"),
        task_template=DatasetRef(uri="path/to/template"),
        experiment_dir=Path("tmp/eval_author"),
        workspace="default",
        base_url="http://localhost:8080",
        config=EvalAuthorConfig(),
    )
    print(result.summary)

asyncio.run(main())
```

## Agent models

Run `nemo setup` and select the default and fast models for the active Platform
context. Experimentalist uses the default model for authoring and the fast model for
summarization. Press Enter at the fast-model prompt to reuse the default model.

The selections are workspace-qualified Platform Model Entity IDs. The Platform
routes each request to the provider registered for that entity and reads its
credential from Platform Secrets; Experimentalist does not accept separate provider,
endpoint, key, or model environment variables.

For non-interactive and isolated environments, `NEMO_DEFAULT_MODEL` and
`NEMO_FAST_MODEL` can override the stored selections. Values must still use
`workspace/model-name` and refer to Model Entities on the target Platform.
