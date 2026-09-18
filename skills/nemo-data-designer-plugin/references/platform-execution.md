# Running Data Designer on NeMo Platform

Data Designer runs as a NeMo Platform service. Inference, seed data, and persona data are all resolved by the platform, not from local configuration on your machine. The CLI surface is `nemo data-designer …`.

## Model configs

Every config needs at least one model alias, and aliases must be declared **programmatically in the script**. `DataDesignerConfigBuilder` accepts them via its constructor or `.add_model_config(...)`:

```python
import data_designer.config as dd

def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="text",
                model="...",
                provider="default/nvidia-build",
                inference_parameters=dd.ChatCompletionInferenceParams(),
            ),
        ],
    )
    ...
```

Pick the right `inference_parameters` class for the generation type: `ChatCompletionInferenceParams`, `EmbeddingInferenceParams`, or `ImageInferenceParams`. The class determines the alias's `generation_type` and which column types can use it.

**`provider` is an Inference Gateway provider** — either a bare name (resolved in the active workspace) or `<workspace>/<provider>`. Discover what's registered with:

```bash
nemo inference providers list
```

A common default created during `nemo setup` is `default/nvidia-build`, but it's optional — confirm before relying on it. If the user names a provider (e.g., "use my-vllm"), trust the name; `validate` will surface a clear error if it isn't reachable.

A provider resolving does **not** mean the `model` you paired with it is servable. `validate` checks the provider, not the model name. Use `check-models` (below) to confirm the model itself.

Set `model` to the `served_model_name` as understood by Inference Gateway, not the `model_entity_id`.

The **Model Aliases** table in `nemo data-designer agent context` output reflects a local library registry that platform execution ignores. Whatever it shows — including an empty list — is not a signal about what your config can use. Do not stop on it, and do not copy aliases out of it.

## Validation

`nemo data-designer validate <path>` compiles the config and checks it against the platform: Inference Gateway provider resolution, Files-service seed sources, Nemotron Personas filesets, and the supported seed-source list.

```bash
nemo data-designer validate <path>
```

```text
  ✘ Seed source 'df' is not supported on the NeMo Platform.
    Use a serializable seed source such as a HuggingFace dataset
    or the Files service.
```

A single invocation surfaces **every** problem it can detect (it doesn't short-circuit on the first failure). Exit code is 0 only when no errors are reported.

**What validate does not cover:** whether the models the config references actually respond. It resolves the provider and confirms the model is enabled on it, but never contacts the model. A green `validate` is therefore not a promise that `preview` will run — see Model health checks below.

Flags:

- `--workspace <name>` — workspace used to resolve Inference Gateway providers and Files-service seed sources. Defaults to the workspace of the active CLI context (`nemo config current-context`), or `default`.
- `--output {text,json}` — `json` emits a structured `ValidationReport` for CI / scripting use.

## Model health checks

`nemo data-designer check-models <path>` probes every model the config references, without running a workload. It sends a tiny generation request to each model alias through Inference Gateway and reports whether it came back.

```bash
nemo data-designer check-models <path>
```

```text
  👀 Checking 'nvidia/nemotron-3.5-lightning-30b-a3b' in provider named 'default/nvidia-build' for model alias 'text'...
  ✅ Passed!

  ✔ All models responded successfully
```

This is the companion to `validate`, and the split mirrors the upstream library:

- `validate` — is the config well-formed, and do the resources it names resolve? No inference.
- `check-models` — do those models actually respond? One small generation per model alias.

Only a live request can tell you a model works. A provider's advertised model list is not reliable — providers commonly expose models in `/v1/models` that fail in practice — so this is the only way to catch a bad model name before a preview does.

Run it once after `validate` passes and before the first `preview`, then again only when a model or provider changes. Each run costs real inference, so it is not part of the edit loop the way `validate` is.

Unlike `validate`, it stops at the first model that fails rather than listing every problem. The per-model log lines identify which alias failed. Models configured with `skip_health_check=True` are skipped.

Flags:

- `--workspace <name>` — same meaning as for `validate`.
- `--output {text,json}` — `json` emits a structured `CheckModelsReport` (each error carries an `error_type` such as `ModelNotFoundError`) and suppresses the per-model log lines.

## Seed data

Seed data comes from HuggingFace or the NeMo Platform Files service. Local files and in-memory dataframes are rejected. See `references/seed-datasets.md`.

## Person data

Nemotron Personas locales are read from filesets in the `system` workspace, created once per locale with `nemo data-designer personas make-fileset`. Nothing is downloaded locally. See `references/person-sampling.md`.

## Artifacts

`nemo data-designer create` stores output at a Jobs-service-managed path, so there is no local artifact folder to name or relocate.

## Related NeMo Platform commands

- `nemo inference providers list` / `nemo models list` — inference providers and models available to `ModelConfig.provider`.
- `nemo files` — manage filesets, including seed data and persona filesets.
- `nemo secrets` — manage API keys referenced by `personas make-fileset` and private HuggingFace seed sources.
- `nemo data-designer retrieval-generate` / `retrieval-prepare` — Nemotron retrieval SDG Stage 0/1. See `references/retrieval-sdg.md`.
