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

Flags:

- `--workspace <name>` — workspace used to resolve Inference Gateway providers and Files-service seed sources. Defaults to the SDK's configured workspace, or `default`.
- `--output {text,json}` — `json` emits a structured `ValidationReport` for CI / scripting use.

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
