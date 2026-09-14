# Platform OpenAI Spec

## Generation

To generate the `openapi.yaml` files, the generator imports each FastAPI app in an isolated subprocess and calls its `.openapi()` method directly (no running server), then applies a series of schema fixes and validation passes.

To generate the updated OpenAPI schema, run:

```bash
uv run --frozen python -m script.generate_openapi_spec
```

## Output layout

The generator no longer emits one spec per microservice and merges them. It now produces:

**One aggregate platform spec**, built from the platform runner (`nmp.platform_runner.server:create_platform_openapi_app`) with plugin services deliberately excluded (`NEMO_PLUGIN_SERVICES_ALLOWLIST=""` — see `SERVICES` in `script/generate_openapi_spec.py`). This aggregate covers the core platform services (entities, jobs, models, inference gateway, secrets, files, platform-common, etc.) and lands in:

| File | Contents |
|------|----------|
| `openapi/openapi.yaml` | Final merged GA + EA platform spec |
| `openapi/ga/openapi.yaml` | GA-only platform spec |
| `openapi/ea/openapi.yaml` | EA-only platform spec |
| `openapi/ga/individual/platform.openapi.yaml` | The platform spec before GA/EA merge |

**One spec per opted-in plugin**, written next to each plugin — never merged into the platform spec. A plugin opts in by declaring a `[tool.nemo.openapi]` table in its own `pyproject.toml`; `discover_plugins()` (`script/openapi_helper/plugin_config.py`) enumerates those, builds each plugin's FastAPI app via the convention loader (or a `factory_override`), and emits:

| Plugin | Output File |
|--------|-------------|
| Agents | `plugins/nemo-agents/openapi/openapi.yaml` |
| Auditor | `plugins/nemo-auditor/openapi/openapi.yaml` |
| Customization | `plugins/nemo-customizer/openapi/openapi.yaml` |
| Data Designer | `plugins/nemo-data-designer/openapi/openapi.yaml` |
| Deployments | `plugins/nemo-deployments/openapi/openapi.yaml` |
| Evaluator | `plugins/nemo-evaluator/openapi/openapi.yaml` |
| Agent Hardener | `plugins/nemo-agent-hardener/openapi/openapi.yaml` |
| Safe Synthesizer | `plugins/nemo-safe-synthesizer/openapi/openapi.yaml` |

The Customization spec is assembled at generation time from whichever customization contributors (`nemo.customization.contributors` entry points — e.g. `automodel`, `rl`, `unsloth`) are installed in the workspace, so its route surface depends on the synced environment. To add a new plugin to this list, add an (empty is fine) `[tool.nemo.openapi]` table to its `pyproject.toml`; if the plugin has more than one `nemo.services` entry point, set `service_name` in that table to disambiguate.

The platform GA and EA specs are merged with the `--keep-versions` flag to preserve version information in the final `openapi.yaml`.

### Conflicts

Currently, there are some conflicts between the schemas coming from different microservices, when they are overlapping types.
Some are bugs, some need to be reconciled (one of the schemas is out of date).
This will be fixed in the next releases.
For now, when that happens, the schemas are duplicated, with a suffix.

### Recursive Schemas

There are some recursive schemas that cause issues with the OpenAPI validator.
These are currently fixed by removing the recursive reference and replacing it with a generic object.

## Examples Integration

The OpenAPI specification includes example requests and responses for various endpoints. These examples are generated from the `/openapi/api-examples` files.

As a POC, the included files have been generated with an LLM(Cursor) to analyze and extract examples from the tool calling notebooks.

Currently, the examples are manually generated from notebooks. A robust system that automatically generates examples from all notebooks will be integrated in the future to ensure comprehensive and up-to-date API documentation.

## NOTE

The aggregate platform spec is now generated directly from the platform runner (`nmp.platform_runner.server`), so it corresponds to what is actually mounted at runtime. Only the GA and EA variants of that single platform spec are merged; individual core services are no longer emitted and merged separately.
