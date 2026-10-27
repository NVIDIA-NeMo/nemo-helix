<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Building each format

Hand-built and converted packages, one section per format. Read `gym-config.md` first; these sections assume its vocabulary.

## Path A — an environment already in Gym (`native-v1`)

Use when the user names an environment that ships with NeMo Gym (`resources_servers/<name>/`, `responses_api_agents/<name>/`) or has written one in that layout. The package is a **slice of the Gym tree** with its directory structure preserved.

`native-v1` ships no `wheels/`, so its per-server venv resolves from a package index at spin-up and **the job needs egress**. That is the only thing the format choice decides — it does *not* mean the server's imports must already be in the image, since Gym installs them at first use either way. On a deny-default cluster, ship the same tree as `wheels-v1` instead (Path C).

Drop `data/`, `tests/` and the `datasets` block (SKILL.md rule 5). Gym's own configs point `datasets[].jsonl_fpath` at an in-tree file; here the dataset FileSet supplies the rows (`rollout-rows.md`).

### Worked example — a tool-using weather environment

This is Gym's own `resources_servers/example_single_tool_call` restated as a package. It exercises every concept in `gym-config.md`: a custom resources server, an image agent referenced by config only, and the `policy_model` server.

```text
weather-env/
  nemo-environment.yaml
  resources_servers/weather_verifier/
    app.py                                     ← your code
    requirements.txt                           ← install marker; REQUIRED
    configs/weather_verifier.yaml
  responses_api_agents/simple_agent/
    configs/weather_agent.yaml                 ← config only; app.py comes from the image
  responses_api_models/vllm_model/
    configs/policy_model.yaml
```

```yaml
# nemo-environment.yaml
format: native-v1
config_paths:                                  # policy_model first — the others ref it
  - responses_api_models/vllm_model/configs/policy_model.yaml
  - resources_servers/weather_verifier/configs/weather_verifier.yaml
  - responses_api_agents/simple_agent/configs/weather_agent.yaml
metadata:
  name: weather-grpo
  description: Single tool call, reward on correct use of get_weather
```

```yaml
# resources_servers/weather_verifier/configs/weather_verifier.yaml
weather_verifier:                              # instance
  resources_servers:
    weather_verifier:                          # implementation dir (yours)
      entrypoint: app.py
      domain: agent                            # required, closed set
      description: Weather tool + verify
```

```yaml
# responses_api_agents/simple_agent/configs/weather_agent.yaml
weather_simple_agent:                          # instance — dataset rows name THIS
  responses_api_agents:
    simple_agent:                              # implementation from the image
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: weather_verifier                 # the instance above
      model_server:
        type: responses_api_models
        name: policy_model
```

```python
# resources_servers/weather_verifier/app.py
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class WeatherResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetWeatherRequest(BaseModel):
    city: str


class GetWeatherResponse(BaseModel):
    city: str
    weather_description: str


class WeatherResourcesServer(SimpleResourcesServer):
    config: WeatherResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_weather")(self.get_weather)   # one route per tool
        return app

    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        return GetWeatherResponse(city=body.city, weather_description=f"The weather in {body.city} is cold.")

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # The reward. Inspect body's rollout and score it; 1.0 here is a placeholder.
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    WeatherResourcesServer.run_webserver()
```

```text
# resources_servers/weather_verifier/requirements.txt
# nemo-gym is prepended automatically; list anything else app.py imports
```

A matching dataset row — note `agent_ref.name` is the **instance**, and that `simple_agent` learns the tool from the row, not from the server:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "what's it like in sf?"}],
    "tools": [{"type": "function", "name": "get_weather", "description": "",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                              "required": ["city"], "additionalProperties": false},
               "strict": true}]
  },
  "agent_ref": {"type": "responses_api_agents", "name": "weather_simple_agent"}
}
```

Gym's in-tree rows omit `agent_ref` because the owning agent config supplies it. **The platform path requires it on every row.** See `rollout-rows.md`.

## Path B — a Prime Intellect / verifiers env (`adapter-wheels-v1`)

The automated path, and the only one with a converter. It downloads the hub package, vendors its full wheel closure, writes both configs and the manifest, and snapshots the hub environment's Hugging Face dataset into the dataset FileSet so training does not call the Hub.

That snapshot is two artifacts: `hub_environment.parquet`, with `vf_env_args.dataset_path` set to `/job/dataset/hub_environment.parquet`, and `.huggingface/`, the cache populated while converting. Loaders that honor `dataset_path` read the parquet; loaders that still call `load_dataset("org/name")` resolve from the cache, offline.

Run it with the converter from SKILL.md § **Before you start**, on a host with internet.

```bash
pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree \
  --hub-version 0.1.5 \
  --nemo-rl-root ~/workspace/RL \
  --out-dir ./ascii-tree-pkg \
  --dataset-dir ./ascii-tree-data \
  --validation-fraction 0.1
```

`--validate-only` needs neither the extra nor hub access.

| Flag | Use it for |
|---|---|
| `--hub-id` | Hub slug. Required for conversion. |
| `--hub-version` | The hub release to convert. SKILL.md step 2 says why to pin it. |
| `--out-dir` | Environment package output. |
| `--dataset-dir` | Prompt JSONL output. Defaults to a sibling of `--out-dir`. |
| `--vf-env-id` | Override the verifiers load id. Defaults to the hub slug's last segment. |
| `--vf-env-args` | JSON object forwarded to the environment loader. Must be a JSON **object**. |
| `--dataset-size` | Cap rows (`-1` = all). |
| `--validation-fraction` | Split fraction in `[0, 1)`. `0` writes training rows only. |
| `--wheels-dir` | Use pre-vendored wheels instead of downloading. |
| `--upload` | Create both FileSets and upload, in one step. Needs `NHX_BASE_URL`. |
| `--validate-only <dir>` | Check an existing package and exit. Never touches the hub. |

What it writes:

```text
ascii-tree-pkg/
  nemo-environment.yaml
  configs/policy_model.yaml       ← points Gym at the job's own vLLM engine
  configs/verifiers_agent.yaml    ← vf_env_id, vf_env_args, max_tokens, temperature
  wheels/*.whl                    ← resolved closure
ascii-tree-data/
  training.jsonl
  validation.jsonl                ← only with --validation-fraction > 0
  hub_environment.parquet         ← `vf_env_args.dataset_path` → `/job/dataset/hub_environment.parquet`
  .huggingface/                   ← HF cache from conversion; Gym host copies to `/job/work/.huggingface`
```

**`pi-to-gym-conversion` supports `x86_64` only today.** It also rejects `aarch64` wheels passed with `--wheels-dir`. On an `arm64` cluster, hand-build a `wheels-v1` package (Path C) with an `aarch64` closure (`references/dependencies.md` § **Vendoring a wheel closure**).

**`adapter.agent` must be on the image allowlist.** The package ships config and wheels; the agent harness itself comes from the training image. Today the allowlist is exactly one entry — `verifiers_agent` (`IMAGE_ADAPTER_ALLOWLIST`). Anything else is rejected at validation, so a user who wants a different harness needs `wheels-v1` or `native-v1`.

**`max_tokens` lives in the package, not the job.** `configs/verifiers_agent.yaml` carries `max_tokens` (default 8192), and `verifiers_agent` reads it in preference to the job's `max_new_tokens`. To bound response length for an adapter env, edit this file before uploading.

## Path C — vendored code (`wheels-v1`)

The catch-all, and the right answer more often than its "bring your own code" framing suggests. Use it for a custom verifier or bespoke agent — **and for a Gym source-tree server whose dependencies have to travel with it**, which is the common case on a cluster without egress.

`wheels-v1` is the least constrained format — `config_paths` may sit anywhere — and correspondingly the one with the least done for you. There is no converter; build it by hand.

Gym runs `{server_type}/{implementation}/` and requires `requirements.txt` or `pyproject.toml` there. A `configs/` + `wheels/` package with **no** server directories only works when every YAML implementation is already in the image (`simple_agent`, `vllm_model`, `verifiers_agent`). A custom `resources_servers/my_env` **must** ship that directory, even if the YAML itself lives under `configs/`.

```text
my-env/
  nemo-environment.yaml
  configs/policy_model.yaml     ← required; see gym-config.md
  configs/my_env.yaml
  resources_servers/my_env/     ← required for a custom implementation
    app.py
    requirements.txt
  wheels/*.whl                  ← the closure
```

```yaml
format: wheels-v1
config_paths:
  - configs/policy_model.yaml
  - configs/my_env.yaml
metadata:
  name: my-env
  description: Custom reward environment
```

`wheels-v1` declares **no adapter**, so its code may be a verifier rather than an agent, and both `responses_api_agents` and `resources_servers` are started from it. Your config decides which servers exist.

### Packaging a Gym server tree as `wheels-v1`

This is the common `wheels-v1` shape — a Gym source slice, not a repackaged Python distribution:

```text
math-with-judge/
  nemo-environment.yaml
  configs/policy_model.yaml
  resources_servers/math_with_judge/
    app.py  client.py  requirements.txt  configs/  prompts/
  wheels/*.whl
```

Drop `data/` and `tests/` (SKILL.md rule 5), add `configs/policy_model.yaml` and the manifest at the root, and vendor `wheels/`. `config_paths` then lists `configs/policy_model.yaml` and the server's own in-tree config — `wheels-v1` allows either location.

Which `nemo_gym` wheel to vendor: `dependencies.md` § **What goes in a server's `requirements.txt`**.
