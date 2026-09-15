<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- NeMo Gym environment packaging for GRPO. Job JSON: `hyperparameters-rl.md`. Prompt rows: `dataset-formats.md` § NeMo-RL (GRPO). -->

# NeMo Gym environments for GRPO

GRPO scores sampled responses against an **environment** — code that runs a rollout and returns a reward. The platform takes that environment as a **FileSet**, and the prompts as a **separate** FileSet. This file covers what a valid environment package looks like in all three formats, and how to get from the user's source to one of them.

**Two FileSets, never one.** The environment package holds code and config; the dataset holds prompt rows. Validation **rejects any `.jsonl` inside the environment package** (`validate_manifest_against_listing`), so a Gym config that points at an in-tree `jsonl_fpath` — which is how environments are shipped in the Gym source tree — will not work here unchanged. Prompts come from the dataset FileSet.

```text
environment FileSet (purpose=environment)   dataset FileSet (purpose=dataset)
  nemo-environment.yaml   ← manifest          training.jsonl    ← required
  configs/ or server dirs                     validation.jsonl  ← optional
  wheels/                 ← 2 of 3 formats
```

Job JSON then references both as strings:

```json
{ "environment": "default/<env-fileset>", "dataset": "default/<dataset-fileset>" }
```

## Before you start

Confirm all of these before building a package — each one invalidates the work if it turns out false:

| Check | How | If it fails |
|---|---|---|
| Platform dispatches to Kubernetes | `nemo jobs list-execution-profiles -f json` shows `backend: kubernetes_job` | Stop — `rl-kubernetes-runtime.md` |
| Cluster runs sandboxed Gym | Operator confirms `sandbox_cluster_capable` + job-storage PVC claim | Stop — fails at submit, no package change helps |
| Whether the cluster has egress | Operator confirms `NMP_RL_SANDBOX_ALLOW_INTERNET` | Decides `native-v1` vs `wheels-v1` — settle **before** layout |
| Which Gym agent the environment runs | `verifiers` env → `verifiers_agent`; a resources server → usually `simple_agent` | Decides the format and every dataset row's `agent_ref.name` |
| Training-image versions | `nemo-gym`, `ray`, `openai` as the Gym actor venv reports them | Needed for a wheels closure that installs offline |

## Pick a format

Source of truth: `services/rl/src/nmp/rl/schemas/environment.py`.

| The user has… | Format | Ships wheels? | `config_paths` must live under |
|---|---|---|---|
| A Gym server tree whose deps resolve at spin-up (cluster has egress, or they are prefetched in the image) | **`native-v1`** | No | `responses_api_agents/`, `resources_servers/`, `responses_api_models/` |
| A Prime Intellect hub env, or any `verifiers` environment installable as a wheel | **`adapter-wheels-v1`** | **Yes** | `configs/` |
| Anything else — including a Gym server tree whose deps must be vendored | **`wheels-v1`** | **Yes** | anywhere in the package |

Decision shortcut: **is there a `verifiers` environment?** → `adapter-wheels-v1` (and use the converter, below). **Otherwise** → `wheels-v1`, unless the cluster can resolve the server's requirements from an index at spin-up, in which case `native-v1` saves you vendoring.

**All three formats run on the same runtime.** The training image carries the NeMo Gym source tree and starts a package's servers the same way it starts Gym's own — same `RunHelper.start`, same per-server venv build, same config merge. Do **not** rank the formats by runtime support or call one "the sandboxed path". What differs is only **where dependencies come from** (§ **Dependency installation**) and **where `config_paths` may live**. Settle dependencies before layout.

`adapter-wheels-v1` is the only format with a bundled converter, and `verifiers_agent` is the only entry on `IMAGE_ADAPTER_ALLOWLIST`. That is the shortest path for a `verifiers` env — not a more supported one.

## How a Gym config YAML is read

Everything else in this file assumes this vocabulary. Source of truth: `nemo_gym/config_types.py`, `nemo_gym/cli/env.py` (`RunHelper.start`, `_resolve_server_dir`).

```yaml
weather_simple_agent:          # 1. INSTANCE — unique at runtime; what refs and dataset rows name
  responses_api_agents:        # 2. SERVER TYPE — one of three, exactly one key here
    simple_agent:              # 3. IMPLEMENTATION — the directory Gym runs: {server_type}/{implementation}/
      entrypoint: app.py       #    run as `python app.py` with cwd = that directory
      resources_server:        # 4. REF — {type, name}; `name` is an INSTANCE, never an implementation
        type: resources_servers
        name: weather_verifier
      model_server:
        type: responses_api_models
        name: policy_model
```

- **Instance ≠ implementation.** The top-level key is the instance; the key one level under the server type is the implementation directory. They are often equal in Gym's own configs (`math_with_judge` / `math_with_judge`), which is exactly why the distinction is easy to miss. Gym's `example_single_tool_call.yaml` is the counter-example: instance `example_single_tool_call_simple_agent`, implementation `simple_agent`.
- **Refs resolve against instances.** `{type, name}` and a dataset row's `agent_ref.name` both name an *instance*. Naming the implementation folder produces `ServerRefNotFoundError`.
- **One implementation per block.** Each server-type mapping is `min_length=1, max_length=1`.

| Server type | Role | Notes |
|---|---|---|
| `resources_servers` | Tools, state, and the `verify()` that returns the reward | **`domain` is required** — see below |
| `responses_api_agents` | The rollout loop (`POST /v1/responses`) | `simple_agent`, `verifiers_agent` ship in the image |
| `responses_api_models` | Proxy to the job's own vLLM engine | Name the instance `policy_model`; `vllm_model` ships in the image |

### `domain` is a closed set, and getting it wrong aborts the run

`domain` is required on every `resources_servers` block and validated against `Domain` in `nemo_gym/config_types.py`:

```text
math  coding  agent  knowledge  instruction_following  long_context
safety  games  translation  e2e  rlhf  other
```

Pick the most specific fit; `other` is the catch-all. An empty string, a missing value, or a typo fails enum validation, and because the block still *looks* like a server (it has an `entrypoint`), Gym classifies it as an **almost-server**: it prints a warning banner and then raises `AlmostServerError`, aborting spin-up. `error_on_almost_servers` defaults to **true** and the platform does not turn it off, so this is a hard failure, not a silently skipped server. Model and agent blocks must **not** carry `domain` — Gym strips it.

### How Gym finds the implementation directory

`_resolve_server_dir` searches, in order, `NEMO_GYM_EXTRA_ROOTS` (NeMo-RL prepends your package root here) → `sys.path` → cwd → the Gym install root in the image. A directory counts as a server **only if it ships `requirements.txt` or `pyproject.toml`**. Three consequences:

1. **You may reference an image implementation without uploading its code.** `simple_agent`, `vllm_model` and `verifiers_agent` resolve from the image. Shipping `responses_api_agents/simple_agent/configs/*.yaml` with no `app.py` and no `requirements.txt` is correct and intended — the config comes from your package, the code from the image.
2. **A custom implementation must ship the install marker.** Without it the directory is skipped, and if a built-in shares the name Gym **silently runs the built-in** — training against the wrong environment with no error. If no built-in shares the name, spin-up fails with `Missing pyproject.toml or requirements.txt for uv venv setup in server dir: …`.
3. **Never put a `pyproject.toml` at the package root.** `setup_env_command` decides "editable install" by testing `{server_dir}/../../pyproject.toml`, which from `<root>/resources_servers/<impl>/` **is the package root**. Present, Gym takes the editable branch — and that branch is the one that does *not* prepend `nemo-gym=={image version}`. Two ways that bites, depending on the server's requirements:

   | Server's `requirements.txt` | Result |
   |---|---|
   | Keeps Gym's `-e nemo-gym[dev] @ ../../` | `uv pip install -r requirements.txt` runs verbatim, so pip installs **your package root** as `nemo-gym` |
   | Does not keep it | Nothing installs `nemo-gym` at all — only the non-editable branch adds the pin — so the server fails on its first import |

   A server-level `pyproject.toml` behaves the same way (`uv pip install -e .`, no `nemo-gym` pin). Keep the server's original directory structure from the Gym checkout; do not repackage the environment as a setuptools distribution.

`requirements.txt` and `pyproject.toml` are **mutually exclusive** in one server directory — shipping both raises at spin-up.

## Dependency installation — settle this first

Source of truth: `nemo_gym/cli/setup_command.py` (`setup_env_command`), `nemo_rl.environments.gym_env_package` (in the RL image), `docker/rl/README.md` § *NeMo-Gym environments: a second, separate venv layer*.

Gym builds **one venv per server** at spin-up, under `NEMO_GYM_VENV_DIR` (`/opt/gym_venvs`). That directory ships **empty** — `NEMO_GYM_PREFETCH_CONFIGS` is off by default — so every environment pays this on first use, built-in ones included.

The platform itself installs nothing. `grpo_driver` calls `bootstrap_environment_package(..., install_wheels=False)`: it validates and stops there, because the venvs a server runs from do not exist until `RunHelper.start`.

**There are two installs, and they have different offline guarantees.** This is the single most-confused point about the wheels formats.

| | Step 1 — Gym builds the venv | Step 2 — NeMo-RL installs the package |
|---|---|---|
| When | During `RunHelper.start`, per server | After `RunHelper.start`, per agent/resources venv |
| What | `uv venv --seed` + `uv pip install` of the server's `requirements.txt` / `pyproject.toml`, **plus** `nemo-gym==<image version>`, `ray[default]==<pin>`, `openai==<pin>` | `uv pip install --no-index --find-links=wheels/ <names>` |
| Wheelhouse role | `UV_FIND_LINKS` — a **candidate pool**. Anything it covers installs from local files; anything it misses **goes to an index** | Hard offline. `--no-index` is explicit |
| Requirements | The server's own, resolved normally | One **unpinned** distribution name per wheel in `wheels/` |

So `--no-index` is real, but it is step 2 only. **A job is offline-clean only when `wheels/` also satisfies step 1**, which means the closure must cover:

- everything the server's `requirements.txt` names, and their transitive deps;
- **`nemo-gym` at exactly the version the training image reports** (below);
- `ray[default]` and `openai` at the versions Gym pins as head-server deps — the sub-venv starts from `uv venv --seed`, so nothing is inherited from the image.
- **`pip`** — `uv venv --seed` installs it into each new virtualenv *before* any `uv pip install` runs, resolving it like any other package. Without it in `wheels/` the seed step reaches for an index and the job fails with `Failed to install seed packages into virtual environment` / `No solution found when resolving: pip`. `uv venv` honours `UV_FIND_LINKS`, so vendoring the wheel is all that is needed. Python 3.13 seeds only `pip`.

Completeness of `wheels/` is what makes a sandboxed run work. The format name alone does not.

Step 2's requirements are unpinned on purpose (`wheelhouse_requirements`): the wheelhouse was resolved against the environment alone, while step 1 resolved the venv from the server's requirements. Pinning would overwrite packages the already-running server has imported. An unpinned requirement that is already satisfied is a no-op — which also means step 2 **cannot repair** a step 1 that resolved the wrong version from an index.

| Format | Server deps come from | Needs egress at spin-up |
|---|---|---|
| `native-v1` | An index, per the server's `requirements.txt` | **Yes** |
| `wheels-v1` | The package's `wheels/`, for whatever it vendors | **No**, if the closure covers step 1 as well |
| `adapter-wheels-v1` | The package's complete `wheels/` closure, including the agent harness dependencies | **No** |

The converter resolves the hub environment and `verifiers_agent` together. Source-only
releases are built into wheels on the internet-connected conversion host, and conversion
fails if any pinned distribution is still missing from `wheels/`. The image-bundled agent
harness uses the matching `verifiers` version pin, allowing Gym to resolve it from the
FileSet with `UV_OFFLINE=1` instead of following a Git URL.

Egress is operator config, never a job field: `NMP_RL_SANDBOX_ALLOW_INTERNET`, plus `NMP_RL_SANDBOX_PUBLIC_DNS_ALLOW` for hosts outside NeMo-RL's built-in `*.com` / `*.org` allowance (e.g. `hub.primeintellect.ai`). Check with the operator before committing a user to `native-v1` on a deny-default cluster. Details: `rl-kubernetes-runtime.md` § **Sandboxed Gym (GRPO)**.

### What goes in a server's `requirements.txt`

Gym reads it two different ways depending on where the server sits:

| Situation | Command Gym runs |
|---|---|
| Inside a Gym checkout (`../../pyproject.toml` exists) | `uv pip install -r requirements.txt <head deps>` |
| Staged FileSet (the normal case) | `(echo 'nemo-gym==<image version>' && grep -v -F '../..' requirements.txt) \| uv pip install -r /dev/stdin <head deps>` |

The second form is why a Gym server copies cleanly: its `-e nemo-gym[dev] @ ../../` line, meaningless outside a checkout, is stripped and replaced with a version pin. It is also why vendoring **the image's exact `nemo-gym` version** matters — a mismatch means uv ignores your wheel and resolves upstream from PyPI.

**Do not ship a literally empty `requirements.txt`.** The file's *existence* is what makes Gym treat the directory as a server, and an empty one does currently install (`grep` finds nothing, `nemo-gym` and the head deps still reach uv). But the resulting venv contains only `nemo-gym`, `ray` and `openai` — so any other import in `app.py` fails at the first rollout — and it leaves the install command's `grep` exiting non-zero, which is harmless only because nothing enables `pipefail` on this path today. List the server's real imports; if it genuinely has none beyond `nemo-gym`, write a comment line rather than leaving the file empty.

## The manifest — `nemo-environment.yaml`

Required at the package root in every format. Parsed by `parse_manifest`; unknown keys are **rejected** (`extra="forbid"`).

```yaml
format: adapter-wheels-v1          # or native-v1 / wheels-v1
config_paths:                      # >=1, relative, no "..", no symlinks, must exist
  - configs/policy_model.yaml
  - configs/verifiers_agent.yaml
adapter:                           # adapter-wheels-v1 ONLY
  agent: verifiers_agent
  agent_type: responses_api_agents
  image_config_root: responses_api_agents/verifiers_agent
metadata:
  name: ascii-tree                 # required
  description: ...                 # optional
  hub_id: primeintellect/ascii-tree
  vf_env_id: ascii-tree            # must match every dataset row's vf_env_id
  adapter_agent: verifiers_agent
```

`metadata` is provenance only — nothing in it changes how the environment installs or runs, **except** that `vf_env_id` is what dataset rows are checked against when you validate them.

### Rules every format shares

| Rule | Enforced by |
|---|---|
| `nemo-environment.yaml` at the package root | `load_manifest` |
| At least one `config_paths` entry, each relative and free of `..` | `_ManifestBase` |
| Every `config_paths` entry exists in the package | `validate_manifest_against_listing` |
| No symlinks, and no path escaping the package root | `validate_package_layout` |
| **No `.jsonl` anywhere in the package** | `validate_manifest_against_listing` |
| `wheels/` non-empty and `*.whl`-only (wheels formats) | `offline_wheel_install_required` |

## Example scripts

Before hand-building anything, check whether an example already covers the case.
`scripts/grpo-examples/` ships two, and they are **examples** — nothing in the platform calls
them, and the supported contract is the FileSet layout itself.

```bash
# Any Gym server -> an environment package
uv run scripts/grpo-examples/gym_to_env_package.py \
  --gym-root ~/workspace/Gym --server resources_servers/math_with_judge \
  --format wheels-v1 --arch x86_64 --out-dir /tmp/mwj-env

# math_with_judge rollout rows (adds agent_ref and expected_answer)
uv run --with datasets scripts/grpo-examples/prepare_math_with_judge.py \
  --out-dir /tmp/mwj-data --train-size 512
```

Upload both, then submit one of `plugins/nemo-rl/tests/fixtures/minimal_grpo*.json`, which
reference `default/math-with-judge-env` and `default/math-with-judge-gym-data`. Full walkthrough:
`scripts/grpo-examples/README.md`.

**NeMo Gym is not vendored in nemo-platform** and is not a platform dependency, so the two
Gym-source formats need a checkout you provide:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym ~/workspace/Gym
```

Everything below is the hand-built path, for an environment the examples do not cover.

## Path A — an environment already in Gym (`native-v1`)

Use when the user names an environment that ships with NeMo Gym (`resources_servers/<name>/`, `responses_api_agents/<name>/`) or has written one in that layout. The package is a **slice of the Gym tree** with its directory structure preserved.

`native-v1` ships no `wheels/`, so its per-server venv resolves from a package index at spin-up and **the job needs egress**. That is the only thing the format choice decides — it does *not* mean the server's imports must already be in the image, since Gym installs them at first use either way. On a deny-default cluster, ship the same tree as `wheels-v1` instead (Path C).

**Drop `data/` and `tests/`.** Gym's own configs point `datasets[].jsonl_fpath` at an in-tree file, and any `.jsonl` in the package is rejected. Remove the `datasets` block and let the dataset FileSet supply rows — see `dataset-formats.md` § **NeMo-RL (GRPO)**.

### Worked example — a tool-using weather environment

This is Gym's own `resources_servers/example_single_tool_call` restated as a package. It exercises every concept above: a custom resources server, an image agent referenced by config only, and the `policy_model` server.

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

Gym's in-tree rows omit `agent_ref` because the owning agent config supplies it. **The platform path requires it on every row.** See `dataset-formats.md` § **NeMo-RL (GRPO)**.

## Path B — a Prime Intellect / verifiers env (`adapter-wheels-v1`)

The automated path, and the only one with a converter. It downloads the hub package, vendors its full wheel closure, writes both configs and the manifest, and snapshots the hub environment's Hugging Face dataset into the dataset FileSet so training does not call the Hub.

That snapshot is two artifacts: `hub_environment.parquet`, with `vf_env_args.dataset_path` set to `/job/dataset/hub_environment.parquet`, and `.huggingface/`, the cache populated while converting. Loaders that honor `dataset_path` read the parquet; loaders that still call `load_dataset("org/name")` resolve from the cache, offline.

**Run it on a host with internet.** Training clusters have no hub egress and consume uploaded FileSets only.

`pi-to-gym-conversion` is a console script that ships with `nemo-rl-plugin`, so on an installed platform run it bare. From a repo checkout, generating a dataset needs the `conversion` extra (verifiers), and it belongs in its own environment: the converter installs the untrusted hub wheel into whatever interpreter runs it, and a `uv sync` of the repo `.venv` prunes both that wheel and `verifiers` back out.

```bash
UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --package nmp-rl --extra conversion
.venv-conversion/bin/pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree \
  --hub-version 0.1.5 \
  --out-dir ./ascii-tree-pkg \
  --dataset-dir ./ascii-tree-data \
  --validation-fraction 0.1
```

`--validate-only` needs neither, so the plain `uv run --package nmp-rl pi-to-gym-conversion` used below for validation is fine from the repo `.venv`.

| Flag | Use it for |
|---|---|
| `--hub-id` | Hub slug. Required for conversion. |
| `--hub-version` | **Pin it.** Unset takes whatever the index offers now; a later release can narrow `Requires-Python` and fail the download, or install but not run on the training image's Python. This is what makes a conversion reproducible. |
| `--out-dir` | Environment package output. |
| `--dataset-dir` | Prompt JSONL output. Defaults to a sibling of `--out-dir`. |
| `--vf-env-id` | Override the verifiers load id. Defaults to the hub slug's last segment. |
| `--vf-env-args` | JSON object forwarded to the environment loader. Must be a JSON **object**. |
| `--dataset-size` | Cap rows (`-1` = all). |
| `--validation-fraction` | Split fraction in `[0, 1)`. `0` writes training rows only. |
| `--wheels-dir` | Use pre-vendored wheels instead of downloading. |
| `--upload` | Create both FileSets and upload, in one step. Needs `NMP_BASE_URL`. |
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

**`pi-to-gym-conversion` vendors `x86_64` wheels today.** On an `arm64` cluster, build the closure separately and pass `--wheels-dir`.

**`adapter.agent` must be on the image allowlist.** The package ships config and wheels; the agent harness itself comes from the training image. Today the allowlist is exactly one entry — `verifiers_agent` (`IMAGE_ADAPTER_ALLOWLIST`). Anything else is rejected at validation, so a user who wants a different harness needs `wheels-v1` or `native-v1`.

**Why two configs — and this is not adapter-only.** `verifiers_agent.yaml` references a model server named `policy_model`; nothing defines that server unless a second config supplies it, and Gym resolves every ref when it merges configs, failing with `ServerRefNotFoundError: ... Available responses_api_models: (none)`. `policy_model.yaml` supplies it, interpolating `${policy_base_url}` / `${policy_api_key}` / `${policy_model_name}` against the config NeMo-RL injects at spin-up — which is what points Gym at the vLLM engine the job is already running instead of starting its own.

**Every format needs this file.** Gym's own server configs reference `policy_model` the same way (`math_with_judge` needs it for both `model_server` and `judge_model_server`, and `should_use_judge: false` does not help — resolution happens before any judging would be skipped). What it contains:

```yaml
policy_model:                      # instance — what every {type, name} ref points at
  responses_api_models:
    vllm_model:                    # implementation, from the training image
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: true
      uses_reasoning_parser: true
```

The three interpolations resolve against the global config NeMo-RL injects at spin-up, which is what points Gym at the vLLM engine the job is already running instead of starting its own. The converter writes it; hand-built packages must add it. Generate it from the same helper the converter uses, so the two cannot drift:

```python
import yaml
from nmp.rl.tasks.environment.package import build_policy_model_yaml

(out_dir / "configs").mkdir(parents=True, exist_ok=True)
(out_dir / "configs" / "policy_model.yaml").write_text(
    yaml.safe_dump(build_policy_model_yaml(), sort_keys=False), encoding="utf-8"
)
```

List it **first** in `config_paths`, ahead of the config that references it. Placement differs by format: `configs/policy_model.yaml` for the two wheels formats; `native-v1` must live under a Gym server prefix. Use `responses_api_models/vllm_model/configs/policy_model.yaml` so the on-disk path matches the YAML implementation (`vllm_model`). A folder named `policy_model` also passes the prefix check; Gym runs `responses_api_models/vllm_model/` from the YAML body, not from the config file's directory.

**`max_tokens` lives in the package, not the job.** `configs/verifiers_agent.yaml` carries `max_tokens` (default 8192), and `verifiers_agent` reads it in preference to the job's `max_new_tokens`. To bound response length for an adapter env, edit this file before uploading — see `hyperparameters-rl.md` § **Known limitation**.

## Path C — vendored code (`wheels-v1`)

The catch-all, and the right answer more often than its "bring your own code" framing suggests. Use it for a custom verifier or bespoke agent — **and for a Gym source-tree server whose dependencies have to travel with it**, which is the common case on a cluster without egress.

`wheels-v1` is the least constrained format — `config_paths` may sit anywhere — and correspondingly the one with the least done for you. There is no converter; build it by hand.

Gym runs `{server_type}/{implementation}/` and requires `requirements.txt` or `pyproject.toml` there. A `configs/` + `wheels/` package with **no** server directories only works when every YAML implementation is already in the image (`simple_agent`, `vllm_model`, `verifiers_agent`). A custom `resources_servers/my_env` **must** ship that directory, even if the YAML itself lives under `configs/`.

```text
my-env/
  nemo-environment.yaml
  configs/policy_model.yaml     ← required; see Path B
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

### Vendoring the closure

Two consumers, per § **Dependency installation**: Gym's venv build reaches the wheelhouse through `UV_FIND_LINKS` (candidate pool — misses go to an index), and NeMo-RL's later install reads it with `--no-index`. Build the closure for the first, and the second follows for free.

Resolve it from the **`nemo-gym` wheel plus the server's requirements**, not from the requirements alone: the sub-venv starts at `uv venv --seed`, so everything `nemo_gym` itself imports has to be present too.

**Wheels must target the architecture of the training nodes, not the build host.** The training images are published for both `linux/amd64` and `linux/arm64`, so confirm which the cluster runs (`kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}'`) and set `ARCH` to `x86_64` or `aarch64` accordingly. The interpreter is **Python 3.13** either way. A closure built for the wrong architecture passes `--validate-only` — which checks layout, never wheel tags — and then fails on the cluster with `has no wheels with a matching platform tag`.

Three versions must match what the **Gym host process** reports, because that is what Gym stamps into every sub-venv install: `nemo-gym` (the sub-venv pin) and `ray[default]` / `openai` (the head-server deps, read as `ray.__version__` / `openai.__version__` in `global_config.py`). Gym runs from its own actor venv under `/opt/ray_venvs`, not the image's default interpreter, so read them from there rather than assuming the base venv agrees:

```bash
IMAGE=<training-image>
# The Gym actor venv is named after its actor FQN, e.g.
# /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python
read GYM_VERSION RAY_VERSION OPENAI_VERSION < <(docker run --rm "$IMAGE" sh -c '
  PY=$(ls -d /opt/ray_venvs/*NemoGym*/bin/python 2>/dev/null | head -1)
  "${PY:-python}" -c "import importlib.metadata as m; print(m.version(\"nemo-gym\"), m.version(\"ray\"), m.version(\"openai\"))"')

mkdir -p my-env/wheels
pip download --dest my-env/wheels \
  --only-binary=:all: \
  --python-version 3.13 \
  --platform "manylinux_2_39_$ARCH" \
  --platform "manylinux_2_28_$ARCH" \
  --platform "manylinux_2_17_$ARCH" \
  --platform "manylinux2014_$ARCH" \
  "nemo-gym==$GYM_VERSION" "ray[default]==$RAY_VERSION" "openai==$OPENAI_VERSION" pip \
  -r resources_servers/my_env/requirements.txt
uv run --package nmp-rl pi-to-gym-conversion --validate-only ./my-env
```

The sub-venv is created with `uv venv --seed`, so nothing carries over from the image — omit `ray[default]` / `openai` and the venv build reaches for an index even when the environment's own closure is complete. A `nemo-gym` version mismatch is worse than an omission: uv ignores your wheel and silently resolves upstream from PyPI.

`--platform` is repeated because pip matches tags literally rather than expanding a compatibility range, so a project publishing only against an older glibc floor is missed unless its tag is named.

**Vendor exactly one version of each distribution.** Several versions of one project is legal for a find-links pool, so it only warns locally — but the package then no longer pins what gets installed. Clear `wheels/` before rebuilding into it: copies overwrite by filename, so a wheel from an earlier run survives whenever the new closure resolved that project to a different version.

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

Keep the Gym directory structure, drop `data/` and `tests/` so no `.jsonl` survives, add `configs/policy_model.yaml` and the manifest at the root, and vendor `wheels/`. `config_paths` then lists `configs/policy_model.yaml` and the server's own in-tree config — `wheels-v1` allows either location.

Do **not** convert the server into a setuptools package installed as `resources_servers.<impl>`, and do not add a root `pyproject.toml`: Gym runs `{server_type}/{implementation}/` directly and treats a root `pyproject.toml` as "this is a Gym checkout" — see § **How Gym finds the implementation directory**.

**Vendor `nemo-gym` at the image's exact version.** Gym builds each server's venv from its `requirements.txt`, which in the source tree starts `-e nemo-gym[dev] @ ../../`. Outside a Gym checkout that relative path does not exist, so `setup_env_command` rewrites the line to `nemo-gym==<image version>`. If the environment must run our Gym fork, `pip download nemo-gym==X` fetches *upstream* at that version string — build the wheel from the fork checkout instead (`uv build --wheel <gym-root>`) and confirm the built version matches.

Validate the tree with `--validate-only`.

## Validate before uploading

Cheapest possible failure. Run it on every package, whichever path built it:

```bash
uv run --package nmp-rl pi-to-gym-conversion --validate-only ./my-env-pkg
# {"valid": true, "format": "adapter-wheels-v1", "name": "ascii-tree"}
```

Exit 1 with the specific violation on failure. The same checks run again at submit time against the FileSet listing, so a package that validates here will not be rejected for layout later.

**What it does not check:** wheel platform tags, whether the closure is complete, whether `config_paths` actually define the servers your rows route to, or anything about dataset rows. Those all surface at spin-up. Validation is a layout check, not a dry run.

## Upload

The converter can do both FileSets in one step:

```bash
export NMP_BASE_URL=http://127.0.0.1:8080
.venv-conversion/bin/pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree --hub-version 0.1.5 \
  --out-dir ./ascii-tree-pkg --upload --workspace default
```

Without `--environment-name` / `--dataset-name` it derives them from the hub slug as **`<slug>-env`** and **`<slug>-env-dataset`** — here `ascii-tree-env` and `ascii-tree-env-dataset`. Read the names back from the `uploaded` block in its JSON output rather than assuming.

Or by hand, which is the only option for Paths A and C. **`--purpose environment` is enforced, not cosmetic**: `check_environment_access` rejects an environment FileSet whose purpose is anything other than `environment` or `generic`, so a package uploaded to a `dataset`-purpose fileset fails at submit. (`dataset`, `environment`, `generic`, `model` are the four valid values.)

`nemo files upload` takes a directory and uploads recursively, preserving relative paths, so one command per fileset is enough:

```bash
ENV=ascii-tree-env
nemo files filesets create "$ENV" --workspace default --purpose environment --exist-ok
nemo files upload ./ascii-tree-pkg/ "$ENV" --workspace default
nemo files list "$ENV" --workspace default

DATA=ascii-tree-env-dataset
nemo files filesets create "$DATA" --workspace default --purpose dataset --exist-ok
nemo files upload ./ascii-tree-data/ "$DATA" --workspace default
```

The **trailing slash** on the local path changes the result (fsspec): it selects the directory's contents rather than the directory itself. The CLI preserves the raw argument specifically so this works. Omit it and everything nests one level deeper under the directory's basename — which the dataset FileSet catches at submit (`must contain training.jsonl`) but the environment FileSet catches only as `Missing nemo-environment.yaml at environment root`. Confirm with `nemo files list` before submitting.

Relative paths must be preserved: `config_paths` is matched against the FileSet listing, so a package flattened during upload fails with `config_paths reference files that are not in the package`.

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `Missing nemo-environment.yaml at environment root` | No manifest, or uploaded one directory too deep | Manifest at the FileSet root, not inside a subdirectory |
| `Invalid nemo-environment.yaml: ... extra_forbidden` | Unknown manifest key | Remove it; the schema is closed |
| `config_paths reference files that are not in the package` | Upload flattened the tree, or a typo | Re-upload preserving relative paths |
| `Prompt JSONL must not live in the environment package` | `.jsonl` inside the package | Move rows to the dataset FileSet |
| `native-v1 config_paths must be under (...)` | Config outside the three Gym server dirs | Move it, or switch to `wheels-v1` |
| `adapter-wheels-v1 config_paths should live under configs/` | Config elsewhere | Move it under `configs/` |
| `format ... installs its dependencies offline, so the package must carry a non-empty wheels/ directory` | Wheels format with no wheels | Vendor the closure |
| `Non-wheel files in wheels/` | Stray file (sdist, README) | `wheels/` holds `*.whl` only |
| `adapter.agent ... is not built into the training image` | Harness not on the allowlist | Use `verifiers_agent`, or switch format |
| `ServerRefNotFoundError: ... Available responses_api_models: (none)` | A config references `policy_model` with nothing defining it | Add a `policy_model` config — **every** format needs one; see Path B |
| `Symlinks are not allowed` | Symlinked config | Ship real files |
| `Environment fileset '…' has purpose 'dataset'; expected purpose='environment'` | Package uploaded to a fileset created with the wrong purpose | Recreate with `--purpose environment` |
| `AlmostServerError` / "Almost-Servers Detected" banner | A server block failed validation but looks like one — most often a `domain` that is empty or not in the closed set | Set a valid `domain` on every `resources_servers` block; remove `domain` from agent/model blocks |
| `Missing pyproject.toml or requirements.txt for uv venv setup in server dir` | Custom implementation shipped without an install marker | Add `requirements.txt` to `{server_type}/{implementation}/` |
| `Found both pyproject.toml and requirements.txt … Please only use one or the other!` | Both markers in one server directory | Keep one |
| Job runs but scores the wrong environment | Custom server dir has no install marker **and** shares a name with a Gym built-in, so Gym resolved the built-in | Add `requirements.txt`, or rename the implementation |
| `ModuleNotFoundError` at the first rollout | Server venv missing an import — empty/incomplete `requirements.txt`, or the wheelhouse missed it at venv-build time | List real imports; vendor the step-1 closure (§ **Dependency installation**) |
| `your requirements are unsatisfiable` on the cluster | `wheels/` vendors two versions of one project | Clear `wheels/` and rebuild the closure |
| `has no wheels with a matching platform tag` | Closure built for the build host (macOS / arm64), not the image | Re-download with the `--python-version` / `--platform` flags in Path C |
| Spin-up stalls, then fails resolving a package | The per-server venv build cannot reach an index | Vendor it (`wheels-v1`), or ask the operator for `NMP_RL_SANDBOX_ALLOW_INTERNET` — see § **Dependency installation** |
| `OpenSandbox is not yet available on this cluster (sandbox_cluster_capable=false)` | Operator has not enabled sandboxed Gym | Platform config, not the package — `rl-kubernetes-runtime.md` § **Sandboxed Gym (GRPO)** |
| `Sandboxed GRPO requires the job-storage PVC claim name` | `NMP_RL_JOB_STORAGE_PVC_CLAIM` unset | Same — operator config |

The last two fail **at submit**, before any GPU is claimed, and no change to the package will fix them.

Environment-incompatible-with-reasoning-parser and vLLM `async_engine` errors are runtime, not packaging — see `troubleshooting.md`.

## Related

| For | Read |
|---|---|
| Gym prompt-row schema and converting a dataset to it | `dataset-formats.md` § **NeMo-RL (GRPO)** |
| GRPO job JSON and hyperparameters | `hyperparameters-rl.md` |
| Kubernetes runtime, sandboxed Gym, PVC, egress | `rl-kubernetes-runtime.md` § **Sandboxed Gym (GRPO)** |
| Manifest schemas (source of truth) | `services/rl/src/nmp/rl/schemas/environment.py` |
| Validation rules (source of truth) | `services/rl/src/nmp/rl/tasks/environment/validate.py` |
| Converter, wheel targeting, `build_policy_model_yaml` | `services/rl/src/nmp/rl/tasks/environment/convert.py`, `package.py`, `__main__.py` (`pi-to-gym-conversion`) |
| Gym config semantics: instance/implementation, `Domain`, almost-servers | `nemo_gym/config_types.py`, `nemo_gym/global_config.py` (Gym checkout) |
| Server-dir resolution and the per-server venv command | `nemo_gym/cli/env.py` (`_resolve_server_dir`, `RunHelper.start`), `nemo_gym/cli/setup_command.py` (`setup_env_command`) |
| How wheels and search roots actually reach Gym | `nemo_rl.environments.gym_env_package` (RL image), `docker/rl/README.md` § **NeMo-Gym environments** |
