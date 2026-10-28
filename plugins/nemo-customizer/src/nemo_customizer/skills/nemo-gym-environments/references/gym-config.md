<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How Gym reads an environment package

The vocabulary and rules every format shares: config YAML, server directories, the `policy_model` config, and the manifest.

## How a Gym config YAML is read

In a NeMo Gym checkout, this comes from `nemo_gym/config_types.py` and `nemo_gym/cli/env.py` (`RunHelper.start`, `_resolve_server_dir`).

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

Pick the most specific fit; `other` is the catch-all. An empty string, a missing value, or a typo fails enum validation, and because the block still *looks* like a server (it has an `entrypoint`), Gym classifies it as an **almost-server**: it prints a warning banner and then raises `AlmostServerError`, aborting spin-up. `error_on_almost_servers` defaults to **true** and the platform does not turn it off, so this is a hard failure, not a silently skipped server. Gym strips `domain` from model and agent blocks.

### How Gym finds the implementation directory

`_resolve_server_dir` searches, in order, `NEMO_GYM_EXTRA_ROOTS` (NeMo-RL prepends your package root here) → `sys.path` → cwd → the Gym install root in the image. A directory counts as a server **only if it ships `requirements.txt` or `pyproject.toml`**. Three consequences:

1. **You may reference an image implementation without uploading its code.** `simple_agent`, `vllm_model` and `verifiers_agent` resolve from the image. Shipping `responses_api_agents/simple_agent/configs/*.yaml` with no `app.py` and no `requirements.txt` is correct and intended — the config comes from your package, the code from the image.
2. **A custom implementation without the install marker is skipped** (SKILL.md rule 4), and if a built-in shares the name Gym **silently runs the built-in** — training against the wrong environment with no error. If no built-in shares the name, spin-up fails with `Missing pyproject.toml or requirements.txt for uv venv setup in server dir: …`.
3. **A `pyproject.toml` at the package root switches Gym to its editable install.** `setup_env_command` decides "editable install" by testing `{server_dir}/../../pyproject.toml`, which from `<root>/resources_servers/<impl>/` **is the package root**. Present, Gym takes the editable branch — and that branch is the one that does *not* prepend `nemo-gym=={image version}`. Two ways that bites, depending on the server's requirements:

   | Server's `requirements.txt` | Result |
   |---|---|
   | Keeps Gym's `-e nemo-gym[dev] @ ../../` | `uv pip install -r requirements.txt` runs verbatim, so pip installs **your package root** as `nemo-gym` |
   | Does not keep it | Nothing installs `nemo-gym` at all — only the non-editable branch adds the pin — so the server fails on its first import |

   A server-level `pyproject.toml` behaves the same way (`uv pip install -e .`, no `nemo-gym` pin).

`requirements.txt` and `pyproject.toml` are **mutually exclusive** in one server directory — shipping both raises at spin-up.

## The `policy_model` config — every format needs one

The converter writes two configs, and the second one is not adapter-only. `verifiers_agent.yaml` references a model server named `policy_model`; nothing defines that server unless a second config supplies it, and Gym resolves every ref when it merges configs, failing with `ServerRefNotFoundError: ... Available responses_api_models: (none)`. `policy_model.yaml` supplies it, interpolating `${policy_base_url}` / `${policy_api_key}` / `${policy_model_name}` against the config NeMo-RL injects at spin-up — which is what points Gym at the vLLM engine the job is already running instead of starting its own.

Gym's own server configs reference `policy_model` the same way (`math_with_judge` needs it for both `model_server` and `judge_model_server`, and `should_use_judge: false` does not help — resolution happens before any judging would be skipped). What it contains:

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

The converter writes it. For a hand-built package, generate it from the same helper the converter uses, so the two cannot drift:

```python
from pathlib import Path

import yaml
from nhx.rl.tasks.environment.package import build_policy_model_yaml

out_dir = Path("my-env")  # the package root, where nemo-environment.yaml lives
(out_dir / "configs").mkdir(parents=True, exist_ok=True)
(out_dir / "configs" / "policy_model.yaml").write_text(
    yaml.safe_dump(build_policy_model_yaml(), sort_keys=False), encoding="utf-8"
)
```

Placement differs by format: `configs/policy_model.yaml` for the two wheels formats; `native-v1` must live under a Gym server prefix. Use `responses_api_models/vllm_model/configs/policy_model.yaml` so the on-disk path matches the YAML implementation (`vllm_model`). A folder named `policy_model` also passes the prefix check; Gym runs `responses_api_models/vllm_model/` from the YAML body, not from the config file's directory.

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

`metadata` is provenance only — nothing in it changes how the environment installs or runs. The one value that matters is `vf_env_id`: every dataset row's `vf_env_id` must equal it. Neither `--validate-only` nor submit compares them, so a mismatch surfaces only when the environment fails to load at spin-up.

### Rules every format shares

| Rule | Enforced by |
|---|---|
| `nemo-environment.yaml` at the package root | `load_manifest` |
| At least one `config_paths` entry, each relative and free of `..` | `_ManifestBase` |
| Every `config_paths` entry exists in the package | `validate_manifest_against_listing` |
| No symlinks, and no path escaping the package root | `validate_package_layout` |
| **No `.jsonl` anywhere in the package** | `validate_manifest_against_listing` |
| `wheels/` non-empty and `*.whl`-only (wheels formats) | `offline_wheel_install_required` |
