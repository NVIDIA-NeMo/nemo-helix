<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example scripts (needs a nemo-helix checkout)

`scripts/grpo-examples/` at the root of a nemo-helix checkout builds the `math_with_judge` text pair and the CLEVR-CoGenT vision pair. They are **not** part of an installed skill and need the repository: packaging imports `nhx-rl` from the source tree. Nothing in the platform calls them, and the supported contract is the FileSet layout itself. Their walkthrough is `scripts/grpo-examples/README.md` in the checkout.

Run them from a dedicated environment, not the project `.venv`: `wheels-v1` downloads wheels with `sys.executable -m pip`, which would change the project environment.

```bash
UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --frozen --package nhx-rl --extra conversion

# Any Gym server -> an environment package
.venv-conversion/bin/python scripts/grpo-examples/gym_to_env_package.py \
  --gym-root ~/workspace/Gym --nemo-rl-root ~/workspace/RL \
  --server resources_servers/math_with_judge \
  --format wheels-v1 --arch x86_64 --out-dir /tmp/mwj-env

# math_with_judge rollout rows (adds agent_ref and expected_answer).
# --no-project so this does not sync the project .venv.
uv run --no-project --with datasets scripts/grpo-examples/prepare_math_with_judge.py \
  --out-dir /tmp/mwj-data --train-size 512

# CLEVR-CoGenT: the vlm server lives in this repo, not in the Gym checkout.
.venv-conversion/bin/python scripts/grpo-examples/gym_to_env_package.py \
  --gym-root ~/workspace/Gym --nemo-rl-root ~/workspace/RL \
  --server-dir scripts/grpo-examples/servers/vlm \
  --config clevr_cogent.yaml \
  --format wheels-v1 --arch x86_64 --out-dir /tmp/clevr-env
uv run --no-project --with datasets --with pillow scripts/grpo-examples/prepare_clevr_cogent.py \
  --out-dir /tmp/clevr-data --train-size 32 --validation-size 16 --prompt qwen
```

`servers/vlm` is the Gym port of NeMo-RL `VLMEnvironment`. `configs/clevr_cogent.yaml` sets `format` weight `0.2` and `exact_alnum` weight `0.8`. Rows carry `expected_answer` and an `input_image` data URI. The agent name is `clevr_cogent_simple_agent`.

**NeMo Gym is not vendored in nemo-helix** and is not a platform dependency. Packaging a Gym
server (`native-v1`, or `wheels-v1` from a Gym tree) runs on the machine that builds the
package, so clone Gym there; the training cluster never needs a checkout:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym ~/workspace/Gym
```
