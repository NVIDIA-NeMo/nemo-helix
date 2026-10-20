<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Instance acceptance check

Prove that one running NeMo Helix instance can do the work a new install has to do: create a secret, run a job, run a deployment, and call the platform as the identity that install is configured to use.

Run it by hand after the instance is up. It uses the saved CLI context in `~/.config/nhx`, and the logged-in user when authentication is on. `--context` selects another saved context. This is a live check, separate from `make test-unit`. The offline tests under `unit/` are the only files `pytest tools` collects.

```bash
uv run python tools/instance_check/check.py
uv run python tools/instance_check/check.py --context prod
uv run python tools/instance_check/check.py --model workspace/model-name
```

Discovery chooses the auth profile, and the suite asks only for what that profile can prove:

- **anonymous** — auth is off. The caller and the workloads reach the platform with no principal.
- **trusted_headers** — auth is on and workload token exchange is off. A job runs as the caller. A deployment is shown to finish; its identity sidecar is not settable from the public deployment-config API.
- **token_exchange** — auth and token exchange are on. Jobs and deployments must present a workload-identity token. A principal id alone is not enough.

A CPU model deployment runs every time, to show the models API can bring a container up. `--model workspace/name` also sends a chat completion through the inference gateway, from the test process and from inside the workload. Each test docstring states its own assertion.

The run creates a `check-<8 hex>` workspace and deletes it at the end. `--workspace` keeps an existing workspace. `--keep` keeps the temporary one. `--image` supplies the deployment task image when the jobs profile does not already name one.
