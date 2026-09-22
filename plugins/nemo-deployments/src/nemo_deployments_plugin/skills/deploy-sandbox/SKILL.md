---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: deploy-sandbox
description: >-
  Deploys an already-built NAT agent as a policy-governed OpenShell sandbox on a
  local NeMo Platform, so the same agent image gets Landlock filesystem isolation
  and a pure default-deny network policy (the sandbox reaches nothing on the
  network directly; its model calls are brokered by the OpenShell gateway through
  the inference.local route) for free. Covers the proven wire-inference.local ->
  package -> DeploymentConfig -> deploy (executor openshell-local) -> wait ->
  query -> zero-egress proof -> cleanup flow. Use when the user asks to deploy an
  agent as a sandbox, sandbox a deployment, run an agent under a Landlock/egress
  policy, or use the openshell-local executor. Trigger keywords - deploy sandbox,
  sandboxed agent, openshell deployment, openshell-local executor, sandbox policy,
  egress policy, landlock, governed deployment, network egress governance,
  inference.local.
triggers:
  - deploy the agent as a sandbox
  - sandbox the deployment
  - deploy with openshell
  - openshell-local executor
  - run the agent under an egress policy
  - governed sandbox deployment
  - landlock the agent
not-for:
  - nemo-build-agent (use to scaffold + build the agent image first)
  - nemo-try-agent (use to query an already-deployed agent)
  - nemo-setup (use to install and start the platform first)
  - nemo-status (use for a read-only health dashboard)
compatibility: >-
  nemo-platform >= 0.1.0; requires the nemo-deployments plugin installed with the
  openshell extra (`openshell>=0.0.92` from PyPI); a running OpenShell docker-driver
  gateway on :17670 (not :8080, which the platform owns) whose JWT signing keys were
  generated once with `generate-certs`; a NeMo Platform reachable from inside a sandbox
  at the sandbox network's gateway address (`--host 0.0.0.0`) with the `openshell-local`
  executor loaded via
  `--config packages/nmp_platform/config/local.yaml`; the gateway's `inference.local`
  route wired to the platform Inference Gateway (`openshell provider create` +
  `openshell inference set`); the agents plugin for the packaging step. Docker-driver
  specific (sandboxes reach the platform at `host.openshell.internal:8080`). Linux
  (glibc >= 2.39) only: verified live end-to-end there, and the `inference.local` hop is
  unverified on macOS (Docker Desktop resolves the sandbox's `host.openshell.internal`
  inside its Linux VM, not to the Mac's loopback).
maturity: experimental
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Write, Edit
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Deploy an agent as a governed OpenShell sandbox

Deploy an ordinary NAT agent through the standard NeMo deployments API, but with the executor set to `openshell-local` so the agent runs inside an OpenShell sandbox under an automatically generated SandboxPolicy: Landlock filesystem isolation, `run_as_user: sandbox`, and a pure default-deny network policy. The agent reaches models through OpenShell's gateway-managed `inference.local` route, so the sandbox itself is granted no direct network egress at all: the model call is brokered over the supervisor's gateway channel rather than through a policy egress rule, and a call from the sandbox to anywhere else is blocked at the boundary.

The headline: the same agent image, deployed unchanged, gets its model access brokered by the gateway and network-egress governance for free. Its model calls resolve through `https://inference.local/v1`; a direct call to anywhere else is blocked at the sandbox boundary.

The full worked runbook with narration beats lives in the notebook at `plugins/nemo-deployments/examples/openshell/DEMO.ipynb`. The already-correct demo agent config lives at `plugins/nemo-deployments/examples/openshell/agent/config.yaml`, and a readable reference for the policy the backend generates (plus an override example) lives at `plugins/nemo-deployments/examples/openshell/local-sandbox-policy.yaml`. This skill is the concrete command path; read those files for context and overrides.

## When to use

- The user has a built (or buildable) NAT agent and wants it deployed under sandbox isolation with a governed default-deny network policy.
- The user explicitly asks for the `openshell-local` executor, a SandboxPolicy, or Landlock/egress governance.

If the agent image does not exist yet, run `nemo-build-agent` first to scaffold and validate the NAT workflow, then return here to deploy it as a sandbox. This skill does not author agent YAML.

## Pre-flight

Verify these prerequisites before touching the deploy steps. Do not improvise around a missing one; stop and tell the user what is not running.

Commands below assume `nemo` and `openshell` are on your PATH. In a repo checkout they live in the workspace venv, so prefix with `.venv/bin/` (e.g. `.venv/bin/openshell`, `.venv/bin/nemo`).

1. **OpenShell gateway reachable on `:17670`, with its JWT signing keys generated.**
   `:17670` is OpenShell's documented docker default. It must NOT be `:8080`: that
   belongs to the NeMo platform (the gateway container image and the CLI's `openshell
   gateway add http://127.0.0.1:8080` example both steer onto `:8080`, the collision
   trap). Docker-driver sandboxes REQUIRE the gateway to mint sandbox JWTs, so you must
   generate the signing keys once before the gateway starts (see `gateway.toml`'s
   `[openshell.gateway.gateway_jwt]` and the `docker-compose.yml` header). Run the
   one-time key generation, then bring the compose stack up and register it:

   ```bash
   # one-time: generate the JWT signing keys the docker driver needs to mint
   # sandbox tokens (writes /var/lib/openshell/tls/jwt/*). Re-run only if the
   # /var/lib/openshell state dir is wiped.
   docker run --rm --user 0 -v /var/lib/openshell:/var/lib/openshell \
     ghcr.io/nvidia/openshell/gateway:0.0.92 generate-certs \
     --output-dir /var/lib/openshell/tls \
     --server-san 127.0.0.1 --server-san localhost --server-san host.openshell.internal

   docker compose -f plugins/nemo-deployments/examples/openshell/docker-compose.yml up -d
   openshell gateway add http://127.0.0.1:17670 --local --name docker-dev
   ```

   Confirm the CLI reaches the gateway and the endpoint column reads `:17670`, not `:8080`:

   ```bash
   openshell gateway list          # * docker-dev  http://127.0.0.1:17670  ... plaintext
   openshell sandbox list          # reaches the gateway (e.g. "No sandboxes found.")
   ```

   Tear the gateway down after the demo with `docker compose -f
   plugins/nemo-deployments/examples/openshell/docker-compose.yml down`.

2. **NeMo Platform up, listening on all interfaces, started with the executor config.** The
   `openshell-local` executor lives in `packages/nmp_platform/config/local.yaml`,
   which is NOT the config `nemo services run` loads by default (that bundled default
   has no `deployments:` openshell executor), so it must be passed with `--config`.
   Start (or restart) it as:

   ```bash
   export NMP_BASE_URL=http://localhost:8080
   nemo services run --host 0.0.0.0 --port 8080 \
     --config packages/nmp_platform/config/local.yaml
   curl -sf http://localhost:8080/health/ready      # {"status":"ready"}
   ```

   If the platform is down, route to `nemo-setup`. `--host 0.0.0.0` matters: the
   `inference.local` route is dialed from the sandbox, not from the gateway process. The
   docker driver writes a literal `172.18.0.1 host.openshell.internal` into each
   sandbox's `/etc/hosts` (the sandbox network's gateway address), so the platform has
   to be listening there. On `--host 127.0.0.1` every model call fails with a 503. It
   also exposes an unauthenticated dev platform on every interface, LAN included, so do
   not run this on an untrusted network. If the platform is up but was started
   without `--config`, the executor is absent and Step 4 fails with an unknown-executor
   error; restart with the `--config` flag above. The executor block that ships in that
   file:

   ```yaml
   deployments:
     executors:
       - name: local-docker      # ordinary docker deployments (the default)
         backend: docker
         config: { pull_images: false, port_range_start: 9000, port_range_end: 9100 }
       - name: openshell-local
         backend: openshell
         config:
           gateway_endpoint: http://127.0.0.1:17670
           serve_workdir: /home/sandbox     # sandbox-writable CWD for `nat serve`
           platform_egress: null            # agent uses inference.local (gateway-managed),
                                            # so the sandbox needs NO direct egress ->
                                            # a pure default-deny SandboxPolicy
     default_executor: local-docker         # sandbox is opt-in; Step 4 names the executor
   ```

   `platform_egress: null` is the shipped default and yields a policy with no egress
   rules at all. See the Alternate branch at the end if you deliberately want the older
   direct-egress path.

3. **Packaging and backend dependencies present.** Step 2 (`nemo agents package`) needs
   `python-on-whales` (the agents plugin `container` extra), and the OpenShell backend
   needs the `openshell` SDK (the deployments plugin ships it in its `openshell` extra):

   ```bash
   .venv/bin/python -c "import python_on_whales, openshell" && echo DEPS_OK || echo DEPS_MISSING
   ```

   If `DEPS_MISSING`, install the backend extra: `uv sync --package
   nemo-deployments-plugin --extra openshell`. Base `make bootstrap-python` does not
   install the platform-restricted `openshell` wheel, so it needs this step. The
   packaging dependencies ship with `nemo-agents-plugin` itself and need nothing extra.

Convenience variable for the curl steps:

```bash
BASE=http://localhost:8080/apis/deployments/v2/workspaces/default
```

## Step 1: Pick a model and wire the `inference.local` route

The agent reaches models through OpenShell's gateway-managed `inference.local` route: the gateway routes `https://inference.local/v1` to a registered provider (the platform Inference Gateway) and injects the real credential, so the sandbox needs no direct egress. First pick a model the Inference Gateway serves, then do the one-time operator wiring on the gateway.

```bash
nemo models list --all-pages      # pick a model the gateway serves; it paginates, so --all-pages
MODEL='default/openai-openai-gpt-4o-mini'   # example only; set to a model the list above shows
```

`model_name` MUST be a model that actually exists on this platform's Inference Gateway. A model the gateway cannot resolve makes `nat serve` exit inside the sandbox, which the reconciler reports as `FAILED` with the serve log tail rather than a `READY` deployment that 502s. Prefer a gpt-4o-mini-class model for clean ReAct tool-calling; avoid verbose reasoning models that emit empty content.

Register the platform Inference Gateway as an `openshell` provider and bind `inference.local` to it (once per gateway). `host.openshell.internal` is the gateway's alias for its own host, where the platform listens. The credential is a dummy: the gateway injects the real one.

```bash
openshell provider create --name nemo-igw --type openai \
  --credential OPENAI_API_KEY=empty \
  --config OPENAI_BASE_URL=http://host.openshell.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1 \
  || echo "(provider may already exist; use 'openshell provider update' to change its config)"
openshell inference set --provider nemo-igw --model "$MODEL"
openshell inference get            # verifies the route gateway-side
```

`inference set` verifies the endpoint gateway-side, so a clean `inference get` is your confirmation the route is live before you deploy.

## Step 2: Package the agent as a sandbox image

Use the checked-in demo config `plugins/nemo-deployments/examples/openshell/agent/config.yaml`. It is a normal NAT ReAct config that is already sandbox-ready: its LLM points at `https://inference.local/v1`, the `general.telemetry` / `nemo_files` tracer block is removed, and the model id is left as a deploy-time param. Its `agent/` folder holds only `config.yaml`, so it is a clean Docker build context.

Two properties of this config are mandatory for the sandbox path, so preserve them if you start from your own agent instead:

- **LLM points at `inference.local`.** `base_url: https://inference.local/v1`, and `api_key` is a non-empty placeholder (`not-used`): it MUST be non-empty because the gateway swaps in the real credential, but an empty string makes NAT reject the config.
- **No `general.telemetry` / `nemo_files` block.** The stock `react-agent.yml` example enables the `nemo_files` tracing plugin, which is NOT installed in the packaged sandbox image, so `nat serve` rejects the config and exits, and the deployment lands in `FAILED` with the config error in its status message. Remove the whole `general.telemetry` block before packaging. (The demo config already has it removed.)

The `wiki` tool is registered in the demo config but its egress to Wikipedia is blocked under the zero-egress policy, so only the `current_datetime` path is exercised below. Trim the `wiki` tool from `tool_names`/`functions` if you want no dead tools.

If you start from your own agent, copy it into its own clean directory named `config.yaml` and edit the copy (not the repo file):

```bash
mkdir -p /tmp/igw-agent
cp <your-agent>.yml /tmp/igw-agent/config.yaml
# then: set base_url: https://inference.local/v1, keep api_key non-empty,
# remove the general.telemetry block, leave model_name as ${NEMO_DEFAULT_MODEL}
```

The clean directory and the `config.yaml` name both matter (verified the hard way): `nemo agents package` uses the agent file's **parent directory as the Docker build context**, so a shared dir like bare `/tmp` can hold a file that breaks `docker buildx` (`failed to unmarshal ... invalid UTF-8`); and it bakes the file into the image at **`/workspace/<basename>`**, so naming it `config.yaml` makes it land at `/workspace/config.yaml`, the exact path Step 3's serve command loads. On a name mismatch `nat serve` never finds its config and exits, and the deployment reports `FAILED` with that error rather than serving.

Build the image with the OpenShell runtime profile (adds the `sandbox` user and required packages). Use the shipped demo config:

```bash
nemo agents package \
  --agent  plugins/nemo-deployments/examples/openshell/agent/config.yaml \
  --nat-version 1.8.0 \
  --sandbox-runtime openshell \
  --tag    nemo-agent-igw:test
```

The config is baked into the image at `/workspace/config.yaml`, which is exactly what Step 3's serve command loads. The model id is injected at deploy time via `nat serve --override llms.llm.model_name <MODEL>` (Step 3), so the unset `${NEMO_DEFAULT_MODEL}` in the baked config is fine.

Verification: `docker image inspect nemo-agent-igw:test >/dev/null 2>&1 && echo IMAGE_OK || echo IMAGE_MISSING`. If `IMAGE_MISSING`, the build failed; surface the `nemo agents package` output and stop.

## Step 3: Create the DeploymentConfig

Describe the container: the image, the `nat serve` command the backend runs inside the sandbox (with the model injected via `--override`), and the port. The agent reaches the LLM through `inference.local`, so there is no proxy handling or egress plumbing here.

```bash
curl -sf -X POST "$BASE/deployment-configs" -H 'content-type: application/json' -d "{
  \"name\": \"igw-agent-cfg\",
  \"containers\": [{
    \"name\": \"agent\",
    \"image\": \"nemo-agent-igw:test\",
    \"command\": [\"/workspace/.venv/bin/nat\",\"serve\",\"--config_file\",\"/workspace/config.yaml\",\"--override\",\"llms.llm.model_name\",\"$MODEL\",\"--host\",\"0.0.0.0\",\"--port\",\"9000\"],
    \"ports\": [{\"containerPort\": 9000, \"name\": \"http\"}]
  }]
}" | jq .
```

`$MODEL` here MUST be the same id you passed to `openshell inference set` in Step 1, so the model the agent requests matches the route the gateway forces.

## Step 4: Deploy (executor = openshell-local)

Sandbox is opt-in: name the executor explicitly (`"executor": "openshell-local"`). The default executor is `local-docker`, so ordinary deployments and the e2e harness keep their normal docker/k8s path.

```bash
curl -sf -X POST "$BASE/deployments" -H 'content-type: application/json' -d '{
  "name": "igw-agent",
  "deployment_config": "igw-agent-cfg",
  "executor": "openshell-local"
}' | jq .
```

Behind the scenes the `OpenShellDeploymentBackend` generates a pure default-deny SandboxPolicy (read-only `/opt` for the interpreter; read-write `/home/sandbox`, `/tmp`, `/dev/shm` for the Dask runtime `nat serve` spins up; `run_as_user: sandbox`; and, because `platform_egress` is null, no network egress rules at all), creates the sandbox from the image and applies the policy, runs the serve command in the sandbox-writable `serve_workdir` (`/home/sandbox`), and exposes the service through the gateway. The agent's model calls leave the sandbox only through the gateway-routed `inference.local` path.

## Step 5: Wait for READY and read the endpoint

The reconciler drives `PENDING -> STARTING -> READY`, typically well under a minute.

```bash
for i in $(seq 1 30); do
  status=$(curl -sf "$BASE/deployments/igw-agent" | jq -r '.status')
  echo "$status"
  [ "$status" = "READY" ] && break
  [ "$status" = "FAILED" ] && { echo "DEPLOY_FAILED"; break; }
  sleep 3
done
URL=$(curl -sf "$BASE/deployments/igw-agent" | jq -r '.endpoints[0].url')
echo "$URL"      # e.g. http://<sandbox>--<svc>.openshell.localhost:17670/
```

If the status reaches `FAILED` or never leaves `PENDING`/`STARTING`, jump to the recovery table below. Do not proceed to invoke until `READY`. `READY` gates on the sandbox being up, the serve launcher exiting cleanly, the serve process still being alive on the poll, and `nat serve` answering a reachability probe on its port, so the endpoint is serving as soon as it reports `READY`. A serve process that exits (bad config, unresolvable model) flips the deployment to `FAILED` on the next poll, with the tail of its log in the status message.

## Step 6: Invoke the agent

Ask a question that forces both a tool call and an LLM call. The agent's LLM call goes to `inference.local`, which the gateway routes to the platform Inference Gateway, so the sandbox makes no direct egress.

```bash
curl -sf -X POST "${URL}generate" -H 'content-type: application/json' \
  -d '{"input_message":"What is the current date and time? Use your current_datetime tool"}' | jq .
# -> {"value":"The current date and time is 2026-... +0000."}
```

Other routes the served workflow exposes: `POST /v1/chat/completions`, `/chat`, `/v1/workflow`, and `/generate/stream`.

Verification: the response must contain a non-empty `value`. A `502 Service endpoint is not reachable` means `nat serve` (port 9000) is not answering. Read the serve log inside the sandbox:

```bash
SBX=$(curl -sf "$BASE/deployments/igw-agent" | jq -r '.endpoints[0].url' | sed -E 's#^https?://##; s#--.*##')
openshell sandbox exec --name "$SBX" -- cat /tmp/nemo-serve.log
```

The most common root causes are: the `MODEL` is not served on your Inference Gateway (`nemo models list --all-pages`), the config did not bake to `/workspace/config.yaml`, or a leftover `general.telemetry` block made NAT reject the config. An empty or error `value` means the LLM call failed at the gateway or the model misbehaved. Either way, check the recovery table.

## Step 7: Prove the sandbox has zero direct egress

The security punchline. Show the sandbox can reach nothing on the network directly:

```bash
# the sandbox name is nmp-<hash>; derive it from the endpoint URL (or read `openshell sandbox list`)
SBX=$(curl -sf "$BASE/deployments/igw-agent" | jq -r '.endpoints[0].url' | sed -E 's#^https?://##; s#--.*##')
openshell sandbox exec --name "$SBX" -- curl -sS -m 5 https://example.com   # BLOCKED (policy default-deny)
# a NONZERO exit from the inner curl is the PASS
# meanwhile Step 6's LLM call still worked: it went via inference.local (gateway-routed, not sandbox egress)
```

## Step 8: Clean up

```bash
curl -sf -X DELETE "$BASE/deployments/igw-agent"          # tears down the sandbox
curl -sf -X DELETE "$BASE/deployment-configs/igw-agent-cfg"
```

Tear down the gateway too when you are done:

```bash
docker compose -f plugins/nemo-deployments/examples/openshell/docker-compose.yml down
```

## Gotchas (all verified against a live gateway)

- **The `inference.local` route must be wired before invoke.** The agent hits `https://inference.local/v1`; if you never ran `openshell provider create` + `openshell inference set` (Step 1), the gateway has no route and `nat serve`'s model calls fail. `openshell inference get` should show the `nemo-igw` provider and your model.
- **Remove `general.telemetry` / `nemo_files` before packaging.** The stock `react-agent.yml` enables the `nemo_files` tracer, which is not installed in the sandbox image, so `nat serve` rejects the config and the deployment goes `FAILED`. The shipped `agent/config.yaml` already removes it.
- **Keep `api_key` non-empty.** `base_url: https://inference.local/v1` with an empty `api_key` makes NAT reject the config. Use a placeholder like `not-used`; the gateway injects the real credential.
- **Serve workdir must be sandbox-writable.** The image owns `/workspace` as its `agent` user, which the `sandbox` user cannot write. `serve_workdir` is `/home/sandbox`; the policy grants read-write on `/home/sandbox`, `/tmp`, and `/dev/shm` (Dask needs POSIX semaphores under `/dev/shm`).
- **Model choice matters for ReAct, and the model must exist on the Inference Gateway.** Use a `model_name` your `nemo models list --all-pages` actually shows; switchyard names drift between platforms and a missing one makes `nat serve` exit (endpoint 502). A gpt-4o-mini-class model gives clean tool-calling; verbose reasoning models can emit empty content that breaks the ReAct loop.
- **The `wiki_search` tool cannot reach Wikipedia under the zero-egress policy.** It stays registered in the demo config, but its external egress is blocked, so only the `current_datetime` path is exercised. Trim `wiki` from `tool_names`/`functions` if you want no dead tools.
- **Platform must bind 0.0.0.0.** The `inference.local` route is dialed from the sandbox, not from the gateway process: the docker driver writes a literal `172.18.0.1 host.openshell.internal` into each sandbox's `/etc/hosts` (the sandbox network's gateway address), so the platform has to be listening there. On `--host 127.0.0.1` every model call fails with a 503. `0.0.0.0` puts an unauthenticated dev platform on your LAN; do not do it on untrusted wifi.
- **`openshell inference set` validates from the host, not from the sandbox.** It prints `Validated Endpoints` whenever the CLI or gateway can reach the URL, which is true even for a loopback-only platform. A green validation followed by a 503 at invoke is exactly that case: the route is fine, the platform is not reachable at the sandbox network's gateway address.
- **Docker sandboxes need gateway-minted JWTs.** If the gateway logs complain about missing signing keys, run the one-time `generate-certs` command in Pre-flight step 1 before `docker compose up`.
- **`sandbox exec` rejects newlines in args.** To push a file into a sandbox by hand, base64-encode it (`echo <b64> | base64 -d > f`) or use `sandbox create --upload <file>`.
- **Endpoint host is docker-driver specific.** The exposed `*.openshell.localhost:17670` URL and the sandbox's `host.openshell.internal` hop to the platform both assume the docker driver.

## If verification fails

| Symptom | Cause | Recovery |
| --- | --- | --- |
| `openshell gateway list` / `sandbox list` cannot reach the gateway | Gateway not running | Start the docker-driver gateway (Pre-flight step 1: `generate-certs` once, then `docker compose up -d`); re-check |
| Gateway logs about missing JWT signing keys | `generate-certs` never run | Run the one-time `generate-certs` command (Pre-flight step 1), then restart the gateway |
| `health/ready` not `ready` | Platform down | Route to `nemo-setup`, return when ready |
| Deploy `FAILED` with an unknown/unregistered executor error | Platform started without the executor config | Restart with `nemo services run --host 0.0.0.0 --port 8080 --config packages/nmp_platform/config/local.yaml` (the bundled default config has no openshell executor) |
| `openshell gateway list` shows the endpoint on `:8080` | Gateway collides with the platform port | Recreate the gateway on `:17670` (docker driver, plaintext) per Pre-flight step 1; `:8080` belongs to the platform |
| Deploy stuck `PENDING`/`STARTING` | Image missing sandbox user, or serve command wrong | `openshell sandbox list`; inspect the sandbox; confirm Step 2 used `--sandbox-runtime openshell` |
| Deploy `FAILED` | Policy or gateway rejected the sandbox, the serve launcher exited non-zero, or the serve process died after launch | Check platform logs (`nemo services logs -n 100`) and the deployments reconciler output; the FAILED status message carries the launcher exit detail or the serve log tail. `nemo deployments logs <name>` returns the workload's own log |
| Invoke returns `502 Service endpoint is not reachable` | The deployment is not `READY`, or `nat serve` died after coming up | Wait for `READY` before invoking; if a `READY` deployment still 502s, read the serve log (`nemo deployments logs <name>`, or `openshell sandbox exec --name <nmp-hash> -- cat /tmp/nemo-serve.log`). A bad config path, a leftover `general.telemetry` block, or an unresolvable `model_name` surfaces as `FAILED`, not a 502 |
| Invoke returns 503 `inference service unavailable` (in the response or the serve log) | The `inference.local` route resolved, but the gateway's upstream hop to the platform failed | Almost always a platform bound to `127.0.0.1`; restart it with `--host 0.0.0.0`. Confirm the address the sandbox actually dials with `openshell sandbox exec --name <nmp-hash> -- cat /etc/hosts` (look for `host.openshell.internal`) and check the platform answers there (`curl -sf http://<that-ip>:8080/health/ready`). A clean `openshell inference set` does NOT rule this out: it validates from the host |
| Invoke returns empty/error `value`, or serve log shows connection refused to `inference.local` | `inference.local` route not wired, or model misbehaved | Confirm `openshell inference get` shows the `nemo-igw` provider + your model; re-run Step 1's `provider create` / `inference set`; try a gpt-4o-mini-class model |
| `example.com` reachable in Step 7 | Egress policy not applied | Confirm executor is `openshell-local` and the generated default-deny policy is attached to the sandbox |
| `ModuleNotFoundError: openshell` at deploy | workspace extra not installed | `uv sync --package nemo-deployments-plugin --extra openshell` |
| `python-on-whales` missing at package | stale env predating the move of the packaging deps into `nemo-agents-plugin` | `uv sync` |

Do not claim the deployment succeeded until Step 6 prints a non-empty `value`.

## Alternate: direct-egress (agent -> host.docker.internal:8080)

The shipped design routes model traffic through `inference.local` and grants the sandbox no direct egress. If you deliberately want the older direct-egress path instead (the agent calls the Inference Gateway itself at `host.docker.internal:8080`), you must **edit** `packages/nmp_platform/config/local.yaml` to set `platform_egress` on the executor (it is NOT the shipped default):

```yaml
config:
  gateway_endpoint: http://127.0.0.1:17670
  serve_workdir: /home/sandbox
  platform_egress:
    host: host.docker.internal
    port: 8080
    binaries: [/workspace/.venv/bin/python3.13]   # the venv python opens the socket, NOT curl
```

Then the sandbox's generated policy has a single egress rule to the Inference Gateway. In this branch you also skip Step 1's `inference.local` wiring and instead point the agent config's `base_url` directly at the Inference Gateway: `http://host.docker.internal:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1`. Note: the connecting binary must be on the egress allowlist (the venv/uv python that makes the LLM call, not curl); a hand-written override that only lists curl silently blocks `nat`. Restart the platform with `--config` after editing so the change takes effect.

## Related skills

- `nemo-build-agent`: scaffold, validate, and build the NAT agent image before deploying it here.
- `nemo-try-agent`: query an already-deployed agent.
- `nemo-status`: read-only health dashboard for platform, agents, providers, and models.
- `inference`: register providers and virtual models the agent's Inference Gateway model name resolves to.

## Known limitations

- The NAT config is baked into the image; deploy-time config injection (`config_files`) is a follow-up.
- `inference.local` requires the one-time operator wiring (provider + `inference set`) per gateway; the deployment backend does not auto-register it yet.
- The exposed endpoint host and the sandbox's `host.openshell.internal` hop to the platform are docker-driver specific.
- The verified end-to-end flow (agent packaged, deployed through `openshell-local` to READY, invoke returning a real `value` via `inference.local`, direct egress to `example.com` blocked) depends on the `openshell-local` executor wiring landing in your platform config (loaded via `--config`) and the `inference.local` route being wired on the gateway.
