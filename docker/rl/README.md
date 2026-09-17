<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo-RL training images (`nmp-rl-base`, `nmp-rl-training`)

GPU images for NeMo Platform's RL customization — **DPO** and **GRPO** — plus the
**NeMo-Gym** environment runtime. A single training image serves both algorithms.

The images are built **from source** on NVIDIA's `cuda-dl-base` (CUDA 13, Python
3.13), rather than layered on a prebuilt PyTorch container. This README records the
decisions that shape the build so the non-obvious constraints don't have to be
rediscovered.

## Image graph

| Dockerfile | Image | Role |
|---|---|---|
| `Dockerfile.nmp-rl-base` | `nmp-rl-base` | Heavy base. Clones NeMo-RL (with submodules), warms the uv cache for the extras we need (`vllm`, `fsdp`, `modelopt`, `nemo_gym`), then **prefetches the per-worker Ray venvs**. All one-time CUDA compiles (mamba-ssm, causal-conv1d, deep_ep, deep_gemm) happen here. |
| `Dockerfile.nmp-rl-training` | `nmp-rl-training` | Thin layer on the base. Adds the pure-Python platform glue editably; entrypoint runs `python -m nmp.rl.tasks.training` (Ray bootstrap → DPO). Also the Gym environment runtime. A `smoke-test` stage runs CPU-only import checks during the build. |

`docker-bake.hcl` wires them: `nmp-rl-training`'s base context defaults to building
`nmp-rl-base` as a dependency, unless `USE_PREBUILT_BASES` / `RL_BASE_CONTEXT` point
at an already-built base image.

## How NeMo-RL runs: the base venv is not where training happens

This is the single most important thing to understand about this image. Nearly every
surprising thing about the build follows from it.

### Background: Ray workers are separate processes

A NeMo-RL job is not one program. Ray splits it into:

- **one driver** — the process started by the entrypoint (`python -m nmp.rl.tasks.training`).
  It reads config and orchestrates.
- **many workers** — *separate operating-system processes*, usually on other nodes, each
  typically owning one GPU. Ray launches each one by literally running a `python` command
  on the target node; our code is then loaded into that process.

An **actor** is a Python object living inside one of those worker processes
(`DTensorPolicyWorker`, `VllmGenerationWorker`, …).

```text
  DRIVER process (node A)  ── orchestrates
     │           │           │
  ┌──▼───┐   ┌───▼──┐   ┌────▼─┐    separate OS processes,
  │worker│   │worker│   │worker│    often on other nodes
  │GPU 0 │   │GPU 1 │   │GPU 2 │
  └──────┘   └──────┘   └──────┘
```

By default Ray starts every worker with the **same** interpreter as the driver. NeMo-RL
cannot do that: its backends are mutually exclusive. RL's `pyproject.toml` declares an
explicit `conflicts` matrix (`vllm` vs `mcore`, `automodel` vs `vllm`, …) because they
require incompatible `transformers` / numpy / CUDA pins. **No single environment can serve
both a vLLM generation worker and a Megatron training worker.**

### `py_executable`: one interpreter per actor

Ray supports a per-actor **`runtime_env`**, whose `py_executable` key sets *the command used
to launch that worker process*. NeMo-RL uses it to give each actor its own environment.

Definitions live in `nemo_rl/distributed/virtual_cluster.py`; the actor and its executable mapping
are in `nemo_rl/distributed/ray_actor_environment_registry.py`:

| py_executable | Definition | Where it lives |
|---|---|---|
| `SYSTEM` | `sys.executable` | the **base venv** `/opt/nemo_rl_venv` |
| `FSDP` | `uv run --locked --extra fsdp …` | per-node venv under `/opt/ray_venvs` |
| `VLLM` | `uv run --locked --extra vllm …` | per-node venv under `/opt/ray_venvs` |
| `NEMO_GYM` | `uv run --locked --extra nemo_gym …` | per-node venv under `/opt/ray_venvs` |
| `AUTOMODEL` / `MCORE` / `SGLANG` / `TRTLLM` | `uv run --locked --extra <x> …` | per-node venv under `/opt/ray_venvs` |

Note the two shapes: `SYSTEM` is a real interpreter **path**, while the others are
**commands** — `uv run --extra vllm … python` means "make an environment with the vllm extra,
then run python in it".

A `uv …` value is a *recipe*, not something handed to Ray directly. `worker_groups.py`
materializes it first, and only the resulting concrete interpreter path is passed on:

```text
registry: actor FQN ──► "uv run --locked --extra vllm --directory /opt/nemo-rl"   (RECIPE)
                              │
                   create_local_venv_on_each_node()      (executed once per NODE)
                              │
                              ▼
              /opt/ray_venvs/<actor-fqn>/bin/python                      (REAL interpreter)
                              │
                   runtime_env = {"py_executable": …, "env_vars": {VIRTUAL_ENV: …}}
                              │
                              ▼
              Ray launches the worker process with THAT python
```

`make_actor_runtime_env` (`nemo_rl/utils/venvs.py`) also sets `VIRTUAL_ENV` /
`UV_PROJECT_ENVIRONMENT` in the worker's env, so libraries that shell out to subprocesses
resolve the same interpreter as their parent.

### Consequences that are easy to get wrong

- **Installing a package into the base venv does not affect training.** The base venv is the
  *driver's* interpreter; policy and generation workers launch with a different
  `py_executable` and never import from it. Only "system Python" actors (math / code / VLM
  environments) use it.
- **`uv sync --extra X` in the Dockerfile only warms the uv cache.** It does not produce the
  venv Ray launches — `create_local_venv` does, at runtime, unless we prebuild it. Worse,
  `uv sync` runs in *exact* mode, so each `--extra` sync prunes the previous one: the base
  venv ends up holding **default dependencies only**.
- **Venvs are keyed by actor FQN, not by extra.** Two actors sharing `--extra vllm` still get
  two separate venv directories — which is why prefetching several vLLM-tier actors
  duplicates a multi-GB stack — which is why venvs are symlinked into a shared uv cache
  rather than copied (see "Link mode" below).
- **Per node, not per worker.** Eight GPU workers on one node share one venv on that node's
  disk; `venvs.py` uses a `STARTED_ENV_BUILDER` lock so it is built once, not eight times.
- **Different workers can have different GPU requirements.** DPO's worker launches with the
  `fsdp` venv (no `deep_ep`); GRPO's generation worker with the 
  `vllm` venv (`deep_ep`, Hopper-only) — in the same image.
- `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` collapses every actor into the base venv. We do **not**
  set it.

### Which extras DPO and GRPO actually use

| Algorithm | Actor | Extra / venv | Notable contents |
|---|---|---|---|
| DPO | `DTensorPolicyWorker` (policy training) | **`fsdp`** | `flash-attn` (prebuilt multi-arch wheel), `mamba-ssm`, `causal-conv1d` |
| GRPO | `DTensorPolicyWorker` (policy training) | **`fsdp`** | as above — plain full-weight GRPO |
| GRPO | `DTensorPolicyWorkerV2` (policy training) | **`automodel`** | `nemo-automodel`, Transformer-Engine, `megatron-fsdp` |
| GRPO | `MegatronPolicyWorker` (policy training) | **`mcore`** | `megatron-bridge`, `megatron-core`, Transformer-Engine |
| GRPO | `VllmGenerationWorker`, `SyncRolloutActor` | **`vllm`** | `vllm`, `deep_ep`, `deep_gemm`, `flashinfer` |
| GRPO (Gym) | `NemoGym` | **`nemo_gym`** | NeMo-Gym workspace member |

GRPO picks its worker from `training.policy_backend`, which the compiler maps straight onto
`dtensor_cfg._v2`: `automodel` (the default) → **V2** and the `automodel` extra, `dtensor` →
**V1** and the `fsdp` extra. Nothing promotes the backend implicitly — a job that pairs
`policy_backend: dtensor` with LoRA, expert parallelism or `automodel_kwargs` is rejected in
`GRPOTraining` before it is scheduled, since those three are V2-only.

DPO does not expose the knob: `dpo_config` writes its `dtensor_cfg` as a literal with no
`_v2` key, so DPO is always **V1** and the `fsdp` extra.

**LoRA requires V2 or Megatron.** DTensor V1 asserts `lora_cfg.enabled is False`
(`nemo_rl/models/policy/lm_policy.py`); the DTensor LoRA implementation lives in
`nemo_automodel.components._peft.lora`, and Megatron carries its own `peft` path. So
`automodel` and `mcore` are the parameter-efficient-training tiers in this image.

GRPO is **critic-free** — no value model — so neither `DTensorValueWorkerV2` (automodel tier)
nor `MegatronValueWorker` (mcore tier) is prefetched.

### Why the `mcore` backend is built

`mcore` is a second GRPO **training** tier alongside `automodel`. It is what backs the
capabilities DTensor does not implement in NeMo-RL today:

| Capability | DTensor (`fsdp` / `automodel`) | Megatron (`mcore`) |
|---|---|---|
| Pipeline parallelism | ❌ `lm_policy.py` reads `pp_size` only from `megatron_cfg` | ✅ |
| FP8 training / FP8 rollouts / FP8 KV-cache | ❌ | ✅ |
| NVFP4 quantization-aware RL | ❌ | ✅ (with `modelopt`) |
| Draft models / EAGLE3 speculative decoding | ❌ hard-gated in `lm_policy.py` | ✅ |
| Megatron-native generation (no refit conversion) | ❌ | ✅ |
| Largest models (`qwen3.5-397ba17b`, `glm5.1`, DeepSeek-V3) | ❌ tops out ~120B MoE | ✅ |
| MoE with expert parallelism, context parallel, LoRA, VLM | ✅ | ✅ |

Upstream weights this way too: of NeMo-RL's seven recipes that reference `nemo_gym`, six run
on Megatron.

Megatron-native generation needs no separate venv — it runs **in-process inside
`MegatronPolicyWorker`**, which is the point of it (training and inference share one weight
layout, so there is no refit conversion). That is why it has no registry entry of its own.

`mcore` also requires `TORCH_CUDA_ARCH_LIST` to be set at **runtime**, not just at build
(`lm_policy.py` raises without it; mcore's inference unified-memory API calls a torch API
that reads it). The publish stage is `FROM builder`, so it inherits the builder's
`TORCH_CUDA_ARCH_LIST="9.0 10.0"` — the same value NeMo-RL's own `docker/Dockerfile` sets.

**Transformer-Engine is shared between `automodel` and `mcore`.** TE is a
training-time *transformer layer* library (fused attention / LayerNorm / GEMM kernels,
fp8). Whether a backend needs it comes down to **who implements the transformer layer**:

| Extra | Where its transformer layers come from | Needs TE? |
|---|---|---|
| `fsdp` | Stock **HuggingFace** modules, sharded by PyTorch DTensor/FSDP2; attention via flash-attn or SDPA | ❌ nothing to accelerate — plain HF/PyTorch |
| `automodel` | **nemo-automodel** rebuilds layers on TE | ✅ |
| `mcore` | **Megatron-Core** builds its parallel layers on TE primitives | ✅ |
| `vllm` | Its own hand-written **inference** kernels (paged attention, fused MoE) | ❌ inference engine, not a training-layer library |

TE is the single longest CUDA compile in the image, and `automodel` and `mcore` **share one
build of it**. RL's `[tool.uv] override-dependencies` collapses every TE requirement —
`automodel`'s declared `v2.14.1` and `megatron-bridge[te]`'s own rev — onto
`git+…/TransformerEngine.git@release_v2.15`, so `uv.lock` holds exactly one resolved
`transformer-engine 2.15.0+42b8400`. Conflicting extras resolve in separate forks, but uv's
built-wheel cache is keyed on the resolved source identity (git URL + commit + platform
tags), not on which extra requested it. The `mcore` sync therefore reuses the wheel the
`automodel` sync built. `deep_ep`, `mamba-ssm` and `causal-conv1d` are pinned identically
across the extras and are reused the same way.

TE is built from source, once per image build: `uv.lock` pins it as a **git source**, so
`uv sync --frozen` compiles it and caches the resulting wheel, and every later venv reuses
that. A prebuilt wheel would not be picked up without changing how the lock sources TE.

**What `mcore` actually adds:** `megatron-bridge` and `megatron-core` are **editable path
sources** from submodules already on disk, so installing them is near-free. The rest is
wheel-only downloads (`flashinfer-*==0.6.8.post1`, `nvshmem4py-cu13`, `cupy-cuda13x`) plus
`nvidia-modelopt` from git. Build-time cost is small; the real cost is one more prefetched
worker venv, which is the only place an extra contributes to image size (`uv sync` is exact,
so the warmup syncs do not accumulate).

Megatron-Bridge pins `setuptools<80.0.0` for its own build, so uv builds its metadata in a
PEP 517 environment against an old setuptools and leaves it in the shipped cache. RL's
`setuptools>=80.10.2` override cannot reach a build environment, so the publish stage drops
cached setuptools archives that no venv symlinks into.

### NeMo-Gym environments: a second, separate venv layer

Do not confuse the two Gym-related venvs:

1. The **`NEMO_GYM` Ray-actor venv** (`/opt/ray_venvs`, from RL's `nemo_gym` extra) — the
   actor that hosts Gym. Prefetched at build time with the other worker venvs.
2. The **per-environment venvs** (`/opt/gym_venvs`, `NEMO_GYM_VENV_DIR`) — **one venv per
   Gym server directory**, created by Gym's own setup step from that directory's
   `pyproject.toml` or `requirements.txt`. This keeps an environment's deps (rdkit, spacy,
   numpy 2.x) isolated from RL's pinned stack.

Two sources of environments, with different timing:

| Environment source | Venv built | Cost |
|---|---|---|
| **Built into the Gym repo** (math_with_judge, code_gen, swe_agents, …) | At **runtime**, on first spin-up — prefetch is off by default (`NEMO_GYM_PREFETCH_CONFIGS` is empty) | Paid per job/node, until you opt back in |
| **User-supplied env FileSet** (downloaded per job) | At **runtime**, on first spin-up | Paid per job/node — unavoidable, the env isn't known at build time |

**Prefetch is disabled by default.**
To bake a set back in, set `NEMO_GYM_PREFETCH_CONFIGS` to a space-separated list of config
paths — NeMo-RL's `examples/nemo_gym/prefetch_super_all_envs.yaml` is the curated union of the
environments it uses across its RLVR / SWE / RLHF stages, purpose-built for prefetching and
maintained in step with the Gym pin. That trades build time and image size for job startup
latency.

User environments therefore *do* add startup time, and cannot be prebaked. Two things bound it:

- The packaging format matters, but only via wheelhouse completeness. There are two
  installs: Gym's per-server venv build reads `wheels/` through `UV_FIND_LINKS` (a
  candidate pool — **an index is still enabled**, so a missing distribution is fetched
  from it), and only NeMo-RL's later `install_environment_wheels` uses `--no-index`.
  Both **`wheels-v1` and `adapter-wheels-v1` run under deny-default network policy**
  when `wheels/` covers the venv build. For `adapter-wheels-v1`, the converter resolves
  the environment and image-bundled agent harness together, builds source distributions
  into wheels, and fails rather than emitting an incomplete closure. The image harness
  pins `verifiers` by version so uv can select that vendored wheel instead of following
  a Git URL. `native-v1` vendors nothing and always needs egress.
- Platform bootstrap for all three formats lives in
  `nmp.rl.tasks.environment.bootstrap.bootstrap_environment_package` (validators +
  offline wheel install). The Gym host / RL image entrypoint should call that —
  not upstream NeMo-RL format APIs.
- Gym reuses a shared **uv cache** (`uv_cache_dir`) and skips venv creation when one
  already exists, so repeated jobs on the same node re-pay much less.

Prefetching built-in envs and installing user envs at runtime are complementary: the
first removes startup cost for environments we ship, the second is the price of letting
users bring their own.

### Venv map: what exists, and who uses it

```text
IMAGE (built once)                                RUNTIME
──────────────────                                ───────

/opt/nemo_rl_venv          base venv
  default deps ONLY  ───────────────────────────> Ray driver + "system python" actors
  (each `uv sync --extra` is pruned by             (math / code / VLM environments)
   the next; extras never persist here)

/opt/ray_venvs/<actor-fqn>   per-ACTOR venvs, prefetched at build (nine)
  ├─ …DTensorPolicyWorker       [fsdp]      ────> policy training      (DPO)
  ├─ …DTensorPolicyWorkerV2     [automodel] ────> policy training      (GRPO, DTensor V2)
  ├─ …MegatronPolicyWorker      [mcore]     ────> policy training      (GRPO, Megatron;
  │                                                also hosts Megatron-native generation)
  ├─ …VllmGenerationWorker      [vllm]      ────> generation           (GRPO, sync)
  ├─ …VllmAsyncGenerationWorker [vllm]      ────> generation           (GRPO, async — Gym
  │                                                forces async rollouts)
  ├─ …SyncRolloutActor          [vllm]      ────> rollout driver       (GRPO, sync path)
  ├─ …NemoGym                   [nemo_gym]  ────> Gym actor            (mode A, colocated)
  ├─ …SandboxedGymActor         [nemo_gym]  ────> Gym proxy actor      (mode B, sandboxed)
  └─ …SandboxEpisodeBrokerActor [nemo_gym]  ────> per-episode sandbox broker (mode B)

/opt/gym_venvs/<env>         per-ENVIRONMENT venvs — empty in the shipped image
  ├─ built-in Gym envs   ── created at RUNTIME (prefetch off; opt in via
  │                         NEMO_GYM_PREFETCH_CONFIGS to bake them at build)
  └─ user FileSet envs   ── created at RUNTIME from the env's pyproject/requirements
```

**DPO job** — one backend, no generation, no Gym:

```text
driver (base venv)
  └─ Policy ─ DTensorPolicyWorker ×N   →  /opt/ray_venvs/…DTensorPolicyWorker   [fsdp]
                └─ reference model lives INSIDE these same workers
                   (init_reference_model=True — no extra worker, no extra venv)
```

**GRPO job, DTensor V2 path** — training + generation + (optionally) Gym.
This is the path the customizer's GRPO compiler emits (`policy.dtensor_cfg._v2: true`,
`megatron_cfg.enabled: false`); see below for the Megatron alternative.

```text
driver (base venv)
  ├─ Policy      ─ DTensorPolicyWorkerV2 ×N → /opt/ray_venvs/…DTensorPolicyWorkerV2 [automodel]
  ├─ Generation  ─ VllmGenerationWorker ×N →  /opt/ray_venvs/…VllmGenerationWorker [vllm]
  │                                            └─ deep_ep / deep_gemm  → Hopper+ only
  ├─ Rollout     ─ SyncRolloutActor        →  /opt/ray_venvs/…SyncRolloutActor     [vllm]
  └─ Env         ─ mode A: NemoGym         →  /opt/ray_venvs/…NemoGym          [nemo_gym]
                   mode B: SandboxedGymActor + SandboxEpisodeBrokerActor
                                           →  /opt/ray_venvs/…SandboxedGymActor [nemo_gym]
                                              /opt/ray_venvs/…BrokerActor       [nemo_gym]
                     └─ Gym stack + user FileSet run in an isolated OpenSandbox pod
                     └─ per-environment venv →  /opt/gym_venvs/<env>
                          ├─ shipped env  → built on first use (prefetch off by default)
                          └─ user FileSet → built on first use (startup cost;
                                            wheels-v1 avoids PyPI, native-v1 needs egress)
```

**GRPO job, Megatron path** — selected by `policy.megatron_cfg.enabled: true`. Only the policy
tier differs; generation, rollout and Gym are unchanged from the diagram above:

```text
driver (base venv)
  ├─ Policy      ─ MegatronPolicyWorker ×N →  /opt/ray_venvs/…MegatronPolicyWorker [mcore]
  │                 └─ Megatron-native generation runs IN-PROCESS here when
  │                    policy.generation.mcore_generation_config is set — training and
  │                    inference share one weight layout, so there is no refit conversion
  │                    and no separate generation venv
  └─ (generation / rollout / Gym as above)
```

Megatron and DTensor are mutually exclusive — `lm_policy.py` raises if both `megatron_cfg`
and `dtensor_cfg` are enabled.

## Hardware: which GPUs run what

The image bundles both **multi-arch prebuilt wheels** and **source-compiled CUDA
extensions**. Whether a workload runs on A100 (or any pre-Hopper GPU) is decided by
**which kernels the training backend actually launches** — not by the arch a package
it never calls happens to be compiled for.

**DPO runs on A100 and up — for dense transformer models.** The customizer's DPO trains
on the **DTensor (FSDP) backend** with the Megatron backend disabled. That path builds
models from HuggingFace + `torch` + `flash-attn` — all shipped as prebuilt multi-arch
wheels that include `sm_80` — and does **not** use Transformer-Engine or `deep_ep`. So
DPO launches no Hopper-only kernel and runs on A100, Ada (L40S), Hopper, and Blackwell
alike.

One exception: **Mamba / hybrid-SSM models**. The `fsdp` extra also carries `mamba-ssm`
and `causal-conv1d`, which are *source-compiled* and therefore built only for the
architectures in `TORCH_CUDA_ARCH_LIST`. A dense transformer never calls them, but a
Mamba or hybrid model on A100 would fail at first kernel launch with the current
`9.0 10.0` build. Add `8.0` to the arch list to cover that case — unlike `deep_ep`,
these two do compile for `sm_80`.

**GRPO targets Hopper / Blackwell (for now).** GRPO
generation uses vLLM, whose MoE path pulls `deep_ep` (and `deep_gemm` for fp8),
source-compiled CUDA extensions that use Hopper-class features (TMA async copy,
warp specialization, NVSHMEM GPU-initiated RDMA) and only build/run on
**SM 9.0 (Hopper) / 10.0 (Blackwell)**. They do not compile for SM 8.0/8.6/8.9.
Upstream NeMo-RL pins `TORCH_CUDA_ARCH_LIST="9.0 10.0"`;
upstream Automodel builds DeepEP for `"9.0 10.0 12.0"` — the same Hopper floor.

**Megatron backend: Hopper / Blackwell as built.** Transformer-Engine — the heavy
fused-kernel library both the Megatron and Automodel backends depend on — is pinned to
`NVTE_CUDA_ARCHS=90;100`, matching NeMo-RL's own `docker/Dockerfile`. TE is **not**
inherently Hopper-only — upstream Automodel builds it for `80;90;100;120` — so if
Megatron or DTensor V2 is later required on A100, TE can be rebuilt to include `8.0`.
`TORCH_CUDA_ARCH_LIST` is additionally a **runtime** requirement for the Megatron
backend, which raises if it is unset; the publish stage inherits `"9.0 10.0"` from the
builder stage.

**Genuinely Hopper-only pieces:** only `deep_ep` and `deep_gemm` (their source builds
fail for `8.0`). Everything else — the `torch`/`vllm`/`flash-attn`/`flashinfer`
wheels, and TE/`mamba-ssm`/`causal-conv1d`, which *can* target `8.0` — is Hopper-only
today only because of the current pins, not by nature.

### Running the image on A100

- **DPO (DTensor):** works — no Hopper-only kernel is launched.
- **Anything that invokes TE, `deep_ep`, or `deep_gemm`** (Megatron backend, MoE
  expert-parallel, fp8): the `.so` loads, but the first kernel launch fails with
  `no kernel image is available for execution on the device` — a graceful Python
  error, not a crash.
- You cannot build a variant of *this image* whose `deep_ep`/`deep_gemm` run on
  A100 — those extensions have no A100 kernels to compile.

## Source: pinning and submodules

- **NeMo-RL is pinned to a commit SHA** via `NEMO_RL_REF` (see `docker-bake.hcl`), not
  a branch. The base clones RL at that ref. A branch ref would let the source move
  underneath the cache, silently invalidating the heavy compile layer on every
  rebuild and breaking `uv sync --frozen` whenever the pinned commit's lock drifted.
  Bump the SHA deliberately, in lockstep with the lock.
- **NeMo-Gym is not cloned separately.** RL pins Gym as a git submodule and declares
  it a uv **workspace member**, alongside the Automodel and Megatron-Bridge (+ nested
  Megatron-LM) submodules. The base's git `ADD` recurses submodules, so Gym rides in
  with RL and its pin stays in lockstep with RL's `uv.lock`. To iterate on a Gym fork
  without a separate pin, override the whole source with a local checkout via
  `--build-context nemo-rl=<path>`.

## Prebuilt wheels vs. source-compiled — and what the arch flags affect

Two kinds of packages coexist:

- **Prebuilt multi-arch wheels** — `torch`, `vllm`, `flash-attn`, `flashinfer`. These
  ship "fat" binaries covering many GPU archs. `nvcc` is never re-run for them, so the
  arch flags **do not** change what GPUs they support.
- **Source-compiled extensions** — `transformer-engine` (TE), `deep_ep`, `deep_gemm`,
  `mamba-ssm`, `causal-conv1d`. These compile for exactly the archs in
  `TORCH_CUDA_ARCH_LIST` / `NVTE_CUDA_ARCHS`, so they are the only packages the arch
  flags touch — and `deep_ep`/`deep_gemm` are what make GRPO Hopper-only.

### Why the CUDA extensions are not extracted into prebuilt wheels

The source-compiled extensions stay inside RL's `uv sync`, and the build reuses them via
the uv cache + venv prefetch rather than via wheel images:

- **`deep_ep` / `deep_gemm`** are `git+https` pip dependencies (not submodules) that
  `deep_ep` builds against an nvshmem + rdma-core environment; neither produces a
  cleanly relocatable wheel.
- **`mamba-ssm` / `causal-conv1d`** are small, fast compiles, so extraction saves little.
  The repo's existing `mamba-ssm-wheel` / `causal-conv1d-wheel` images are **not**
  reusable here regardless: they are cp311/cp312, CUDA 12.8, and pinned to different
  commits than RL (a cp312/cu128 wheel cannot import on cp313/cu130, and the version
  deltas would fail `uv sync --frozen`). Reusing the pattern would mean new cp313/cu130
  stages pinned to RL's exact commits, kept in lockstep with `uv.lock`.
- **Transformer-Engine** is the longest compile. It comes in with the `automodel` extra,
  which the GRPO policy worker needs, so it is built from source here.
- **CPython is copied, not uv-managed.** RL uses `.python-version` to pin an exact patch
  release. CPython 3.13.15 is copied from `python:3.13.15-slim-bookworm` into `/opt/cpython`
  (uv's catalog has no linux-gnu 3.13.15 build; bookworm's OpenSSL 3.0 matches Ubuntu
  24.04 on cuda-dl-base, while trixie _ssl needs OPENSSL_3.3.0 and does not import).
  `UV_PYTHON=/opt/cpython/bin/python3.13` overrides `.python-version` so worker venvs
  cannot silently stay on 3.13.14.

## Layering for fast CI rebuilds

The heavy compile layer is the expensive part, so the base is structured to reuse it
whenever the *dependency graph* hasn't changed:

- Only the **resolver inputs** (`pyproject.toml`, `uv.lock`, the `3rdparty` workspace
  members, `research/`, and the top-level `nemo_rl` package stub) are copied **before**
  the heavy `uv sync`. A source-only RL bump (Python changed, deps unchanged) is then a
  cache hit on the compile layer; only the cheap editable-install step below re-runs.
- The **full RL source** and the editable root install come **after** the sync.
- The SHA pin keeps the resolver-input layer deterministic, so a warm builder reuses
  the whole compile across rebuilds. A brand-new builder has a cold cache and
  recompiles from scratch.
- To iterate on `services/rl` without re-entering the base build at all: build the base
  once and point training at it with `USE_PREBUILT_BASES=1 BASE_TAG_RL=<tag>` or
  `RL_BASE_CONTEXT=docker-image://<base-image>`.
- Only the extras that are actually used are synced (`fsdp`, `automodel`, `mcore`, `vllm`,
  `modelopt`, `nemo_gym`); `sglang` and `trtllm` are dropped.

### Prefetching the per-worker venvs (build once, not per job)

The warmup `uv sync --extra …` calls populate the **uv cache at `/opt/uv_cache`, which
ships inside the image** — it has to, because the prefetched venvs symlink into it (see
"Link mode" below). Do not confuse it with the `--mount=type=cache` the training image
uses for its editable install, which is build-only and never enters the image. The venvs
training actually runs in are the per-worker ones under `/opt/ray_venvs`, so the base runs
`nemo_rl/utils/prefetch_venvs.py` after the source copy to bake them in — the same
approach NeMo-RL's own release stage uses.

Prefetched (the filters match **actor FQNs**, not extra names):

| Filter | Extra | Needed by |
|---|---|---|
| `dtensor_policy_worker.DTensorPolicyWorker` | `fsdp` | DPO policy training |
| `dtensor_policy_worker_v2.DTensorPolicyWorkerV2` | `automodel` | GRPO policy training — selected by `policy.dtensor_cfg._v2: true`; the only LoRA-capable DTensor worker. The V1 filter does not match it (`dtensor_policy_worker_v2.`) |
| `megatron_policy_worker.MegatronPolicyWorker` | `mcore` | GRPO policy training on Megatron — selected by `policy.megatron_cfg.enabled: true`; also hosts Megatron-native generation in-process. Does not match modelopt's `megatron_quant_policy_worker.MegatronQuantPolicyWorker` |
| `vllm.vllm_worker` | `vllm` | GRPO generation — matches **both** `VllmGenerationWorker` and `VllmAsyncGenerationWorker` (NeMo-Gym forces async rollouts, so both are on the path) |
| `sync_rollout_actor.SyncRolloutActor` | `vllm` | GRPO rollout driver (future, sync path) |
| `nemo_gym.NemoGym` | `nemo_gym` | Gym environment actor (mode A, colocated) |
| `nemo_gym_actor.SandboxedGymActor` | `nemo_gym` | Sandboxed Gym (mode B) — the trusted proxy actor in the training pod |
| `broker_actor.SandboxEpisodeBrokerActor` | `nemo_gym` | Trusted episode broker — creates per-episode sandboxes so the job sandbox never holds the OpenSandbox credential |

Filters are **substring matches on actor FQNs**, so eight filters yield nine venvs
(`vllm.vllm_worker` matches the sync and async workers alike). They are deliberately specific —
a bare `vllm` would also match `nemo_rl.modelopt`'s `vllm_quant_worker` and pull in the
modelopt+vllm combination, and a bare `megatron` would pull in `MegatronValueWorker` and
modelopt's `MegatronQuantPolicyWorker`.

`tests/smoke_gpu/test_rl_training.py` asserts this set exactly
(`test_prefetched_venvs_match_expected_set`), so a filter that drifts fails the build rather
than silently shipping an image whose workers rebuild their venv on the node at job start.

One venv is built **per actor, not per extra** — `prefetch_venvs.py` passes the actor FQN as
the venv name — so the three `nemo_gym`-extra actors above each get their own directory and
each needs its own filter. `opensandbox` / `tenacity` come in through the extra itself, since
RL declares `nemo_gym = ["nemo_gym[sandbox]"]`.

Without this, each venv is built **on the node at first run**, re-resolving and
recompiling `deep_ep` / `mamba-ssm` / `causal-conv1d` against a cold uv cache on every
job. Not prefetched (they build on the node if a config selects them): `sglang`, `trtllm`,
modelopt-quant workers, `DTensorValueWorkerV2` / `MegatronValueWorker` (GRPO is critic-free,
so there is no value model), and the async-GRPO actors (`AsyncTrajectoryCollector`,
`ReplayBuffer`).

### Link mode: why the uv cache ships inside the image

This image materializes roughly 30 venvs (5 per-worker Ray venvs plus the bundled Gym
environment venvs), and several of them overlap heavily — the two vLLM-tier worker venvs
are the *same* multi-GB stack. How uv puts files into a venv therefore dominates image
size:

| `UV_LINK_MODE` | Venv contents | Image cost | Fragile? |
|---|---|---|---|
| `copy` | Real copies of every file | **~29 × the packages** | No — self-contained |
| `symlink` | Symlinks pointing into the uv cache | **1 × the packages** (the cache) + near-free symlinks | Yes — breaks if the cache is missing |

We use **`symlink`**, like NeMo-RL's own release stage. That has one hard requirement: the
symlinks are resolved at *runtime*, so the uv cache must ship **inside the image** and stay
readable by the non-root user. Hence `UV_CACHE_DIR=/opt/uv_cache`:

- It cannot be a BuildKit `--mount=type=cache` — those never enter the image, so every
  symlink would dangle.
- It cannot sit at uv's default `~/.cache/uv`, because `/root` is mode `700` and the
  training image runs as a configured non-root UID (1000 by default, the same reason
  CPython is copied under `/opt/cpython`).
- **`/opt/uv_cache` must never be deleted** — pruning it breaks every venv that points into
  it. The prune step in the publish stage deliberately leaves it alone.

**It does not warm runtime installs in the training image.** `Dockerfile.nmp-rl-training`
repoints `UV_CACHE_DIR` at a writable per-user path, because `/opt/uv_cache` holds the code
the trainer executes and must stay read-only (user-authored Gym environment code runs in
this container). uv reads only `UV_CACHE_DIR` — there is no read-through to a second cache —
and NeMo-Gym resolves its own cache by shelling out to `uv cache dir`
(`nemo_rl/environments/nemo_gym.py`), which honours that same variable. So anything built on
the node at runtime (Gym environment venvs, non-prefetched actors) starts from an **empty**
cache and re-downloads.

That is the main argument for prefetching: a venv that is not baked in is not just
un-materialized, it is rebuilt against a cold cache — which for the `automodel` tier would
mean compiling Transformer-Engine on every node at job time.

#### `vllm/` is a private copy per vLLM venv

NeMo-RL **patches vLLM in place at worker startup** (`nemo_rl/models/generation/vllm/patches.py`
rewrites `v1/executor/ray_executor.py`, `model_executor/models/llama_eagle3.py` and
`tool_parsers/hermes_tool_parser.py`, taking a `<file>.patch_lock` beside each). Symlinking breaks
that in two independent ways, both observed on the shipped image:

1. **Permission** — the prefetched venvs are build-owned/read-only for the configured runtime UID
   (1000 by default), so creating `ray_executor.py.patch_lock` fails with `EACCES` and the
   `VllmAsyncGenerationWorker` actor dies in its creation task.
2. **Sharing** — every vLLM-tier venv symlinks that file to *one* physical copy in `/opt/uv_cache`.
   The patch writes a **per-venv `py_executable`** into it, so two venvs cannot share it. No amount
   of `chown` fixes this; the file must not be shared at all.

So the publish stage replaces `site-packages/vllm/` with a real, private, runtime-owned copy in each
vLLM-tier venv.

The two need separate fixes: **ownership** solves (1), **private copies** solve (2). Running as root
would silence the `EACCES` but leaves the sharing intact — the first venv to patch wins and the next
one launches under another venv's interpreter.

Note this is *not* what upstream NeMo-RL does. Its published image symlinks too (74,567 symlinks vs
925 real files in one vLLM venv, both vLLM venvs sharing one `ray_executor.py`), and it runs as
**root** — so (1) never surfaces there and (2) stays latent, because vLLM 0.25 defaults
`VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` and the V2 executor never reads the patched call site. Running
non-root is required here, so we can inherit neither the root workaround nor that assumption.

## Image size

For operators sizing nodes and registries, and for anyone deciding what to trim. This is a
large image by design: it ships a full training stack plus every venv the workers and
bundled environments need, so nothing has to be resolved or compiled at job start.

**Compressed** (~21.5 GB)
**On disk** (~74 GB)

### The SWE agent setups

The single largest item inside the source tree is the `swe_agents` environment, whose
per-variant setup directories are materialized during the Gym prefetch:

| Path (under `…/Gym/responses_api_agents/swe_agents/`) | Size |
|---|---|
| `swe_openhands_setup` | 5.3 GB |
| `swe_swebench_multilingual_setup` | 194 MB |
| `swe_swebench_setup` | 189 MB |
| `swe_r2e_gym_setup` | 168 MB |
| everything else (app, prompts, tests, data) | ~5 MB |

### Why `NRL_CONTAINER=1` is required

NeMo-RL gates `get_nemo_gym_uv_cache_dir()` on this variable. Without it the helper returns
`None`, NeMo-Gym never learns the shared cache location, and it silently falls back to its
own default **inside the source tree** — producing a multi-GB second copy of packages that
already exist in `/opt/uv_cache` (9 GB in the build that first exposed this). The companion
venv-directory helper is *not* gated the same way, so venvs land correctly while the cache
does not, which makes the problem easy to miss.

The base sets `NRL_CONTAINER=1`, and the build fails loudly if an in-tree Gym cache ever
reappears. The variable additionally enables the frozen-environment wrapper scripts and the
container fingerprint check, so the build also writes `/opt/nemo_rl_container_fingerprint`
(as NeMo-RL's own release stage does) to keep that check meaningful rather than warning on
every import.

### Levers, if size becomes a problem

- **`NEMO_GYM_PREFETCH_CONFIGS`** — already empty by default, which is the single largest
  saving available: the environment venvs are 4.2 GB, and the build step that produces them
  costs far more than that, because resolving 25 environments pulls their dependencies into the
  shared cache and materializes the SWE agent setups (~5.9 GB). Setting it back to a non-empty
  list moves that cost from job-start latency to image size and build time. Note `swe_agents` is
  the set the sandboxed-Gym work uses, and it is also the least reliable to build.
- **`--extra modelopt`** can be dropped from the warmup sync if quantization is not needed.
- The **CUDA devel toolkit** cannot be removed: `deep_ep` JIT-compiles kernels at runtime and
  `native-v1` environments compile dependencies on the node.
- The **uv cache cannot be pruned** — every venv symlinks into it (see "Link mode").

## Build knobs

| Arg | Default | Purpose |
|---|---|---|
| `NEMO_RL_REPO` / `NEMO_RL_REF` | see `docker-bake.hcl` | NeMo-RL source + pinned commit. |
| `TORCH_CUDA_ARCH_LIST` | `"9.0 10.0"` | Archs for the torch-based source extensions (deep_ep, deep_gemm, mamba, causal-conv1d). |
| `NVTE_CUDA_ARCHS` | `90;100` | Archs for Transformer-Engine, which the `automodel` extra compiles from source. Narrow to one arch for a faster dev build. |
| `UV_SYNC_MODE` | `--frozen` | Reproducible sync. Set to empty to relock if a bumped RL commit's lock has drifted. |
| `NEMO_GYM_PREFETCH_CONFIGS` | *(empty — prefetch off)* | Space-separated Gym config paths whose environment venvs are baked into `/opt/gym_venvs`. Empty means every environment installs at runtime on first use. Set to `examples/nemo_gym/prefetch_super_all_envs.yaml` to bake NeMo-RL's curated set back in. |
| `BUILD_UID` / `BUILD_GID` | `2000` / `2000` | Non-runtime owner for `/opt/uv_cache`, `/opt/nemo_rl_venv`, `/opt/nemo-rl`, and prefetched venvs. Build steps use `umask 022` so the configured runtime UID can read/traverse these trees without a late recursive chmod layer. |
| `RUNTIME_UID` / `RUNTIME_GID` | `1000` / `1000` | Non-root identity for published `nmp-rl-base`. `RUNTIME_UID` owns the private vLLM copies; `/opt/ray_venvs` and `/opt/gym_venvs` are owned by root with `RUNTIME_GID` as the writable group. Keep `nmp-rl-training`'s `USER_UID` / `USER_GID` aligned if overriding the defaults. |
