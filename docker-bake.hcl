# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#######
# NeMo Platform Docker build target configuration.
#######

variable "CACHE_REGISTRY" {
  default = "my-registry"
}

variable "CACHE_REGISTRY_BACKUP" {
  default = ""
}

variable "CACHE_VERSION" {
  default = "cache"
}

variable "IMAGE_REGISTRY" {
  default = "my-registry"
}

# Registry where pinned base images are published.
# In CI this matches CI_REGISTRY_IMAGE; locally override if needed.
variable "BASE_REGISTRY" {
  default = "my-registry"
}

variable "USE_PREBUILT_BASES" {
  default = ""
}

variable "NMP_COLLECT_SOURCES" {
  default = "0"
}

variable "NMP_PYTHON_IMAGE" {
  default = "python:3.13.15-slim-trixie"
}

variable "DISTROLESS_BASE_3_13" {
  # NGC 3.13-v4.1.3 (2026-09-09) ships CPython 3.13.15 and OpenSSL 3.5.7,
  # clearing the interpreter and OpenSSL CVEs open against 3.13-v4.0.9.
  # glibc is still 2.41-12+deb13u3 (CVE-2026-5450/5928 need deb13u4) as of
  # this tag; tracked separately pending a newer NGC publish, not blocking
  # this bump.
  default = "nvcr.io/nvidia/distroless/python:3.13-v4.1.3"
}

variable "NMP_API_RUNTIME_BASE" {
  default = "nmp-python-base"
}

variable "NMP_CORE_RUNTIME_BASE" {
  default = "nmp-python-base"
}

variable "NMP_CPU_TASKS_RUNTIME_BASE" {
  default = "nmp-python-base"
}

variable "AUTOMODEL_BASE_CONTEXT" {
  default = ""
}

variable "USE_LOCAL_WHEELS" {
  default = ""
}

variable "CAUSAL_CONV1D_WHEEL_CONTEXT" {
  default = ""
}

variable "MAMBA_SSM_WHEEL_CONTEXT" {
  default = ""
}

variable "FFMPEG_VLM_WHEEL_CONTEXT" {
  default = ""
}

variable "DISTROLESS_BASE" {
  default = "nvcr.io/nvidia/distroless/python:3.11-v4.0.8"
}

variable "DOCKERHUB_MIRROR" {
  default = "docker.io/library"
}

variable "WHEELS_REGISTRY" {
  default = "my-registry"
}

variable "BAKE_TAG" {
  default = "local"
}

# Tag for a published nmp-python-base image. Bake builds that target from
# NMP_PYTHON_IMAGE (currently python:3.13.15-slim-trixie); this SHA is not
# wired as a FROM pin. Rebuild the target in CI after changing NMP_PYTHON_IMAGE.
variable "BASE_TAG_PYTHON" {
  default = "d9e1851f309d3cf5389c0fc0e1049bd3c87593f8"
}

# Pin for nmp-automodel-base.
variable "BASE_TAG_AUTOMODEL" {
  default = "2c1a9ef2535a6648a272d9b74dae97fb672b8234"
}

# Pin for nmp-rl-base (prebuilt-base tag) + the NeMo-RL / NeMo-Gym source the base builds from.
# Defaults point at the soluwalana/{RL,Gym} forks. If the RL fork's uv.lock and the Gym ref drift,
# build with UV_SYNC_MODE= to relock.
variable "BASE_TAG_RL" {
  default = "local"
}
variable "NEMO_RL_REPO" {
  default = "https://github.com/soluwalana/RL.git"
}
# Pin to an immutable commit SHA, not the branch name. The base clones RL at this ref
# (docker/rl/Dockerfile.nmp-rl-base, nemo-rl stage); a branch ref would let the fork move under us,
# silently invalidating the base's heavy uv-sync layer on every training rebuild (and breaking
# `uv sync --frozen` if that commit's lock drifted). Bump deliberately, in lockstep with the fork's
# uv.lock, when advancing RL.
#
# RL pins Gym as a git submodule (-> soluwalana/Gym over https), so Gym rides in with the RL git ADD
# - no separate Gym pin needed.
variable "NEMO_RL_REF" {
  default = "9932dc8aa63a55fd431670d1b7c9d0bf3b2d2373" # soluwalana/RL nmp/customizer
}
variable "RL_BASE_CONTEXT" {
  default = ""
}

# The tag for base images if needed
variable "WHEELS_TAG" {
  default = "54ae40bf653127f1300399912e6c1083f0b96771"
}

variable "BAKE_CACHE_SOURCE_BRANCH" {
  default = ""
}

variable "BAKE_CACHE_TARGET_BRANCH" {
  default = ""
}

variable "PUBLISH_LATEST" {
  default = false
}

variable "CI_COMMIT_SHA" {
  default = ""
}

variable "BUILD_ARCH" {
  default = ""
}

variable "FASTEMBED_CACHE_CONTEXT" {
  default = "docker/fastembed-cache-empty"
}

variable "CUDA_VERSION" {
  default = "12.8.1"
}

variable "SAFE_SYNTHESIZER_CONTAINER_VARIANT" {
  default = "cu129"
}

# Versions for the mamba wheel builder.
variable "MAMBA_22_COMMIT" {
  default = "6b32be06d026e170b3fdaf3ae6282c5a6ff57b06"
}

variable "MAMBA_23_COMMIT" {
  default = "v2.3.0"
}

variable "CAUSAL_CONV1D_VERSION" {
  default = "v1.5.3"
}

function "get_causal_conv1d_wheel_image" {
  params = []
  result = "${WHEELS_REGISTRY}/causal-conv1d-wheel:${WHEELS_TAG}"
}

function "get_mamba_ssm_wheel_image" {
  params = []
  result = "${WHEELS_REGISTRY}/mamba-ssm-wheel:${WHEELS_TAG}"
}

function "get_ffmpeg_vlm_wheel_image" {
  params = []
  result = "${WHEELS_REGISTRY}/ffmpeg-vlm-wheel:${WHEELS_TAG}"
}

function "get_arch_tag" {
  params = []
  result = BUILD_ARCH == "linux/arm64" ? "linux-arm64" : "linux-amd64"
}

function "base_tags" {
  params = [name]
  result = [
    notequal(BAKE_TAG, "") ? "${BASE_REGISTRY}/${name}:${BAKE_TAG}" : "",
  ]
}

function "automodel_base_context" {
  params = []
  result = notequal(AUTOMODEL_BASE_CONTEXT, "") ? AUTOMODEL_BASE_CONTEXT : notequal(USE_PREBUILT_BASES, "") ? "docker-image://${BASE_REGISTRY}/nmp-automodel-base:${BASE_TAG_AUTOMODEL}" : "target:nmp-automodel-base-builder"
}

function "rl_base_context" {
  params = []
  result = notequal(RL_BASE_CONTEXT, "") ? RL_BASE_CONTEXT : notequal(USE_PREBUILT_BASES, "") ? "docker-image://${BASE_REGISTRY}/nmp-rl-base:${BASE_TAG_RL}" : "target:nmp-rl-base-builder"
}

function "causal_conv1d_wheel_context" {
  params = []
  result = notequal(CAUSAL_CONV1D_WHEEL_CONTEXT, "") ? CAUSAL_CONV1D_WHEEL_CONTEXT : notequal(USE_LOCAL_WHEELS, "") ? "target:causal-conv1d-wheel" : "docker-image://${get_causal_conv1d_wheel_image()}"
}

function "mamba_ssm_wheel_context" {
  params = []
  result = notequal(MAMBA_SSM_WHEEL_CONTEXT, "") ? MAMBA_SSM_WHEEL_CONTEXT : notequal(USE_LOCAL_WHEELS, "") ? "target:mamba-ssm-wheel" : "docker-image://${get_mamba_ssm_wheel_image()}"
}

function "ffmpeg_vlm_wheel_context" {
  params = []
  result = notequal(FFMPEG_VLM_WHEEL_CONTEXT, "") ? FFMPEG_VLM_WHEEL_CONTEXT : notequal(USE_LOCAL_WHEELS, "") ? "target:ffmpeg-vlm-wheel" : "docker-image://${get_ffmpeg_vlm_wheel_image()}"
}

function "wheel_tags" {
  params = [name]
  result = [
    notequal(WHEELS_TAG, "") ? "${WHEELS_REGISTRY}/${name}:${WHEELS_TAG}" : "",
  ]
}

function "sha_and_maybe_latest_tags" {
  params = [name]
  result = [
    notequal(BAKE_TAG, "") ? "${IMAGE_REGISTRY}/${name}:${BAKE_TAG}" : "",
    PUBLISH_LATEST ? "${IMAGE_REGISTRY}/${name}:latest" : "",
    and(notequal(BAKE_TAG, ""), and(notequal("", CI_COMMIT_SHA), notequal(BAKE_TAG, CI_COMMIT_SHA))) ? "${IMAGE_REGISTRY}/${name}:${CI_COMMIT_SHA}" : "",
  ]
}

function "maybe_registry_cache_to" {
  params = [name]
  result = [
    and(notequal(BAKE_CACHE_TARGET_BRANCH, ""), notequal(BUILD_ARCH, "")) ? "type=registry,ref=${CACHE_REGISTRY}/${name}:${CACHE_VERSION}-${BAKE_CACHE_TARGET_BRANCH}-${get_arch_tag()},mode=max,compression=zstd,force-compression=true" : ""
  ]
}

function "image_output" {
  params = []
  result = ["type=image,compression=zstd,force-compression=true"]
}

function "maybe_registry_cache_from" {
  params = [name]
  result = [
    notequal(BAKE_CACHE_SOURCE_BRANCH, "") ? "type=registry,ref=${CACHE_REGISTRY}/${name}:${CACHE_VERSION}-${BAKE_CACHE_SOURCE_BRANCH}-linux-arm64" : "",
    notequal(BAKE_CACHE_SOURCE_BRANCH, "") ? "type=registry,ref=${CACHE_REGISTRY}/${name}:${CACHE_VERSION}-${BAKE_CACHE_SOURCE_BRANCH}-linux-amd64" : "",
    and(notequal(CACHE_REGISTRY_BACKUP, ""), notequal(BAKE_CACHE_SOURCE_BRANCH, "")) ? "type=registry,ref=${CACHE_REGISTRY_BACKUP}/${name}:${CACHE_VERSION}-${BAKE_CACHE_SOURCE_BRANCH}-linux-arm64" : "",
    and(notequal(CACHE_REGISTRY_BACKUP, ""), notequal(BAKE_CACHE_SOURCE_BRANCH, "")) ? "type=registry,ref=${CACHE_REGISTRY_BACKUP}/${name}:${CACHE_VERSION}-${BAKE_CACHE_SOURCE_BRANCH}-linux-amd64" : "",
  ]
}

function "get_platforms" {
  params = []
  result = BUILD_ARCH != "" ? [BUILD_ARCH] : ["linux/amd64", "linux/arm64"]
}

# Semantic groups for parallel CI builds

# Auditor images
group "docker-auditor" {
  targets = [
    "auditor-tasks-docker",
  ]
}

group "all-multi-platform" {
  targets = [
    "docker-multi-platform",
  ]
}

group "docker-multi-platform" {
  targets = [
    "docker-cpu",
    "auditor-tasks-docker",
  ]
}

group "docker-python-base" {
  targets = [
    "nmp-python-base-builder",
    "nmp-python-dev-base-builder",
  ]
}

group "all-arm64" {
  targets = [
    "docker-multi-platform",
  ]
}

group "all-amd64" {
  targets = [
    "docker-multi-platform",
    "docker-gpu",
  ]
}

group "docker" {
  targets = [
    "docker-cpu",
    "docker-gpu",
    "docker-auditor",
  ]
}

# =============================================================================
# Consolidated Container Builds
# =============================================================================
# Consolidated container images for Python services and task runners.

# Build groups for consolidated containers
group "docker-cpu" {
  targets = [
    "nmp-api-docker",
    "nmp-cpu-tasks-docker",
    "nmp-gym-tasks-docker",
  ]
}

# CI extension that adds the colocated Gym task smoke-test stage to docker-cpu.
# The larger sandbox-host image has a dedicated Platform-Deploy build job.
group "docker-cpu-ci" {
  targets = [
    "docker-cpu",
    "nmp-cpu-tasks-smoke-test",
    "nmp-gym-tasks-smoke-test",
  ]
}

group "docker-gpu" {
  targets = [
    "safe-synthesizer-tasks-docker",
    "safe-synthesizer-tasks-smoke-test",
  ]
}

group "nmp-automodel-gpu-wheels" {
  targets = [
    "causal-conv1d-wheel",
    "mamba-ssm-wheel",
  ]
}

group "nmp-automodel" {
  targets = [
    "nmp-automodel-base-builder",
    "nmp-automodel-training-docker",
    "nmp-automodel-training-smoke-test",
    "nmp-customizer-tasks",
    "nmp-customizer-tasks-smoke-test",
  ]
}

group "nmp-unsloth" {
  targets = [
    "nmp-unsloth-training",
  ]
}

group "nmp-rl" {
  targets = [
    "nmp-rl-base-builder",
    "nmp-rl-training",
    "nmp-rl-training-smoke-test",
    "nmp-customizer-tasks",
  ]
}

group "nmp-customizer" {
  targets = [
    "nmp-customizer-tasks",
    "nmp-customizer-tasks-smoke-test",
  ]
}

# Pruned workspace slice for nmp-customizer-tasks (keep in sync with
# docker/customizer/pyproject.workspace.toml + Dockerfile.platform-workspace members).
target "customizer-platform-workspace" {
  target     = "platform-workspace"
  context    = "."
  dockerfile = "docker/customizer/Dockerfile.platform-workspace"
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

target "nmp-customizer-tasks" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-customizer-tasks"
  contexts = {
    platform-workspace       = "target:customizer-platform-workspace"
    causal-conv1d-wheel-src  = causal_conv1d_wheel_context()
    mamba-ssm-wheel-src      = mamba_ssm_wheel_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nmp-customizer-tasks")
  cache-from = maybe_registry_cache_from("nmp-customizer-tasks")
  tags       = sha_and_maybe_latest_tags("nmp-customizer-tasks")
  output     = image_output()
  platforms  = get_platforms()
}

target "nmp-customizer-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-customizer-tasks"
  contexts = {
    platform-workspace       = "target:customizer-platform-workspace"
    causal-conv1d-wheel-src  = causal_conv1d_wheel_context()
    mamba-ssm-wheel-src      = mamba_ssm_wheel_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
    SMOKE_MARKER         = "smoke_nmp_customizer_tasks"
  }
  cache-from = maybe_registry_cache_from("nmp-customizer-tasks")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Pruned workspace slice for nmp-rl images (keep in sync with
# docker/rl/pyproject.workspace.toml + Dockerfile.platform-workspace members).
target "rl-platform-workspace" {
  target     = "platform-workspace"
  context    = "."
  dockerfile = "docker/rl/Dockerfile.platform-workspace"
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Heavy base: cuda-dl-base + NeMo-RL (with its Gym/Automodel/Megatron-Bridge submodules) built FROM
# SOURCE (Python 3.13, CUDA 13). RL is pinned via NEMO_RL_REF; Gym rides in as RL's own submodule
target "nmp-rl-base-builder" {
  target     = "nmp-rl-base"
  context    = "."
  dockerfile = "docker/rl/Dockerfile.nmp-rl-base"
  args = {
    NEMO_RL_REPO        = NEMO_RL_REPO
    NEMO_RL_REF         = NEMO_RL_REF
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nmp-rl-base")
  cache-from = maybe_registry_cache_from("nmp-rl-base")
  tags       = base_tags("nmp-rl-base")
  output     = image_output()
  platforms  = get_platforms()
}

# GPU DPO + GRPO training image (also the Gym environment runtime): base + platform glue.
# Bootstraps Ray at runtime.
target "nmp-rl-training" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/rl/Dockerfile.nmp-rl-training"
  contexts = {
    platform-workspace = "target:rl-platform-workspace"
    nmp-rl-base        = rl_base_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nmp-rl-training")
  cache-from = maybe_registry_cache_from("nmp-rl-training")
  tags       = sha_and_maybe_latest_tags("nmp-rl-training")
  output     = image_output()
  platforms  = get_platforms()
}

# CPU-only import smoke tests for the training image (smoke-test stage runs pytest during build).
target "nmp-rl-training-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/rl/Dockerfile.nmp-rl-training"
  contexts = {
    platform-workspace = "target:rl-platform-workspace"
    nmp-rl-base        = rl_base_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
    SMOKE_MARKER         = "smoke_nmp_rl_training"
  }
  cache-from = maybe_registry_cache_from("nmp-rl-training")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Base images for consolidated containers
target "nmp-python-base" {
  target     = "nmp-python-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-python-base"
  args = {
    NMP_PYTHON_IMAGE = "${NMP_PYTHON_IMAGE}"
  }
  cache-from = maybe_registry_cache_from("nmp-python-base")
  platforms  = get_platforms()
}

target "nmp-python-base-builder" {
  target     = "nmp-python-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-python-base"
  args = {
    NMP_PYTHON_IMAGE = "${NMP_PYTHON_IMAGE}"
  }
  cache-to   = maybe_registry_cache_to("nmp-python-base")
  cache-from = maybe_registry_cache_from("nmp-python-base")
  tags       = base_tags("nmp-python-base")
  output     = image_output()
  platforms  = get_platforms()
}

target "nmp-python-dev-base" {
  target     = "nmp-python-dev-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-python-base"
  args = {
    NMP_PYTHON_IMAGE = "${NMP_PYTHON_IMAGE}"
  }
  cache-from = maybe_registry_cache_from("nmp-python-dev-base")
  platforms  = get_platforms()
}

target "nmp-python-dev-base-builder" {
  target     = "nmp-python-dev-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-python-base"
  args = {
    NMP_PYTHON_IMAGE = "${NMP_PYTHON_IMAGE}"
  }
  cache-to   = maybe_registry_cache_to("nmp-python-dev-base")
  cache-from = maybe_registry_cache_from("nmp-python-dev-base")
  tags       = base_tags("nmp-python-dev-base")
  output     = image_output()
  platforms  = get_platforms()
}

target "nmp-jobs-launcher" {
  target     = "artifacts"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-jobs-launcher"
  platforms  = get_platforms()
}

target "nmp-studio-ui" {
  target     = "artifacts"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-studio-ui"
  platforms  = get_platforms()
  args = {
    VITE_VERSION_SHA = CI_COMMIT_SHA
    DOCKERHUB_MIRROR = DOCKERHUB_MIRROR
  }
}

target "nmp-gpu-base" {
  target     = "nmp-gpu-base"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-gpu-base"
  args = {
    CUDA_VERSION = CUDA_VERSION
  }
  platforms  = get_platforms()
}

target "nmp-gpu-base-py312" {
  target     = "nmp-gpu-base-py312"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-gpu-base-py312"
  args = {
    CUDA_VERSION = CUDA_VERSION
  }
  platforms  = get_platforms()
}

target "nmp-gpu-runtime-base" {
  target     = "nmp-gpu-runtime-base"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-gpu-base"
  contexts = {
    nmp-gpu-base = "target:nmp-gpu-base"
  }
  platforms  = get_platforms()
}

# Shared workspace layer (copy all workspace files for uv sync)
target "nmp-workspace" {
  target     = "nmp-workspace"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-workspace"
  contexts = {
    nmp-python-base           = "target:nmp-python-base"
  }
  args = {
    NMP_BASE = "nmp-python-base"
  }
  platforms  = get_platforms()
}

target "nmp-gpu-workspace" {
  target     = "nmp-workspace"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nmp-workspace"
  contexts = {
    nmp-gpu-base               = "target:nmp-gpu-base"
  }
  args = {
    NMP_BASE = "nmp-gpu-base"
  }
  platforms  = get_platforms()
}

# NMP API - All Python services (core + application)
target "nmp-api-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-api"
  contexts = {
    nmp-python-base           = "target:nmp-python-base"
    nmp-workspace             = "target:nmp-workspace"
    nmp-jobs-launcher         = "target:nmp-jobs-launcher"
    nmp-studio-ui             = "target:nmp-studio-ui"
    policy-wasm-artifacts     = "target:root-policy-wasm-artifacts"
    root-busybox              = "target:root-busybox"
    fastembed-cache           = FASTEMBED_CACHE_CONTEXT
  }
  args = {
    NMP_PLATFORM_VERSION  = notequal(BAKE_TAG, "") ? BAKE_TAG : "dev"
    NMP_CODE_REVISION     = notequal(CI_COMMIT_SHA, "") ? CI_COMMIT_SHA : "dev"
    NMP_API_RUNTIME_BASE  = NMP_API_RUNTIME_BASE
    NMP_COLLECT_SOURCES   = NMP_COLLECT_SOURCES
    SWITCHYARD_NATIVE_REF = ""
  }
  cache-to   = maybe_registry_cache_to("nmp-api")
  cache-from = maybe_registry_cache_from("nmp-api")
  tags       = sha_and_maybe_latest_tags("nmp-api")
  output     = image_output()
  platforms  = get_platforms()
}

# NMP Core - Core infrastructure services only
target "nmp-core-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-core"
  contexts = {
    nmp-python-base           = "target:nmp-python-base"
    nmp-workspace             = "target:nmp-workspace"
    nmp-jobs-launcher         = "target:nmp-jobs-launcher"
    policy-wasm-artifacts     = "target:root-policy-wasm-artifacts"
    root-busybox              = "target:root-busybox"
  }
  args = {
    NMP_CORE_RUNTIME_BASE = NMP_CORE_RUNTIME_BASE
    NMP_COLLECT_SOURCES   = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nmp-core")
  cache-from = maybe_registry_cache_from("nmp-core")
  tags       = sha_and_maybe_latest_tags("nmp-core")
  output     = image_output()
  platforms  = get_platforms()
}

# NMP CPU Tasks - CPU-only batch task execution
target "nmp-cpu-tasks-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-cpu-tasks"
  contexts = {
    nmp-python-base           = "target:nmp-python-base"
    nmp-workspace             = "target:nmp-workspace"
    root-busybox              = "target:root-busybox"
  }
  args = {
    NMP_COLLECT_SOURCES        = NMP_COLLECT_SOURCES
    NMP_CPU_TASKS_RUNTIME_BASE = NMP_CPU_TASKS_RUNTIME_BASE
  }
  cache-to   = maybe_registry_cache_to("nmp-cpu-tasks")
  cache-from = maybe_registry_cache_from("nmp-cpu-tasks")
  tags       = sha_and_maybe_latest_tags("nmp-cpu-tasks")
  output     = image_output()
  platforms  = get_platforms()
}

# Cheap import validation for Evaluator's standard and sandboxed Gym task entrypoints.
target "nmp-cpu-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-cpu-tasks"
  contexts = {
    nmp-python-base           = "target:nmp-python-base"
    nmp-workspace             = "target:nmp-workspace"
    root-busybox              = "target:root-busybox"
  }
  args = {
    NMP_COLLECT_SOURCES        = NMP_COLLECT_SOURCES
    NMP_CPU_TASKS_RUNTIME_BASE = NMP_CPU_TASKS_RUNTIME_BASE
  }
  cache-from = maybe_registry_cache_from("nmp-cpu-tasks")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Dedicated colocated Gym task image. Gym and Ray remain isolated from the shared CPU task image.
target "nmp-gym-tasks-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-gym-tasks"
  contexts = {
    nmp-python-base = "target:nmp-python-base"
    nmp-cpu-tasks   = "target:nmp-cpu-tasks-docker"
    nmp-workspace   = "target:nmp-workspace"
  }
  cache-to   = maybe_registry_cache_to("nmp-gym-tasks")
  cache-from = maybe_registry_cache_from("nmp-gym-tasks")
  tags       = sha_and_maybe_latest_tags("nmp-gym-tasks")
  output     = image_output()
  platforms  = get_platforms()
}

# Cheap import/CLI validation for the isolated Gym environment. Built in docker-cpu-ci.
target "nmp-gym-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-gym-tasks"
  contexts = {
    nmp-python-base = "target:nmp-python-base"
    nmp-cpu-tasks   = "target:nmp-cpu-tasks-docker"
    nmp-workspace   = "target:nmp-workspace"
  }
  cache-from = maybe_registry_cache_from("nmp-gym-tasks")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Sandboxed Gym host. Evaluator provisions this image through OpenSandbox so
# user-authored environments run outside the trusted task container.
target "nmp-gym-host-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/gym-host/Dockerfile"
  contexts = {
    nmp-python-base = "target:nmp-python-base"
  }
  args = {
    NMP_PYTHON_IMAGE = NMP_PYTHON_IMAGE
  }
  cache-to   = maybe_registry_cache_to("nmp-gym-host")
  cache-from = maybe_registry_cache_from("nmp-gym-host")
  tags       = sha_and_maybe_latest_tags("nmp-gym-host")
  output     = image_output()
  platforms  = get_platforms()
}

# Validate the published host's CLI, runtime module, and locked dependencies.
target "nmp-gym-host-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/gym-host/Dockerfile"
  contexts = {
    nmp-python-base = "target:nmp-python-base"
  }
  args = {
    NMP_PYTHON_IMAGE = NMP_PYTHON_IMAGE
  }
  cache-from = maybe_registry_cache_from("nmp-gym-host")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Python wheel builders (causal-conv1d, mamba-ssm, av, opencv-python-headless).
# CUDA extensions only ship source on PyPI; av/opencv bundle FFmpeg. Pre-built for
# amd64 and arm64. Wheels live at /wheels/*.whl inside each image.

target "causal-conv1d-wheel" {
  target     = "causal-conv1d-wheel"
  context    = "."
  dockerfile = "docker/base/Dockerfile.python-wheels"
  cache-to   = maybe_registry_cache_to("causal-conv1d-wheel")
  cache-from = maybe_registry_cache_from("causal-conv1d-wheel")
  tags       = wheel_tags("causal-conv1d-wheel")
  output     = image_output()
  args = {
    CUDA_VERSION          = CUDA_VERSION
    CAUSAL_CONV1D_VERSION = CAUSAL_CONV1D_VERSION
  }
  platforms = get_platforms()
}

target "mamba-ssm-wheel" {
  target     = "mamba-ssm-wheel"
  context    = "."
  dockerfile = "docker/base/Dockerfile.python-wheels"
  cache-to   = maybe_registry_cache_to("mamba-ssm-wheel")
  cache-from = maybe_registry_cache_from("mamba-ssm-wheel")
  tags       = wheel_tags("mamba-ssm-wheel")
  output     = image_output()
  args = {
    CUDA_VERSION    = CUDA_VERSION
    MAMBA_22_COMMIT = MAMBA_22_COMMIT
    MAMBA_23_COMMIT = MAMBA_23_COMMIT
  }
  platforms = get_platforms()
}

target "ffmpeg-vlm-wheel" {
  target     = "ffmpeg-vlm-wheel"
  context    = "."
  dockerfile = "docker/base/Dockerfile.python-wheels"
  cache-to   = maybe_registry_cache_to("ffmpeg-vlm-wheel")
  cache-from = maybe_registry_cache_from("ffmpeg-vlm-wheel")
  tags       = wheel_tags("ffmpeg-vlm-wheel")
  output     = image_output()
  platforms  = get_platforms()
}

target "safe-synthesizer-tasks-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.safe-synthesizer-tasks"
  args = {
    CONTAINER_VARIANT    = "${SAFE_SYNTHESIZER_CONTAINER_VARIANT}"
    NMP_COLLECT_SOURCES  = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("safe-synthesizer-tasks")
  cache-from = maybe_registry_cache_from("safe-synthesizer-tasks")
  tags       = sha_and_maybe_latest_tags("safe-synthesizer-tasks")
  output     = image_output()
  #platforms  = get_platforms()
  platforms  = ["linux/amd64"]
}

# Smoke test - built in parallel with safe-synthesizer-tasks-docker, never pushed.
# Fails the build if any critical import fails (missing package or ABI mismatch).
target "safe-synthesizer-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.safe-synthesizer-tasks"
  args = {
    CONTAINER_VARIANT    = "${SAFE_SYNTHESIZER_CONTAINER_VARIANT}"
    NMP_COLLECT_SOURCES  = NMP_COLLECT_SOURCES
  }
  cache-from = maybe_registry_cache_from("safe-synthesizer-tasks")
  output     = ["type=cacheonly"]
  platforms  = ["linux/amd64"]
}

# root
target "root-artifact-base" {
  target     = "root-artifact-base"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
}

target "root-uv-artifacts" {
  target     = "root-uv-artifacts"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
}

target "root-distroless-base-3-11" {
  target     = "root-distroless-base-3-11"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
  args = {
    DISTROLESS_BASE = DISTROLESS_BASE
  }
}

target "root-distroless-base-3-13" {
  target     = "root-distroless-base-3-13"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
  args = {
    DISTROLESS_BASE_3_13 = DISTROLESS_BASE_3_13
  }
}

target "root-lib-source-artifacts" {
  target     = "root-lib-source-artifacts"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

target "root-service-source-artifacts" {
  target     = "root-service-source-artifacts"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

target "root-script-source-artifacts" {
  target     = "root-script-source-artifacts"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

target "root-golang-base" {
  target     = "root-golang-base"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

target "root-golang-base-1-24" {
  target     = "root-golang-base-1-24"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

target "root-golang-base-1-25" {
  target     = "root-golang-base-1-25"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

target "root-golang-artifacts" {
  target     = "root-golang-artifacts"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

# Auth policy WASM bundle (built from Rego during container build; multi-arch).
target "root-policy-wasm-artifacts" {
  target     = "root-policy-wasm-artifacts"
  context    = "."
  dockerfile = "docker/base/Dockerfile.policy-wasm"
  platforms  = get_platforms()
}

target "root-busybox" {
  target     = "root-busybox"
  context    = "."
  dockerfile = "docker/Dockerfile.bake"
  platforms  = get_platforms()
}

target "root-nmp-persistence-test" {
  target          = "root-nmp-persistence-test"
  context         = "."
  dockerfile      = "docker/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["root-nmp-persistence-test"]
}

target "root-nmp-common-test" {
  target          = "root-nmp-common-test"
  context         = "."
  dockerfile      = "docker/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["root-nmp-common-test"]
}

target "root-nemo-platform-test" {
  target          = "root-nemo-platform-test"
  context         = "."
  dockerfile      = "docker/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["root-nemo-platform-test"]
}

target "buildkit-test" {
  target          = "buildkit-test"
  context         = "."
  dockerfile      = "docker/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["buildkit-test"]
  platforms       = get_platforms()
}

# Automodel and Unsloth

target "automodel-platform-workspace" {
  target     = "platform-workspace"
  context    = "."
  dockerfile = "docker/automodel/Dockerfile.platform-workspace"
  platforms  = get_platforms()
}

target "nmp-automodel-base-builder" {
  target          = "nmp-automodel-base"
  context         = "."
  dockerfile      = "docker/automodel/Dockerfile.nmp-automodel-base"
  no-cache-filter = ["automodel-clone"]
  cache-to        = maybe_registry_cache_to("nmp-automodel-base")
  cache-from      = maybe_registry_cache_from("nmp-automodel-base")
  tags            = base_tags("nmp-automodel-base")
  output          = image_output()
  contexts = {
    causal-conv1d-wheel-image = causal_conv1d_wheel_context()
    mamba-ssm-wheel-image     = mamba_ssm_wheel_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
  }
  platforms = get_platforms()
}

target "nmp-automodel-training-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/automodel/Dockerfile.nmp-automodel-training"
  contexts = {
    platform-workspace = "target:automodel-platform-workspace"
    nmp-automodel-base = automodel_base_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nmp-automodel-training")
  cache-from = maybe_registry_cache_from("nmp-automodel-training")
  tags       = sha_and_maybe_latest_tags("nmp-automodel-training")
  output     = image_output()
  platforms = get_platforms()
}

target "nmp-automodel-training-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/automodel/Dockerfile.nmp-automodel-training"
  contexts = {
    platform-workspace = "target:automodel-platform-workspace"
    nmp-automodel-base = automodel_base_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
    SMOKE_MARKER         = "smoke_nmp_automodel_training"
  }
  cache-from = maybe_registry_cache_from("nmp-automodel-training")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

target "unsloth-platform-workspace" {
  context    = "."
  dockerfile = "docker/unsloth/Dockerfile.platform-workspace"
  target     = "platform-workspace"
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

target "nmp-unsloth-training" {
  context    = "."
  dockerfile = "docker/Dockerfile.nmp-unsloth-training"
  target     = "runtime"
  contexts = {
    platform-workspace        = "target:unsloth-platform-workspace"
    causal-conv1d-wheel-image = causal_conv1d_wheel_context()
    mamba-ssm-wheel-image     = mamba_ssm_wheel_context()
  }
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nmp-unsloth-training")
  cache-from = maybe_registry_cache_from("nmp-unsloth-training")
  tags       = sha_and_maybe_latest_tags("nmp-unsloth-training")
  output     = image_output()
  platforms  = get_platforms()
}

# Guardrails Callout service (Envoy ext_proc gRPC)
target "guardrails-callout-test" {
  target = "test"
  contexts = {
    root-golang-base = "target:root-golang-base-1-25"
  }
  cache-from      = maybe_registry_cache_from("guardrails-callout")
  context         = "services/guardrails/callouts"
  dockerfile      = "../../../docker/dockerfiles/services/guardrails/callouts/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["test"]
}

target "guardrails-callout-docker" {
  target = "docker"
  contexts = {
    root-golang-base = "target:root-golang-base-1-25"
  }
  context    = "services/guardrails/callouts"
  dockerfile = "../../../docker/dockerfiles/services/guardrails/callouts/Dockerfile.bake"
  cache-to   = maybe_registry_cache_to("guardrails-callout")
  cache-from = maybe_registry_cache_from("guardrails-callout")
  tags       = sha_and_maybe_latest_tags("guardrails-callout")
  output     = image_output()
  platforms  = get_platforms()
}

# Optional: mock LLM backend
# Do not add to the docker group to avoid publishing.
target "guardrails-callout-mock-llm" {
  target = "mock-llm"
  contexts = {
    root-golang-base = "target:root-golang-base-1-25"
  }
  context    = "services/guardrails/callouts"
  dockerfile = "../../../docker/dockerfiles/services/guardrails/callouts/Dockerfile.bake"
  tags       = sha_and_maybe_latest_tags("guardrails-callout-mock-llm")
  output     = image_output()
  platforms  = get_platforms()
}

# Auditor
target "auditor-tasks-docker" {
  target  = "release"
  context = "."
  contexts = {
    root-lib-source-artifacts = "target:root-lib-source-artifacts"
    root-busybox              = "target:root-busybox"
  }
  dockerfile = "docker/Dockerfile.auditor-tasks"
  args = {
    NMP_COLLECT_SOURCES = NMP_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("auditor-tasks")
  cache-from = maybe_registry_cache_from("auditor-tasks")
  tags       = sha_and_maybe_latest_tags("auditor-tasks")
  output     = image_output()
  platforms  = get_platforms()
}
