# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#######
# NeMo Helix Docker build target configuration.
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

variable "NHX_COLLECT_SOURCES" {
  default = "0"
}

variable "NHX_PYTHON_IMAGE" {
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

variable "NHX_API_RUNTIME_BASE" {
  default = "nhx-python-base"
}

variable "NHX_CORE_RUNTIME_BASE" {
  default = "nhx-python-base"
}

variable "NHX_CPU_TASKS_RUNTIME_BASE" {
  default = "nhx-python-base"
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

# Tag for a published nhx-python-base image. Bake builds that target from
# NHX_PYTHON_IMAGE (currently python:3.13.15-slim-trixie); this SHA is not
# wired as a FROM pin. Rebuild the target in CI after changing NHX_PYTHON_IMAGE.
variable "BASE_TAG_PYTHON" {
  default = "d9e1851f309d3cf5389c0fc0e1049bd3c87593f8"
}

# Pin for nhx-automodel-base.
variable "BASE_TAG_AUTOMODEL" {
  default = "2c1a9ef2535a6648a272d9b74dae97fb672b8234"
}

# Pin for nhx-rl-base (prebuilt-base tag) + the NeMo-RL / NeMo-Gym source the base builds from.
# Defaults point at the soluwalana/{RL,Gym} forks. If the RL fork's uv.lock and the Gym ref drift,
# build with UV_SYNC_MODE= to relock.
variable "BASE_TAG_RL" {
  default = "local"
}
variable "NEMO_RL_REPO" {
  default = "https://github.com/soluwalana/RL.git"
}
# Pin to an immutable commit SHA, not the branch name. The base clones RL at this ref
# (docker/rl/Dockerfile.nhx-rl-base, nemo-rl stage); a branch ref would let the fork move under us,
# silently invalidating the base's heavy uv-sync layer on every training rebuild (and breaking
# `uv sync --frozen` if that commit's lock drifted). Bump deliberately, in lockstep with the fork's
# uv.lock, when advancing RL.
#
# RL pins Gym as a git submodule (-> soluwalana/Gym over https), so Gym rides in with the RL git ADD
# - no separate Gym pin needed.
variable "NEMO_RL_REF" {
  default = "9932dc8aa63a55fd431670d1b7c9d0bf3b2d2373" # soluwalana/RL nhx/customizer
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
  result = notequal(AUTOMODEL_BASE_CONTEXT, "") ? AUTOMODEL_BASE_CONTEXT : notequal(USE_PREBUILT_BASES, "") ? "docker-image://${BASE_REGISTRY}/nhx-automodel-base:${BASE_TAG_AUTOMODEL}" : "target:nhx-automodel-base-builder"
}

function "rl_base_context" {
  params = []
  result = notequal(RL_BASE_CONTEXT, "") ? RL_BASE_CONTEXT : notequal(USE_PREBUILT_BASES, "") ? "docker-image://${BASE_REGISTRY}/nhx-rl-base:${BASE_TAG_RL}" : "target:nhx-rl-base-builder"
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
    "nhx-auditor-tasks-docker",
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
    "nhx-auditor-tasks-docker",
  ]
}

group "docker-python-base" {
  targets = [
    "nhx-python-base-builder",
    "nhx-python-dev-base-builder",
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
    "nhx-api-docker",
    "nhx-tasks-docker",
    "nhx-gym-tasks-docker",
  ]
}

# CI extension that adds the colocated Gym task smoke-test stage to docker-cpu.
# The larger sandbox-host image has a dedicated Platform-Deploy build job.
group "docker-cpu-ci" {
  targets = [
    "docker-cpu",
    "nhx-agents-deepagents-e2e-docker",
    "nhx-tasks-smoke-test",
    "nhx-gym-tasks-smoke-test",
  ]
}

group "docker-gpu" {
  targets = [
    "nhx-safe-synthesizer-tasks-docker",
    "nhx-safe-synthesizer-tasks-smoke-test",
  ]
}

group "nhx-automodel-gpu-wheels" {
  targets = [
    "causal-conv1d-wheel",
    "mamba-ssm-wheel",
  ]
}

group "nhx-automodel" {
  targets = [
    "nhx-automodel-base-builder",
    "nhx-automodel-training-docker",
    "nhx-automodel-training-smoke-test",
    "nhx-customizer-tasks",
    "nhx-customizer-tasks-smoke-test",
  ]
}

group "nhx-unsloth" {
  targets = [
    "nhx-unsloth-training",
  ]
}

group "nhx-rl" {
  targets = [
    "nhx-rl-base-builder",
    "nhx-rl-training",
    "nhx-rl-training-smoke-test",
    "nhx-customizer-tasks",
  ]
}

group "nhx-customizer" {
  targets = [
    "nhx-customizer-tasks",
    "nhx-customizer-tasks-smoke-test",
  ]
}

# Pruned workspace slice for nhx-customizer-tasks (keep in sync with
# docker/customizer/pyproject.workspace.toml + Dockerfile.platform-workspace members).
target "customizer-platform-workspace" {
  target     = "platform-workspace"
  context    = "."
  dockerfile = "docker/customizer/Dockerfile.platform-workspace"
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

target "nhx-customizer-tasks" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-customizer-tasks"
  contexts = {
    platform-workspace       = "target:customizer-platform-workspace"
    causal-conv1d-wheel-src  = causal_conv1d_wheel_context()
    mamba-ssm-wheel-src      = mamba_ssm_wheel_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-customizer-tasks")
  cache-from = maybe_registry_cache_from("nhx-customizer-tasks")
  tags       = sha_and_maybe_latest_tags("nhx-customizer-tasks")
  output     = image_output()
  platforms  = get_platforms()
}

target "nhx-customizer-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-customizer-tasks"
  contexts = {
    platform-workspace       = "target:customizer-platform-workspace"
    causal-conv1d-wheel-src  = causal_conv1d_wheel_context()
    mamba-ssm-wheel-src      = mamba_ssm_wheel_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
    SMOKE_MARKER         = "smoke_nhx_customizer_tasks"
  }
  cache-from = maybe_registry_cache_from("nhx-customizer-tasks")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Pruned workspace slice for nhx-rl images (keep in sync with
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
target "nhx-rl-base-builder" {
  target     = "nhx-rl-base"
  context    = "."
  dockerfile = "docker/rl/Dockerfile.nhx-rl-base"
  args = {
    NEMO_RL_REPO        = NEMO_RL_REPO
    NEMO_RL_REF         = NEMO_RL_REF
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-rl-base")
  cache-from = maybe_registry_cache_from("nhx-rl-base")
  tags       = base_tags("nhx-rl-base")
  output     = image_output()
  platforms  = get_platforms()
}

# GPU DPO + GRPO training image (also the Gym environment runtime): base + platform glue.
# Bootstraps Ray at runtime.
target "nhx-rl-training" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/rl/Dockerfile.nhx-rl-training"
  contexts = {
    platform-workspace = "target:rl-platform-workspace"
    nhx-rl-base        = rl_base_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-rl-training")
  cache-from = maybe_registry_cache_from("nhx-rl-training")
  tags       = sha_and_maybe_latest_tags("nhx-rl-training")
  output     = image_output()
  platforms  = get_platforms()
}

# CPU-only import smoke tests for the training image (smoke-test stage runs pytest during build).
target "nhx-rl-training-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/rl/Dockerfile.nhx-rl-training"
  contexts = {
    platform-workspace = "target:rl-platform-workspace"
    nhx-rl-base        = rl_base_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
    SMOKE_MARKER         = "smoke_nhx_rl_training"
  }
  cache-from = maybe_registry_cache_from("nhx-rl-training")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Base images for consolidated containers
target "nhx-python-base" {
  target     = "nhx-python-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-python-base"
  args = {
    NHX_PYTHON_IMAGE = "${NHX_PYTHON_IMAGE}"
  }
  cache-from = maybe_registry_cache_from("nhx-python-base")
  platforms  = get_platforms()
}

target "nhx-python-base-builder" {
  target     = "nhx-python-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-python-base"
  args = {
    NHX_PYTHON_IMAGE = "${NHX_PYTHON_IMAGE}"
  }
  cache-to   = maybe_registry_cache_to("nhx-python-base")
  cache-from = maybe_registry_cache_from("nhx-python-base")
  tags       = base_tags("nhx-python-base")
  output     = image_output()
  platforms  = get_platforms()
}

target "nhx-python-dev-base" {
  target     = "nhx-python-dev-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-python-base"
  args = {
    NHX_PYTHON_IMAGE = "${NHX_PYTHON_IMAGE}"
  }
  cache-from = maybe_registry_cache_from("nhx-python-dev-base")
  platforms  = get_platforms()
}

target "nhx-python-dev-base-builder" {
  target     = "nhx-python-dev-base-builder"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-python-base"
  args = {
    NHX_PYTHON_IMAGE = "${NHX_PYTHON_IMAGE}"
  }
  cache-to   = maybe_registry_cache_to("nhx-python-dev-base")
  cache-from = maybe_registry_cache_from("nhx-python-dev-base")
  tags       = base_tags("nhx-python-dev-base")
  output     = image_output()
  platforms  = get_platforms()
}

target "nhx-jobs-launcher" {
  target     = "artifacts"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-jobs-launcher"
  platforms  = get_platforms()
}

target "nhx-studio-ui" {
  target     = "artifacts"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-studio-ui"
  platforms  = get_platforms()
  args = {
    VITE_VERSION_SHA = CI_COMMIT_SHA
    DOCKERHUB_MIRROR = DOCKERHUB_MIRROR
  }
}

target "nhx-gpu-base" {
  target     = "nhx-gpu-base"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-gpu-base"
  args = {
    CUDA_VERSION = CUDA_VERSION
  }
  platforms  = get_platforms()
}

target "nhx-gpu-base-py312" {
  target     = "nhx-gpu-base-py312"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-gpu-base-py312"
  args = {
    CUDA_VERSION = CUDA_VERSION
  }
  platforms  = get_platforms()
}

target "nhx-gpu-runtime-base" {
  target     = "nhx-gpu-runtime-base"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-gpu-base"
  contexts = {
    nhx-gpu-base = "target:nhx-gpu-base"
  }
  platforms  = get_platforms()
}

# Shared workspace layer (copy all workspace files for uv sync)
target "nhx-workspace" {
  target     = "nhx-workspace"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-workspace"
  contexts = {
    nhx-python-base           = "target:nhx-python-base"
  }
  args = {
    NHX_BASE = "nhx-python-base"
  }
  platforms  = get_platforms()
}

target "nhx-gpu-workspace" {
  target     = "nhx-workspace"
  context    = "."
  dockerfile = "docker/base/Dockerfile.nhx-workspace"
  contexts = {
    nhx-gpu-base               = "target:nhx-gpu-base"
  }
  args = {
    NHX_BASE = "nhx-gpu-base"
  }
  platforms  = get_platforms()
}

# NHX API - All Python services (core + application)
target "nhx-api-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-api"
  contexts = {
    nhx-python-base           = "target:nhx-python-base"
    nhx-workspace             = "target:nhx-workspace"
    nhx-jobs-launcher         = "target:nhx-jobs-launcher"
    nhx-studio-ui             = "target:nhx-studio-ui"
    policy-wasm-artifacts     = "target:root-policy-wasm-artifacts"
    root-busybox              = "target:root-busybox"
    fastembed-cache           = FASTEMBED_CACHE_CONTEXT
  }
  args = {
    NHX_PLATFORM_VERSION = notequal(BAKE_TAG, "") ? BAKE_TAG : "dev"
    NHX_CODE_REVISION    = notequal(CI_COMMIT_SHA, "") ? CI_COMMIT_SHA : "dev"
    NHX_API_RUNTIME_BASE = NHX_API_RUNTIME_BASE
    NHX_COLLECT_SOURCES  = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-api")
  cache-from = maybe_registry_cache_from("nhx-api")
  tags       = sha_and_maybe_latest_tags("nhx-api")
  output     = image_output()
  platforms  = get_platforms()
}

# E2E-only nhx-api variant with the DeepAgents harness installed.
target "nhx-agents-deepagents-e2e-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-api"
  contexts = {
    nhx-python-base           = "target:nhx-python-base"
    nhx-workspace             = "target:nhx-workspace"
    nhx-jobs-launcher         = "target:nhx-jobs-launcher"
    nhx-studio-ui             = "target:nhx-studio-ui"
    policy-wasm-artifacts     = "target:root-policy-wasm-artifacts"
    root-busybox              = "target:root-busybox"
    fastembed-cache           = FASTEMBED_CACHE_CONTEXT
  }
  args = {
    NHX_PLATFORM_VERSION      = notequal(BAKE_TAG, "") ? BAKE_TAG : "dev"
    NHX_CODE_REVISION         = notequal(CI_COMMIT_SHA, "") ? CI_COMMIT_SHA : "dev"
    NHX_API_RUNTIME_BASE      = NHX_API_RUNTIME_BASE
    NHX_API_DEPENDENCY_GROUP  = "agents-deepagents-e2e-services"
    NHX_COLLECT_SOURCES       = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-agents-deepagents-e2e")
  cache-from = maybe_registry_cache_from("nhx-agents-deepagents-e2e")
  tags       = sha_and_maybe_latest_tags("nhx-agents-deepagents-e2e")
  output     = image_output()
  platforms  = get_platforms()
}

# NHX Core - Core infrastructure services only
target "nhx-core-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-core"
  contexts = {
    nhx-python-base           = "target:nhx-python-base"
    nhx-workspace             = "target:nhx-workspace"
    nhx-jobs-launcher         = "target:nhx-jobs-launcher"
    policy-wasm-artifacts     = "target:root-policy-wasm-artifacts"
    root-busybox              = "target:root-busybox"
  }
  args = {
    NHX_CORE_RUNTIME_BASE = NHX_CORE_RUNTIME_BASE
    NHX_COLLECT_SOURCES   = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-core")
  cache-from = maybe_registry_cache_from("nhx-core")
  tags       = sha_and_maybe_latest_tags("nhx-core")
  output     = image_output()
  platforms  = get_platforms()
}

# NHX CPU Tasks - CPU-only batch task execution
target "nhx-tasks-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-tasks"
  contexts = {
    nhx-python-base           = "target:nhx-python-base"
    nhx-workspace             = "target:nhx-workspace"
    root-busybox              = "target:root-busybox"
  }
  args = {
    NHX_COLLECT_SOURCES        = NHX_COLLECT_SOURCES
    NHX_CPU_TASKS_RUNTIME_BASE = NHX_CPU_TASKS_RUNTIME_BASE
  }
  cache-to   = maybe_registry_cache_to("nhx-tasks")
  cache-from = maybe_registry_cache_from("nhx-tasks")
  tags       = sha_and_maybe_latest_tags("nhx-tasks")
  output     = image_output()
  platforms  = get_platforms()
}

# Cheap import validation for Evaluator's standard and sandboxed Gym task entrypoints.
target "nhx-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-tasks"
  contexts = {
    nhx-python-base           = "target:nhx-python-base"
    nhx-workspace             = "target:nhx-workspace"
    root-busybox              = "target:root-busybox"
  }
  args = {
    NHX_COLLECT_SOURCES        = NHX_COLLECT_SOURCES
    NHX_CPU_TASKS_RUNTIME_BASE = NHX_CPU_TASKS_RUNTIME_BASE
  }
  cache-from = maybe_registry_cache_from("nhx-tasks")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Dedicated colocated Gym task image. Gym and Ray remain isolated from the shared CPU task image.
target "nhx-gym-tasks-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-gym-tasks"
  contexts = {
    nhx-python-base = "target:nhx-python-base"
    nhx-tasks   = "target:nhx-tasks-docker"
    nhx-workspace   = "target:nhx-workspace"
  }
  cache-to   = maybe_registry_cache_to("nhx-gym-tasks")
  cache-from = maybe_registry_cache_from("nhx-gym-tasks")
  tags       = sha_and_maybe_latest_tags("nhx-gym-tasks")
  output     = image_output()
  platforms  = get_platforms()
}

# Cheap import/CLI validation for the isolated Gym environment. Built in docker-cpu-ci.
target "nhx-gym-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-gym-tasks"
  contexts = {
    nhx-python-base = "target:nhx-python-base"
    nhx-tasks   = "target:nhx-tasks-docker"
    nhx-workspace   = "target:nhx-workspace"
  }
  cache-from = maybe_registry_cache_from("nhx-gym-tasks")
  output     = ["type=cacheonly"]
  platforms  = get_platforms()
}

# Sandboxed Gym host. Evaluator provisions this image through OpenSandbox so
# user-authored environments run outside the trusted task container.
target "nhx-gym-host-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/gym-host/Dockerfile"
  contexts = {
    nhx-python-base = "target:nhx-python-base"
  }
  args = {
    NHX_PYTHON_IMAGE = NHX_PYTHON_IMAGE
  }
  cache-to   = maybe_registry_cache_to("nhx-gym-host")
  cache-from = maybe_registry_cache_from("nhx-gym-host")
  tags       = sha_and_maybe_latest_tags("nhx-gym-host")
  output     = image_output()
  platforms  = get_platforms()
}

# Validate the published host's CLI, runtime module, and locked dependencies.
target "nhx-gym-host-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/gym-host/Dockerfile"
  contexts = {
    nhx-python-base = "target:nhx-python-base"
  }
  args = {
    NHX_PYTHON_IMAGE = NHX_PYTHON_IMAGE
  }
  cache-from = maybe_registry_cache_from("nhx-gym-host")
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

target "nhx-safe-synthesizer-tasks-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-safe-synthesizer-tasks"
  args = {
    CONTAINER_VARIANT    = "${SAFE_SYNTHESIZER_CONTAINER_VARIANT}"
    NHX_COLLECT_SOURCES  = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-safe-synthesizer-tasks")
  cache-from = maybe_registry_cache_from("nhx-safe-synthesizer-tasks")
  tags       = sha_and_maybe_latest_tags("nhx-safe-synthesizer-tasks")
  output     = image_output()
  #platforms  = get_platforms()
  platforms  = ["linux/amd64"]
}

# Smoke test - built in parallel with nhx-safe-synthesizer-tasks-docker, never pushed.
# Fails the build if any critical import fails (missing package or ABI mismatch).
target "nhx-safe-synthesizer-tasks-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-safe-synthesizer-tasks"
  args = {
    CONTAINER_VARIANT    = "${SAFE_SYNTHESIZER_CONTAINER_VARIANT}"
    NHX_COLLECT_SOURCES  = NHX_COLLECT_SOURCES
  }
  cache-from = maybe_registry_cache_from("nhx-safe-synthesizer-tasks")
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

target "root-nhx-persistence-test" {
  target          = "root-nhx-persistence-test"
  context         = "."
  dockerfile      = "docker/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["root-nhx-persistence-test"]
}

target "root-nhx-common-test" {
  target          = "root-nhx-common-test"
  context         = "."
  dockerfile      = "docker/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["root-nhx-common-test"]
}

target "root-nemo-helix-test" {
  target          = "root-nemo-helix-test"
  context         = "."
  dockerfile      = "docker/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["root-nemo-helix-test"]
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

target "nhx-automodel-base-builder" {
  target          = "nhx-automodel-base"
  context         = "."
  dockerfile      = "docker/automodel/Dockerfile.nhx-automodel-base"
  no-cache-filter = ["automodel-clone"]
  cache-to        = maybe_registry_cache_to("nhx-automodel-base")
  cache-from      = maybe_registry_cache_from("nhx-automodel-base")
  tags            = base_tags("nhx-automodel-base")
  output          = image_output()
  contexts = {
    causal-conv1d-wheel-image = causal_conv1d_wheel_context()
    mamba-ssm-wheel-image     = mamba_ssm_wheel_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
  }
  platforms = get_platforms()
}

target "nhx-automodel-training-docker" {
  target     = "runtime"
  context    = "."
  dockerfile = "docker/automodel/Dockerfile.nhx-automodel-training"
  contexts = {
    platform-workspace = "target:automodel-platform-workspace"
    nhx-automodel-base = automodel_base_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-automodel-training")
  cache-from = maybe_registry_cache_from("nhx-automodel-training")
  tags       = sha_and_maybe_latest_tags("nhx-automodel-training")
  output     = image_output()
  platforms = get_platforms()
}

target "nhx-automodel-training-smoke-test" {
  target     = "smoke-test"
  context    = "."
  dockerfile = "docker/automodel/Dockerfile.nhx-automodel-training"
  contexts = {
    platform-workspace = "target:automodel-platform-workspace"
    nhx-automodel-base = automodel_base_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
    SMOKE_MARKER         = "smoke_nhx_automodel_training"
  }
  cache-from = maybe_registry_cache_from("nhx-automodel-training")
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

target "nhx-unsloth-training" {
  context    = "."
  dockerfile = "docker/Dockerfile.nhx-unsloth-training"
  target     = "runtime"
  contexts = {
    platform-workspace        = "target:unsloth-platform-workspace"
    causal-conv1d-wheel-image = causal_conv1d_wheel_context()
    mamba-ssm-wheel-image     = mamba_ssm_wheel_context()
  }
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-unsloth-training")
  cache-from = maybe_registry_cache_from("nhx-unsloth-training")
  tags       = sha_and_maybe_latest_tags("nhx-unsloth-training")
  output     = image_output()
  platforms  = get_platforms()
}

# Guardrails Callout service (Envoy ext_proc gRPC)
target "nhx-guardrails-callout-test" {
  target = "test"
  contexts = {
    root-golang-base = "target:root-golang-base-1-25"
  }
  cache-from      = maybe_registry_cache_from("nhx-guardrails-callout")
  context         = "services/guardrails/callouts"
  dockerfile      = "../../../docker/dockerfiles/services/guardrails/callouts/Dockerfile.bake"
  output          = ["type=cacheonly"]
  no-cache-filter = ["test"]
}

target "nhx-guardrails-callout-docker" {
  target = "docker"
  contexts = {
    root-golang-base = "target:root-golang-base-1-25"
  }
  context    = "services/guardrails/callouts"
  dockerfile = "../../../docker/dockerfiles/services/guardrails/callouts/Dockerfile.bake"
  cache-to   = maybe_registry_cache_to("nhx-guardrails-callout")
  cache-from = maybe_registry_cache_from("nhx-guardrails-callout")
  tags       = sha_and_maybe_latest_tags("nhx-guardrails-callout")
  output     = image_output()
  platforms  = get_platforms()
}

# Optional: mock LLM backend
# Do not add to the docker group to avoid publishing.
target "nhx-guardrails-callout-mock-llm" {
  target = "mock-llm"
  contexts = {
    root-golang-base = "target:root-golang-base-1-25"
  }
  context    = "services/guardrails/callouts"
  dockerfile = "../../../docker/dockerfiles/services/guardrails/callouts/Dockerfile.bake"
  tags       = sha_and_maybe_latest_tags("nhx-guardrails-callout-mock-llm")
  output     = image_output()
  platforms  = get_platforms()
}

# Auditor
target "nhx-auditor-tasks-docker" {
  target  = "release"
  context = "."
  contexts = {
    root-lib-source-artifacts = "target:root-lib-source-artifacts"
    root-busybox              = "target:root-busybox"
  }
  dockerfile = "docker/Dockerfile.nhx-auditor-tasks"
  args = {
    NHX_COLLECT_SOURCES = NHX_COLLECT_SOURCES
  }
  cache-to   = maybe_registry_cache_to("nhx-auditor-tasks")
  cache-from = maybe_registry_cache_from("nhx-auditor-tasks")
  tags       = sha_and_maybe_latest_tags("nhx-auditor-tasks")
  output     = image_output()
  platforms  = get_platforms()
}
