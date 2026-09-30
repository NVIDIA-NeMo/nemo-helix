# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nhx-rl-training image import smoke tests.

Built as part of the docker-bake.hcl bake group (smoke-test stage) and run on a CPU runner —
no GPU hardware required.

Two failure classes are caught at .so load time, before any GPU device is touched:

  ModuleNotFoundError  — package missing from the environment (e.g. a platform-glue dependency
                         that did not resolve, or a source tree excluded from a layer)

  ImportError          — CUDA extension .so has an undefined symbol; the wheel/build was compiled
                         against a different PyTorch version than the one installed (ABI mismatch)

WHICH ENVIRONMENT EACH IMPORT BELONGS TO
----------------------------------------
NeMo-RL does not run training in the image's base venv. Every Ray actor launches with its own
per-node venv under /opt/ray_venvs (see docker/rl/README.md, "How NeMo-RL runs"). The base venv
holds only the default dependency set, because each ``uv sync --extra`` during the build prunes
the previous extra's packages.

So ``import vllm`` from the base venv is *expected* to fail and would prove nothing. These tests
run each import in the venv that actually owns the package, using that venv's own interpreter.
"""

import subprocess
from pathlib import Path

import pytest
from file_removals import assert_file_patterns_absent, read_file_patterns
from python_package_versions import assert_python_package_min_versions

RAY_VENVS = Path("/opt/ray_venvs")
FINAL_FILE_REMOVALS = Path("/smoke_test/removals/files/final/customizer-codecs.txt")
SOUNDFILE_FILE_REMOVALS = {
    "/opt/nemo_rl_venv/lib/python3.*/site-packages/_soundfile_data/libsndfile_*.so",
    "/opt/ray_venvs/*/lib/python3.*/site-packages/_soundfile_data/libsndfile_*.so",
    "/opt/uv_cache/archive-v0/*/_soundfile_data/libsndfile_*.so",
    "/opt/nemo_rl_venv/lib/python3.*/site-packages/soundfile.py",
    "/opt/ray_venvs/*/lib/python3.*/site-packages/soundfile.py",
    "/opt/uv_cache/archive-v0/*/soundfile.py",
}
BASE_VENV_MINIMUM_PYTHON_PACKAGE_VERSIONS = {
    "wandb": "0.28.2",
}

# Actor FQN -> packages that must import inside that actor's venv.
#
# MUST list every venv the build prefetches (the eight filters in docker/rl/Dockerfile.nhx-rl-base
# resolve to NINE actors, because `vllm.vllm_worker` matches the sync and async workers alike).
# Listing all nine is what makes a broken prefetch filter fail here instead of silently shipping an
# image whose workers rebuild their venv on the node at job start.
WORKER_VENV_IMPORTS = {
    # DPO policy training, DTensor V1 (--extra fsdp)
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": [
        "torch",
        "flash_attn",
        "mamba_ssm",
        "causal_conv1d",
    ],
    # GRPO policy training, DTensor V2 (--extra automodel). nemo_automodel and transformer_engine
    # are checked through distribution metadata instead — see WORKER_VENV_DRIVER_LINKED below.
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": [
        "torch",
    ],
    # GRPO policy training, Megatron (--extra mcore). Also hosts Megatron-native generation in
    # process. megatron_core / megatron_bridge / transformer_engine are checked through
    # distribution metadata — see WORKER_VENV_DRIVER_LINKED below.
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": [
        "torch",
    ],
    # GRPO generation (--extra vllm). deep_ep is checked separately — see
    # WORKER_VENV_DRIVER_LINKED below. This is the full check for the vllm tier; the two actors
    # after it share the same extra.
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": [
        "torch",
        "vllm",
        "deep_gemm",
    ],
    # Async GRPO generation (--extra vllm). NeMo-Gym forces async rollouts, so this actor is on the
    # GRPO+Gym path. Subset of the imports above: same extra, so deep_ep/deep_gemm resolve to the
    # same cache entries already validated by VllmGenerationWorker.
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": [
        "torch",
        "vllm",
    ],
    # GRPO rollout driver, sync path (--extra vllm)
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": [
        "torch",
        "vllm",
    ],
    # NeMo-Gym environment actor (--extra nemo_gym). opensandbox is the sandbox client SDK; it
    # reaches this venv only through RL's `nemo_gym = ["nemo_gym[sandbox]"]`, so this assertion is
    # what proves the extra really carries it.
    "nemo_rl.environments.nemo_gym.NemoGym": [
        "nemo_gym",
        "opensandbox",
    ],
    # Sandboxed-Gym mode B: the trusted proxy actor in the training pod. Shares the nemo_gym extra
    # with NemoGym, but venvs are named per ACTOR, so it gets its own and needs its own filter.
    "nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor": [
        "nemo_gym",
        "opensandbox",
        "sandboxed_gym",
        "nemo_rl.environments.sandbox.nemo_gym_actor",
    ],
    # Trusted episode broker: creates per-episode sandboxes so the untrusted job sandbox never
    # holds the OpenSandbox credential. Keyed by the FQN Ray derives the venv path from.
    "sandboxed_gym.ray.broker_actor.SandboxEpisodeBrokerActor": [
        "nemo_gym",
        "opensandbox",
        "sandboxed_gym.ray.broker_actor",
    ],
}

# Packages that link libcuda.so.1 — the NVIDIA *driver*, which the container runtime injects on a
# GPU host and which is deliberately absent from the image (the CUDA toolkit ships; the driver does
# not). Importing these on the CPU build runner fails with
# `ImportError: libcuda.so.1: cannot open shared object file` for purely environmental reasons, so
# presence is verified through distribution metadata instead — which still catches the case that
# matters here: the extra failing to install the package at all.
#
# Transformer-Engine belongs here for the same reason: importing it loads
# libtransformer_engine.so, which resolves against the driver and cuDNN. It is listed under BOTH
# training tiers on purpose — RL's [tool.uv] override-dependencies collapses every TE requirement
# onto one git rev, so a single build is shared, and asserting it in each venv is what proves the
# sharing actually happened rather than one tier silently missing it.
WORKER_VENV_DRIVER_LINKED = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": ["deep_ep"],
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": [
        "nemo-automodel",
        "transformer-engine",
    ],
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": [
        "megatron-core",
        "megatron-bridge",
        "transformer-engine",
    ],
}

WORKER_IMPORT_CASES = [(fqn, mod) for fqn, mods in sorted(WORKER_VENV_IMPORTS.items()) for mod in mods]
WORKER_DIST_CASES = [(fqn, d) for fqn, ds in sorted(WORKER_VENV_DRIVER_LINKED.items()) for d in ds]


def _import_in_venv(venv: Path, module: str) -> subprocess.CompletedProcess:
    """Import ``module`` using ``venv``'s own interpreter."""
    return subprocess.run(
        [str(venv / "bin" / "python"), "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )


# --- base venv: only what genuinely lives there -------------------------------------------------


@pytest.mark.smoke_nhx_rl_training
def test_base_python_package_min_versions():
    assert_python_package_min_versions(BASE_VENV_MINIMUM_PYTHON_PACKAGE_VERSIONS)


@pytest.mark.smoke_nhx_rl_training
def test_torch_importable():
    """torch is a default dependency, so it is present in the base (driver) venv."""
    import torch  # noqa: F401


@pytest.mark.smoke_nhx_rl_training
def test_nemo_rl_importable():
    import nemo_rl  # noqa: F401


@pytest.mark.smoke_nhx_rl_training
def test_nhx_rl_training_importable():
    # Exercises the full platform-glue import chain the training entrypoint pulls in
    # (nemo_helix SDK -> plugin -> nhx_common -> nhx_customization_common -> services/rl).
    from nhx.rl.tasks.training import __main__ as training_main  # noqa: F401


@pytest.mark.smoke_nhx_rl_training
def test_sandboxed_gym_driver_imports():
    """The driver must be able to import the mode-B sandbox modules.

    Sandboxed GRPO calls spinup_nemo_gym_actor() in the DRIVER process, which imports the
    adapter and sandboxed_gym.host.models at module scope, so the base venv needs both even
    though the Gym actor itself runs in its own venv. The `uv sync --all-groups` in the base
    image is exact and prunes the nemo_gym extra, so this is only satisfied by the explicit
    `uv pip install` of the Gym workspace member — without it, mode B fails at Gym spin-up
    with a ModuleNotFoundError, minutes into a run and only on a sandbox-capable cluster.
    """
    from nemo_rl.environments.sandbox.nemo_gym_actor import SandboxedGymActorConfig  # noqa: F401
    from sandboxed_gym.host.models import NemoGymSandboxedConfig  # noqa: F401


# --- per-worker venvs: where training actually runs ---------------------------------------------


@pytest.mark.smoke_nhx_rl_training
@pytest.mark.parametrize("actor_fqn", sorted(WORKER_VENV_IMPORTS))
def test_worker_venv_prefetched(actor_fqn):
    """Each actor's venv must be baked into the image, not built on the node at job start."""
    python = RAY_VENVS / actor_fqn / "bin" / "python"
    present = sorted(p.name for p in RAY_VENVS.iterdir()) if RAY_VENVS.is_dir() else []
    assert python.exists(), f"missing prefetched venv for {actor_fqn} (expected {python}); present: {present}"


@pytest.mark.smoke_nhx_rl_training
def test_prefetched_venvs_match_expected_set():
    """The prefetched venvs must be exactly the set this file covers — no more, no less.

    Catches prefetch-filter drift in both directions: a filter that stopped matching an actor
    (its venv would be rebuilt on the node at job start), and one that matched too broadly (e.g.
    a bare `vllm` filter also pulling in the modelopt-quant workers, inflating the image).
    """
    assert RAY_VENVS.is_dir(), f"{RAY_VENVS} does not exist; nothing was prefetched"
    found = {p.name for p in RAY_VENVS.iterdir() if (p / "bin" / "python").exists()}
    expected = set(WORKER_VENV_IMPORTS)
    assert found == expected, (
        f"prefetched venvs do not match this file's map.\n"
        f"  only in image: {sorted(found - expected)}\n"
        f"  only in map:   {sorted(expected - found)}\n"
        f"Keep WORKER_VENV_IMPORTS in sync with the prefetch filters in "
        f"docker/rl/Dockerfile.nhx-rl-base."
    )


@pytest.mark.smoke_nhx_rl_training
@pytest.mark.parametrize(("actor_fqn", "module"), WORKER_IMPORT_CASES)
def test_worker_venv_imports(actor_fqn, module):
    """Import each package in the venv that owns it (catches ABI mismatches and missing extras)."""
    venv = RAY_VENVS / actor_fqn
    if not (venv / "bin" / "python").exists():
        pytest.skip(f"venv for {actor_fqn} not present; covered by test_worker_venv_prefetched")
    result = _import_in_venv(venv, module)
    assert result.returncode == 0, f"`import {module}` failed in {venv}:\n{result.stderr}"


@pytest.mark.smoke_nhx_rl_training
@pytest.mark.parametrize(("actor_fqn", "dist"), WORKER_DIST_CASES)
def test_worker_venv_driver_linked_dists_installed(actor_fqn, dist):
    """Verify driver-linked packages are installed without importing them.

    These cannot be imported on a CPU build runner (no libcuda.so.1), so this reads their
    distribution metadata instead. A missing extra still fails here; a genuine ABI problem in
    them would only surface on a GPU.
    """
    venv = RAY_VENVS / actor_fqn
    if not (venv / "bin" / "python").exists():
        pytest.skip(f"venv for {actor_fqn} not present; covered by test_worker_venv_prefetched")
    result = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            f"import importlib.metadata as m; print(m.version({dist!r}))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{dist} is not installed in {venv}:\n{result.stderr}"


@pytest.mark.smoke_nhx_rl_training
def test_worker_venvs_symlink_into_shared_cache():
    """Worker venvs must symlink into /opt/uv_cache rather than carry their own copies.

    Guards the link-mode/cache layout: a silent fallback to copy mode would multiply image size
    across the ~30 venvs this image ships.
    """
    venv = RAY_VENVS / "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"
    if not (venv / "bin" / "python").exists():
        pytest.skip("policy worker venv not present; covered by test_worker_venv_prefetched")
    torch_init = next(venv.glob("lib/python*/site-packages/torch/__init__.py"), None)
    assert torch_init is not None, f"torch not found in {venv}"
    assert torch_init.is_symlink(), f"{torch_init} is not a symlink; link mode fell back to copy"
    assert str(torch_init.resolve()).startswith("/opt/uv_cache/"), (
        f"{torch_init} resolves outside the shared cache: {torch_init.resolve()}"
    )


@pytest.mark.smoke_nhx_rl_training
def test_soundfile_libsndfile_removed():
    patterns = read_file_patterns(FINAL_FILE_REMOVALS)
    assert SOUNDFILE_FILE_REMOVALS.issubset(patterns)
    assert_file_patterns_absent(patterns)


@pytest.mark.smoke_nhx_rl_training
def test_transformers_audio_backend_probe_is_off():
    """Removing the codec must also switch off the probe that guards its import.

    ``transformers.audio_utils`` does ``if is_soundfile_available(): import soundfile``,
    and that probe is ``find_spec("soundfile")`` -- file presence, not loadability. Delete
    the codec but keep the module and the probe says yes to a backend that then fails to
    dlopen, so ``from transformers import AutoProcessor`` raises. That import is at module
    scope in ``nemo_rl.algorithms.grpo``, so the GRPO driver dies before it reads a config.
    """
    from transformers.utils.import_utils import is_soundfile_available

    assert not is_soundfile_available()


@pytest.mark.smoke_nhx_rl_training
def test_grpo_driver_module_imports():
    """The exact import the GRPO driver performs first, and the one the codec strip broke."""
    from nemo_rl.algorithms.grpo import MasterConfig  # noqa: F401


@pytest.mark.smoke_nhx_rl_training
def test_no_git_directories_shipped():
    """The published image must not carry repository history.

    A ``.git`` tree would expose full history and, for a local-checkout build, that checkout's
    remotes and any credentials embedded in them.
    """
    source_root = Path("/opt/nemo-rl")
    if not source_root.exists():
        pytest.skip("/opt/nemo-rl not present")
    found = [str(p) for p in source_root.rglob(".git")]
    assert not found, f"unexpected .git entries in the image: {found[:5]}"
