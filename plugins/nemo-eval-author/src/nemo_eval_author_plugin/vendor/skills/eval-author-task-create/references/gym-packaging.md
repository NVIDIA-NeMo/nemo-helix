<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gym component dependencies and packaging

Use these checks when completing a Gym task and its version/rerun plan. They
adapt NeMo Helix's creation and packaging practices to task authoring. Native
Gym tasks do not require a Helix installation or platform upload.

## Lock every environment that executes the task

Gym creates separate environments for its servers. A lock for Eval Author or
the Gym parent interpreter does not lock the agent, resources server, and model
server environments. Record the actual host image digest or source revision and
the dependency inputs for each selected component in `reproducibility.md`.

- Keep lock projects in `reproducibility/locks/<component>/`, separate from the
  native task layout. For Gym v0.6.0, placing `pyproject.toml` two levels above a
  component changes its editable-install detection. Do not introduce that file
  at the draft/package root merely to hold a lock.
- Preserve each custom implementation's dependency entrypoint: `requirements.txt`
  or `pyproject.toml`, not both in one server directory. Inspect the installed
  Gym setup command, export or constrain the selected pins into the inputs it
  actually consumes, and verify the installed result. An unused `uv.lock` next
  to an unconstrained requirements file does not fix component versions.
- Resolve for the execution Python version, OS, and architecture. Include the
  host's injected Gym and head-server dependencies; derive their versions from
  the selected host runtime or its build lock. Helix's task image, sandbox host,
  and RL image can use different locks. Do not substitute the repository-wide
  lock or assume the upstream Gym lock matches the deployed image.
- Preserve component instance names, implementation directory names, and
  references as distinct identities. Check that the composed config selects the
  intended custom code rather than a same-named built-in. Avoid duplicate or
  cross-type instance names and retain the resolved implementation paths.

## Package dependencies for the target runtime

Choose index-backed installation or a vendored wheelhouse based on the target's
network access. For a wheelhouse, retain one resolved version per distribution,
wheel filenames and SHA-256 digests, the input lock, and target compatibility.
Include transitive dependencies, the selected Gym build, host-injected packages,
and any seed/build requirements used while creating fresh server environments.

A successful package-layout check does not validate wheel tags or dependency
closure. A successful `--no-index` package install does not prove every server's
setup is offline: Gym's child installs may still fall back to an index. For an
offline claim, verify the complete startup with fresh environments and caches,
with index/network access disabled under the target's supported configuration.
Report missing artifacts rather than resolving newer packages during a rerun.

## Prove the packaged task executes

During authorized validation, exercise the task from the exported inputs in a
fresh temporary workspace using the selected runtime. Check imports in both the
host and the actual child component environment, and record package versions
and source locations from those processes. Run the positive and negative
controls through the real HTTP tools and verifier after restoring fixture state.

Where deployment mounts the package read-only, keep installation caches,
virtual environments, mutable state, and outputs in the designated writable
work directory. Test that layout instead of relying on an editable checkout.
Retain logs for layout validation, dependency setup, startup, controls, and
repeated runs separately; identify the stage that failed. Existing scripted
HTTP controls do not by themselves prove fresh per-component installation or
deployment compatibility.

## Optional export for Helix Evaluator

Apply this section only when the user selects Helix Evaluator as the consumer.
Confirm the target Helix revision's contract before packaging. Keep the native
Gym draft and local verifier fixtures intact; prepare a separate export under
`.eval-author/task-drafts/<task-slug>/exports/helix/` with its own file digests.

At the source revision below, Evaluator accepts `native-v1` and `wheels-v1`:

```yaml
format: native-v1
config_paths:
  - resources_servers/custom_task/configs/custom_task.yaml
metadata:
  name: custom-task
  description: The task's observed-outcome verifier and tools.
```

Write this manifest as `nemo-environment.yaml` at the export root. It is a Helix
package descriptor, separate from Gym's `environments/<name>/manifest.yaml`.

- Make every config path relative, contained, and present. For `native-v1`, paths
  must be under `resources_servers/` or `responses_api_agents/`. Include custom
  implementation code and the dependency files its server setup consumes.
- Keep all `.jsonl` files out of the environment export, including local verifier
  fixtures. Retain them in the draft and supply evaluation rows as the separate
  dataset/taskset required by Evaluator; do not copy the whole draft blindly.
- Keep model configuration operator-owned: do not export `responses_api_models/`
  or model-server blocks inside another config. Record the selected platform
  model and its settings in the run plan instead.
- For `wheels-v1`, use that format value and add a flat, non-empty `wheels/`
  containing only compatible `.whl` files, with one resolved version per
  distribution. Wheel presence alone does not establish offline startup.
- Use the selected Helix installation's
  `sandboxed_gym.environment_package.load_environment_package`, component
  inspection, and namespace validation before deployment. These checks validate
  layout and composition; follow them with the execution checks above.

The RL/Customizer package contract is different; an RL adapter package is not
automatically an Evaluator environment. Use the selected consumer's validator.
Preparing this export does not submit a job or upload a FileSet. Hand off its
path, digests, runtime requirements, dataset identity, and validation evidence
for any separately requested platform execution.

## Source contracts

These practices were checked against NeMo Helix commit
`a101f9523e20867e0a1c511189951484c3327056` and Gym v0.6.0. Recheck the selected
runtime's source/help when using another version:

- [Helix package validation](https://github.com/NVIDIA-NeMo/nemo-helix/blob/a101f9523e20867e0a1c511189951484c3327056/packages/sandboxed_gym/src/sandboxed_gym/environment_package.py)
- [Helix host and child dependency installation](https://github.com/NVIDIA-NeMo/nemo-helix/blob/a101f9523e20867e0a1c511189951484c3327056/packages/sandboxed_gym/src/sandboxed_gym/runtime/gym_host_runtime.py)
- [Helix image lock projects](https://github.com/NVIDIA-NeMo/nemo-helix/blob/a101f9523e20867e0a1c511189951484c3327056/docker/locks/README.md)
- [Gym per-server setup](https://github.com/NVIDIA-NeMo/Gym/blob/3045a793346a31291d7ea4ae6af3f94a35036ce5/nemo_gym/cli/setup_command.py)
