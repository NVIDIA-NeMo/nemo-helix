---
name: helix-specialized-agent-tutorial
description: Use when a user wants to run, resume or troubleshoot the NeMo Helix specialized-agent tutorial in this repo (the Meridian Compute incident agent from the GTC Berlin session), including scripts/preflight.sh, any numbered step in docs/tutorial.md, a preflight FAIL line or an error that may be in KNOWN-ISSUES.md, what works without a GPU or on a Mac, or applying the recipe (taxonomy, policy, data recipe, fine-tuned triage model) to their own agent.
compatibility: NeMo Helix 0.7.x from PyPI (nemo-helix[all] in requirements.txt; training images tagged 0.7.0 and 0.7.1 on nvcr.io) with its nemo CLI; Python 3.12 or 3.13 with uv; Docker; steps 6 to 10 need Linux x86_64 with an NVIDIA GPU of 40 GB or more, CUDA 13 and the NVIDIA Container Toolkit; network access to an inference provider and huggingface.co.
---

# Guide a user through the Helix specialized-agent tutorial

`docs/tutorial.md` is the source of truth. This skill covers how to walk someone through it: which path to take, how to gate each step and where to look when something breaks. Don't paraphrase the tutorial; open the step and run what it says.

## Say this first

In your first reply, before any commands, give the time estimate. With the packaged data on one A100 80GB, about 2 to 4 hours end to end plus downloading about 21 GB of training images in step 7a: steps 0 to 6 about 30 minutes, training about 30 minutes, deploying 20 to 45 minutes, evaluations about 15 minutes, and the agent comparison 30 minutes for 50 incidents or about 2 hours for 200.

Skip it when the user only wants to adapt the recipe or fix one error.

## Ask only what you don't know

Read the user's message first. Ask only the questions it doesn't answer, all in one message, and offer the default for each so they can reply "defaults are fine".

| Question | Default if they don't care |
|---|---|
| Which machine: Linux x86_64 with an NVIDIA GPU, or a laptop? Picks the path in `reference/paths.md` | None; you need this one |
| Which Helix, and more than one `nemo` CLI context? | Local Helix at `http://localhost:8080`, one context. For remote, see `reference/paths.md` |
| Which provider, and the exact model names for the agent's default model and the baseline (plus writer and judge if generating) | None; read them from `nemo models list` after step 1 |
| Packaged or generated data? | Packaged data in `data/meridian-example/` (step 3) |
| Budget and time limit | Ask before each step that spends money or holds a GPU (below) |

## Run the tutorial one step at a time

On a fresh machine, clone the repo first (the `git clone` line in step 1), then start with `bash scripts/preflight.sh` (add `--no-gpu` on a laptop). uv, Docker, the NVIDIA driver with CUDA 13 and the container runtime must pass before step 1; step 1 doesn't install them. On a fresh clone, FAILs fixed by step 1 are expected (`reference/failures.md` lists which); do step 1 and rerun preflight.

GPU rule: steps 6 to 10 run on the GPU machine. A GPU, CUDA or NVIDIA container runtime FAIL blocks steps 7 and 8 directly (training and serving run in GPU containers), and steps 9 and 10 call the step 8 deployment. So don't start step 6 on a machine with any of those FAILs. Before jumping to a later step, check the dependency list in `reference/steps.md`.

Then for each step in order:

1. Open the step in `docs/tutorial.md` and run its commands as written, from the repo root, with `.venv` active.
2. Compare the output with the step's **You should see**, using the gate in `reference/steps.md`. Numbers marked "in our run" are for direction, not exact matches.
3. If it passes, say so in one line and move on.
4. If it fails, stop. Find the error in `reference/failures.md` and read the KNOWN-ISSUES entry it names.
   - Entry marked expected: tell the user and continue.
   - Entry says wait: wait. Don't rerun.
   - Otherwise apply the fix and rerun the step once. For steps 2, 3 (generate), 7, 8, 9 and 10, ask before the rerun, since each spends provider credits or GPU time.
   - Nothing matches: show the user the exact error and the command, and stop.

## Steps that need a go-ahead

Ask before each of these, even if the user said "run it all":

- Step 2 (calls the provider, spends credits) and step 3 generation (provider money, hours on a free tier).
- Step 7a image pull (about 21 GB; the source build fallback takes 30 to 60 minutes), 7b training and 8 deploy (hold the GPU).
- Step 9 (the deployment keeps its GPU while it runs).
- Step 10's `BATCHES="1 2 3 4"` run (about 2 hours).
- Restarting the Helix services and every delete in the cleanup.

## Long jobs

Running a documented blocking command once is fine: `nemo jobs watch <job>`, or `training/deploy_model.py`, which waits for the deployment on its own. What you don't do is write your own loop of `sleep` and status checks. For a job with no blocking command, show the status once and tell the user the command to check it.

## Step 7a: pull the published images

The training images are public on `nvcr.io/nvidia/nemo-helix/` (anonymous pull, since 2026-10-11). Run step 7a as written: it reads the installed Helix version, pulls `nhx-customizer-tasks` and `nhx-unsloth-training` with that tag and tags them `my-registry/<image>:local`, because a pip-installed Helix (0.7.1) still defaults to those names. Step 8 needs the `my-registry/nhx-customizer-tasks:local` tag too, so don't skip the `docker tag` lines.

- **Pull fails** (`manifest unknown`, not found): there's no published tag for the installed version. Show the version, and offer the two options in 7a: install a Helix version with images, or the source build fallback (`scripts/gpu-build-images.sh`, 30 to 60 minutes). Let the user choose.
- **7b still fails pulling `nhx-unsloth-training` or `nhx-customizer-tasks`:** step 7a gives the override (`NHX_UNSLOTH_TRAINING_IMAGE`, `NHX_CUSTOMIZER_TASKS_IMAGE` set to the nvcr.io names, then `nemo services restart`). The restart affects running services, so tell the user and get a yes first. If the override doesn't fix it, stop and ask; don't try other settings.

## Step 10: write the temperatures into the agent

Before creating the step 10 agent, edit `agent/agent-decision-model.yaml` under `mcp.servers.meridian-ops.env`:

- `ROUTER_TEMPERATURE_ACTION`: the `temperature=` value from the `calibration:` line `eval/analyze_results.py` printed for the action rows in step 9.
- `ROUTER_TEMPERATURE_RUNBOOK`: step 9 only calibrates the action question. Step 10 in `docs/tutorial.md` shows how to fit this one with `--question runbook`; if the user skips that, keep ours (`"1.40"`).
- `ROUTER_URL` and `ROUTER_MODEL` if their deployment or adapter name differs from ours.

Keep the values quoted. Show the user `git diff agent/agent-decision-model.yaml` and get a yes before `nemo agents create`.

## Adapting to the user's own agent

Use `reference/adapt.md`.

## Cleanup

When the user is done, walk through the "Clean up" section at the end of `docs/tutorial.md`. Ask before each delete, and show sizes before removing images or the Helix checkout. Step 10 needs the step 2 agent deployment, so don't delete it mid-tutorial. Skip deletes for anything the user never created.

If the user is stopping for the day but coming back, offer to delete only the step 8 deployment (`meridian-qwen`). That frees the GPU, and redeploying from the saved adapter takes 20 to 45 minutes. Keep the adapter, filesets, images and the step 2 deployment.

If the machine is rented, remind the user to stop or delete it with their provider when they're done, since an idle GPU box keeps billing.

## Gotchas

- The user sets their own API keys in their shell. Never echo, print or commit a key, and never put one in a file or command you write.
- Product names: the Helix post-training plugin and the Helix Evals plugin. `nemo customization` and `nemo evaluator` are CLI subcommands; use them only inside commands.
- Speed: the fine-tuned model is faster measured on its own. A whole agent run takes about the same time either way (A7).
- Only suggest commands from `docs/tutorial.md`, `KNOWN-ISSUES.md`, a preflight `fix:` line or `--help`. For anything else, such as installing the NVIDIA Container Toolkit, point to the vendor's install guide (P6 has the link).
