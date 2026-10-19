# Agent instructions

This repo is a tutorial: fine-tune a small open model to take over one decision inside an incident agent at Meridian Compute (a made-up company) with NeMo Helix. Data Designer generates the data, the Helix post-training plugin trains the model, Deployments serve it and the Helix Evals plugin scores it.

## How to help a user run it

- Run `bash scripts/preflight.sh` first (`--no-gpu` on a laptop). On a fresh clone, FAILs for the nemo CLI, services, provider and packages go away after step 1.
- Follow the steps in `docs/tutorial.md` in order. After each one, compare the output with its **You should see** block. If it doesn't match, stop and find the fix in `KNOWN-ISSUES.md`.
- For a first run, use the packaged data in `data/meridian-example/` (step 3).
- Only run commands that appear in `docs/tutorial.md`, `KNOWN-ISSUES.md` or a preflight `fix:` line.
- Run commands from the repo root with `.venv` active.

The skill in `skills/helix-specialized-agent-tutorial/` has the step gates, the error-to-fix map and the guide for adapting the recipe to another agent. Read its `SKILL.md` when a user asks to run, troubleshoot or adapt the tutorial.

## Safety

- Never print, log or commit an API key or token. The user sets keys in their own shell.
- Ask before steps that hold a GPU or cost money: the step 2 agent run, data generation, pulling or building the training images, training, deploying, evaluations and long agent evaluations.
- When the user is done, walk through the "Clean up" section at the end of `docs/tutorial.md`, asking before each delete. Deployments hold their GPU until deleted.
- Running a documented blocking command once (`nemo jobs watch`, `training/deploy_model.py`) is fine. Don't write your own sleep-and-check loop; otherwise show a long job's status once and stop.

## Editing the site

`docs/index.html` is written by hand. `docs/tutorial.html` is generated from `docs/tutorial.md`; after editing the markdown, run `python scripts/build_site.py` (needs pandoc) and commit both files.

## Names

Say "NeMo Helix", "Data Designer", "the Helix post-training plugin" and "the Helix Evals plugin". The CLI subcommands `nemo customization` and `nemo evaluator` appear only inside commands. Speed claims apply to the fine-tuned model measured on its own; a whole agent run takes about the same time.
