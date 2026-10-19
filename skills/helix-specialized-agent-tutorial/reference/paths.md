# Pick a path

Ask which machine the user has, then pick one row. Steps are in `docs/tutorial.md`.

| User has | Path | Notes |
|---|---|---|
| Nothing set up, just wants the numbers | `README.md` "Just want to look at our numbers?" block, plus the recompute commands in `RESULTS.md` | No Helix, GPU or key needed. Reads `results/`. |
| A laptop or a Mac, no NVIDIA GPU | Steps 0 (`--no-gpu`), 1, 2, 3 (packaged data), 4, 5, then the results-only commands above | Needs Docker Desktop running for the local Helix services. Steps 6 to 10 need a Linux GPU machine; don't try them here. |
| A Linux x86_64 machine with an NVIDIA GPU | Steps 0 to 10 in order, then Clean up | The authors used one A100 80GB. Under 80 GB, preflight warns; see the batch-size note in step 7. |
| A laptop for the early steps, a GPU machine for the rest | Steps 0 to 5 on the laptop, then steps 0, 1, 2, 5 and 6 to 10 on the GPU machine | Step 6 says so. Step 10's baseline agent eval calls the step 2 agent deployment on the same Helix, so step 2 has to be done on the GPU machine too. |

## Data: packaged or generated

For a first run, recommend the packaged data (step 3, "Recommended for a first run"). It's the dataset behind `RESULTS.md`, so the user can compare their numbers to ours.

Generate only if the user asks or is adapting the recipe. Warn about time and cost first, using step 3's **You should see** and KNOWN-ISSUES D1: about 0.07 rows a second on the free build.nvidia.com tier, so the 3,600 training rows take about 14 hours, against 21 minutes on a provider with high rate limits. A `create -n 10` smoke test is fine on any tier.

## Already cloned

If the user is already in the repo, skip the `git clone ... && cd` line in step 1 and start at `uv venv`.

## Remote Helix

The tutorial assumes `http://localhost:8080`. For a remote Helix:

- Preflight reads `NHX_BASE_URL` (see the header of `scripts/preflight.sh`).
- Pass `--base-url` to `nemo` commands when there's more than one context (P2).
- Python scripts need a token from the CLI (P5).
- `agent/agent-decision-model.yaml` and `eval/specs-agent/*.json` hold localhost URLs. Tell the user these need their Helix URL; don't guess it.
