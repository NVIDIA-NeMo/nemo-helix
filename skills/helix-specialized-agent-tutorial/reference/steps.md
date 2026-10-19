# Step gates

The commands and full expected output live in `docs/tutorial.md`. This table says what to check before moving on, who runs it and what needs the user's go-ahead. Where a number is only "ours", read it from the step's **You should see**; a different value isn't a failure.

"User" means the user runs it or confirms first. "Agent" means you can run it once the user has agreed to start the step.

| Step | Where | Who | Pass when | Ask first because |
|---|---|---|---|---|
| 0 Preflight | Any | Agent | Last line `Result: ready (N warning(s))`, exit 0. On a fresh clone, FAILs fixed by step 1 are expected | |
| 1 Install, provider | Any | User for the key and `nemo setup` prompts; agent for the `uv` lines | `{"status":"ready"}`, a provider with status `READY`, model names written down | The key is the user's. Start services from the activated `.venv` (A2) |
| 2 Agent baseline | Any | User confirms | `ROUTER_MODEL` in `agent/agent-with-context.yaml` matches a name from `nemo models list`; the reply has a runbook ID on line 1 and an action on line 2. A wrong action is fine: it's the baseline | Spends provider credits |
| 3 Data | Any | Agent for packaged; user confirms generation | Packaged: nothing to run. Generated: validate and check-models pass, each job ends `completed` | Generation costs provider money; about 14 hours on a free tier (D1) |
| 4 Check data | Any | Agent | `PASS: data looks ready to train on`, with 0 incidents that give away the answer | |
| 5 Upload | Any | Agent | `nemo files list` shows `train.jsonl`, `validation.jsonl` and `calibration.jsonl` | |
| 6 Register base model | GPU | Agent | `model spec resolved: True` | Steps 0, 1, 2 and 5 must be done on this machine too |
| 7a Images | GPU | User confirms | Both `nhx-` images pulled from nvcr.io with the installed Helix version's tag, both `my-registry/...:local` tags exist, and the vLLM image is pulled. Fallback: the build finishes and both `my-registry/...:local` images exist | About 21 GB to download; the build fallback takes 30 to 60 minutes |
| 7b Train | GPU | User confirms | Job ends `completed` and the loss falls | Holds a GPU for the length of the job |
| 8 Deploy | GPU | User confirms | `gateway ready: True`; `providers get` lists the base and the adapter. The adapter's top answer should be an action letter with most of the probability | Holds a GPU until deleted; slow to come up (S2) |
| 9 Evaluate | GPU | User confirms | `analyze_results.py` prints JSON per model; fine-tuned beats untuned on the action by a wide margin. Note the `temperature=` on the `calibration:` line | The deployment holds its GPU meanwhile |
| 10 Agent swap | GPU | User confirms the YAML diff, then agent for one incident; user confirms `BATCHES="1 2 3 4"` | Temperatures written into `agent/agent-decision-model.yaml` and shown as a diff; the incident returns a runbook and an action; `analyze_agent.py` prints one line per agent | Four batches take about 2 hours; each must finish inside the 1-hour job limit (E6) |
| 11 Adapt | Any | Both | See `adapt.md` | |

## What each step needs

Use this to answer "can I jump to step N?". "Same Helix" means the step reads something an earlier step created on that Helix, so after moving machines it has to be redone there.

| Step | Needs |
|---|---|
| 1 | Docker running (0 shows it) |
| 2 | 1 |
| 3 | Packaged: nothing. Generated: 1 |
| 4 | 3 (packaged data needs nothing else) |
| 5 | 1, 4; packaged data is unzipped in step 5 itself |
| 6 | 1 on the GPU machine; preflight 0 passing there |
| 7a | 1 on the GPU machine (the pull tag comes from the installed Helix version); Docker and the NVIDIA container runtime passing preflight |
| 7b | 5 and 6 on the same Helix, 7a |
| 8 | 6 and 7b (the adapter) on the same Helix |
| 9 | 5 (test and calibration files) and 8 running |
| 10 | 2 on the same Helix (the baseline agent deployment), 8 running, 9's calibration temperature |
| 11 | Nothing; reads code and step 11 |

So jumping straight to step 7 on a fresh GPU machine means doing 0, 1, 4, 5 and 6 there first, and 2 as well if they want step 10.

## Cleanup

Use the "Clean up" section at the end of `docs/tutorial.md`. Ask before each delete. Delete agent deployments before stopping the Helix services (P3). If the user stops after step 2 and plans to come back for step 10, keep the step 2 deployment or redeploy it later.
