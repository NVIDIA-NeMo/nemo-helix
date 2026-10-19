# Adapt the recipe to the user's agent

Step 11 in `docs/tutorial.md` has the method. This file adds the questions to ask first and the code facts step 11 doesn't spell out.

## First reply: questions plus the facts that already apply

In one message, ask the questions below that the user hasn't answered, and list the code facts from the table that already apply to what they told you (for example, 20 options is more than `LETTERS` holds, and 20 options hits the top-20 logprobs limit). Don't propose file edits yet; the edits depend on the answers. Propose them after the user replies.

1. Which step of their agent, and what are its options? How many, and is there a second question that depends on the first (like runbook, then action)?
2. Real labeled examples or none? If real: how many per option, including the rare ones? Is the label the right answer or only what happened (a ticket that bounced between queues has a misleading final label)?
3. What does the step see at decision time? Anything added later (resolution notes, final queue) gives away the label.
4. Is the second answer a fixed function of the first plus something in the text (Meridian's action is runbook plus severity), or a judgment call?
5. Date range of the data, and did the options or policy change during it?
6. Does the data hold personal or customer information that has to be removed before upload?

## Code facts to plan around

| Fact | Where | What it means for them |
|---|---|---|
| Option letters are fixed: `LETTERS = "ABCDEFGHIJKL"` (12) and `ACTION_LETTERS = "WXYZ"` (4), paired with `zip()` | `agent/mcps/prompting.py` | More than 12 first-level options or 4 second-level options get dropped silently. Extend the strings, keep the two sets from overlapping and keep each letter a single token. With 20 options, `A` to `T` for the first question and `WXYZ` for the second don't overlap. |
| Both prompt renderers say "Meridian Compute's GPU platform operations" and "incident" | `agent/mcps/prompting.py` | Change the wording. Data generation, training rows, the agent tool and the eval specs all render through these functions. |
| `private_context` and `policy` live on each runbook, not at the top level | `taxonomy.json` | `private_context` is what the "notes in the prompt" baseline gets. Keep it per option. |
| `relabel_actions()` overwrites every action with `policy[runbook][severity]` | `scripts/split_dataset.py` | Right for Meridian's deterministic policy. With real labels that aren't a fixed function, remove it or the model learns the lookup table instead of their data. |
| Leak checks come from `LEAK_PATTERNS` | `data/incidents.py`, imported by `scripts/split_dataset.py` and `scripts/check_data.py` | Update the patterns to their own option names, even if they skip generation. |
| Held-out scenario types come from `holdout_families` | `taxonomy.json`, read by `data/incidents.py`, `scripts/check_data.py`, `scripts/split_dataset.py` and `eval/analyze_results.py` | With real data, hold out a later time period instead (step 11 says so) and expect the held-out report to be empty. |
| Logits mode asks for the top 20 next tokens (`"logprobs": 20`) and scores only option letters found there | `target()` in `eval/make_eval_specs.py`, `_logit_letter()` in `agent/mcps/ops_tools.py`, the logits call in `eval/bench_latency.py`; letters are read back in `load()` in `eval/analyze_results.py` | See "Many options and the top-20 limit" below. |
| `max_seq_length: 2048` | `training/job.json`, `training/job-nemotron-nano.json` | Long inputs plus many option definitions can exceed it. Check the longest rendered prompt. |
| Model, deployment and adapter names are hard-coded per arm | `eval/make_eval_specs.py` (`ARMS`), `agent/agent-decision-model.yaml`, `eval/specs-agent/*.json`, `scripts/run-agent-evals.sh` | Rename to match what they register and deploy. |
| The agent's tools and instructions are Meridian-specific | `agent/mcps/ops_tools.py`, `agent/agent-*.yaml` | Replace them, but keep the decision step as a tool driven by `ROUTER_MODEL` and `ROUTER_MODE` so the model can be swapped with a setting. |

## Many options and the top-20 limit

The model returns its 20 most likely next tokens. The code keeps those that are an option letter (after stripping spaces, so `"A"` and `" A"` both count as A) and renormalizes over the letters it found. A letter outside the top 20 gets no score, can't be picked, and its row counts as wrong.

- More than 20 options: at least N minus 20 options never get a score, on every call.
- Exactly 20: all 20 fit only if the top 20 tokens are exactly one form of each letter, with no newline or other token among them. In practice some drop out.
- Under 20: a letter the model thinks unlikely can still fall out. That mostly affects confidence, which is overstated because it's renormalized over fewer letters.

Fixes, either one:

1. Raise `logprobs` in all three places. A provider caps it; vLLM rejects values above its `--max-logprobs` server setting, and `training/deploy_model.py` doesn't set that flag today, so raising it means adding the flag to the deployment's `additional_args` and redeploying. Check the cap in the vLLM docs for the image you deploy.
2. Split the first question into two smaller ones (group, then option within the group), each under about 15 options. This changes the prompts, the training rows and the agent tool, so it's more work but stays inside the limit.

`ROUTER_MODE=chat` avoids the limit for the agent, since it reads the answer as text, but it returns no confidence, and the eval specs for the fine-tuned arms use logits mode, so step 9's scores and calibration still need fix 1 or 2.

## Real labeled data instead of generation

Step 11's "Bring your own labeled tickets" subsection has the user-facing version with code. There's no path for real data in the scripts today: `scripts/split_dataset.py` reads a Data Designer job and splits at random, balanced by option, and `--split test` keeps the held-out scenario types. For real tickets, build the files yourself before step 5:

1. Sort the tickets by date. Keep the latest period (for example the last two or three months) as the test set, and split the earlier tickets into train, validation and calibration (the tutorial uses 84%, 8% and 8%).
2. Write `train.jsonl` and `validation.jsonl` as prompt and completion rows, two per ticket, the way `train_rows()` in `scripts/split_dataset.py` does: `{"prompt": render_runbook_question(text, tax), "completion": " " + letter_for_runbook(id, tax)}` and `{"prompt": render_action_question(text, id, tax), "completion": " " + letter_for_action(action, tax)}`. The functions are in `agent/mcps/prompting.py`. Keep the leading space in the completion.
3. Write `calibration.jsonl` and the test file (`eval.jsonl`) with the fields `eval_row()` in `scripts/split_dataset.py` writes, since the eval specs and `eval/analyze_results.py` read them.
4. Run `python scripts/check_data.py <train dir> <test dir>` on the result, then upload as in step 5.

Expect the held-out family report to be empty, because real tickets have no scenario types.

## Then

Rerun steps 4 to 10 on their data. Compare against a general model with their notes in its prompt, and train more than one seed before trusting a difference (step 11). For how much data to plan for, quote the step 11 paragraph that starts "Our dataset had about 200"; don't make up a number.
