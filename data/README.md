# Data: use ours, or generate your own

The tutorial covers both paths: [step 3](../docs/tutorial.md#step-3-get-the-incident-history) to get the data, [step 4](../docs/tutorial.md#step-4-check-the-data) to check it and [step 5](../docs/tutorial.md#step-5-upload-the-data-as-filesets) to upload it.

## The packaged Meridian dataset

`meridian-example/` holds the exact data behind [RESULTS.md](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/RESULTS.md), compressed:

| File | Rows | What it is |
|---|---|---|
| `train.jsonl.gz` | 5,006 | 2,503 incidents, two questions each (which runbook, which action), as prompt and completion pairs |
| `validation.jsonl.gz` | 474 | 237 incidents, same format, for validation loss |
| `calibration.jsonl.gz` | 240 | Incidents with labels and both prompts, for fitting confidence and the threshold |
| `test.jsonl.gz` | 416 | Test incidents with labels and both prompts; 116 come from scenario types never seen in training |

`python scripts/check_data.py` prints counts, label balance and the checks for answers given away in the text.

## Generating your own with Data Designer

`incidents.py` is the Data Designer recipe. It reads `../taxonomy.json` (your runbooks, actions, policy, service names, scenario types and private notes) and writes labeled incidents. Edit the taxonomy for your company; the recipe reads everything from it.

Settings, as environment variables:

| Setting | Default | What it is |
|---|---|---|
| `MERIDIAN_SPLIT` | `train` | `train` leaves out the held-out scenario types; `test` includes them |
| `MERIDIAN_PROVIDER` | `default/openrouter` | The Helix provider for the writer and judge |
| `MERIDIAN_WRITER_MODEL` | `nvidia/nemotron-3.5-lightning` | Writes the incidents |
| `MERIDIAN_JUDGE_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Scores label support, whether the text gives away the answer, and realism |
| `MERIDIAN_EXTRA_BODY` | turns reasoning off for Nemotron | Set `'{}'` for providers that reject unknown fields |

Use different models for the writer and the judge, and neither should be the model you compare against later.

What to expect:

| Provider | What we saw |
|---|---|
| A hosted provider with high rate limits | 3,600 training and 500 test rows in 20 min 40 s, in parallel; $1.29 at OpenRouter list prices |
| build.nvidia.com free tier | About 0.07 rows a second; 3 to 5 hours for 3,600 rows. Fine for a 10-row smoke test |

The quality gates in `../scripts/split_dataset.py` dropped 17% of our generated training rows, mostly because the text gave away the answer.
