# Results

What we measured when a small fine-tuned open model took over one decision in the Meridian incident agent: which runbook owns an incident and what action to take. Runs date from 2026-10-07 and 2026-10-08. Every row-level score behind these tables is in [`results/`](results/), so you can recompute them without calling any model.

Meridian Compute is made up and its incident history is synthetic.

## Setup

- **Test set:** 416 synthetic incidents never seen in training. 116 of them come from scenario types held out of training entirely.
- **Training data:** 2,503 incidents (5,006 records, two questions each), generated with Data Designer. Nemotron 3.5 Lightning wrote the incidents and Nemotron 3 Super judged them. Both differ from the large model we compare against (Nemotron 3 Ultra).
- **Fine-tuning:** the Helix post-training plugin with the Unsloth backend, LoRA rank 16, two epochs, one A100 80GB per job.
- **Models fine-tuned:** Qwen3.5-4B and Nemotron Nano 9B v2, a small model from NVIDIA's open Nemotron family.
- **Serving:** Helix Deployments (vLLM), with the base model and its adapters on one deployment.
- **Scoring:** the Helix Evals plugin, plus [`eval/analyze_results.py`](eval/analyze_results.py) for macro-F1 and calibration.

The trained model reads the probability of each option in one forward pass instead of writing an answer (a decision model shape). That gives a confidence with every answer.

## The two questions in the triage step

The triage step answers two questions for every incident: which of 12 runbooks owns it, and which of 4 actions Meridian's policy calls for.

| Model | Right runbook | Right action | Action macro-F1 | Action calibration error |
|---|---|---|---|---|
| Nemotron 3 Ultra 550B, no Meridian notes | 99.3% | 45.4% | 0.33 | n/a |
| Nemotron 3 Ultra 550B, Meridian notes in the prompt | 99.3% | 70.0% | 0.67 | n/a |
| Strands Decider 2B, off the shelf | 98.6% | 38.9% | n/a | n/a |
| Qwen3.5-4B-Base, untuned | 93.8% | 37.7% | 0.20 | 0.20 |
| Nemotron Nano 9B v2, untuned | 96.9% | 35.8% | 0.16 | 0.46 |
| **Qwen3.5-4B, fine-tuned in Helix** | **100%** | **82.0%** | **0.77** | **0.048** |
| **Nemotron Nano 9B v2, fine-tuned in Helix** | **99.3%** | **80.3%** | **0.76** | **0.033** |

We trained each fine-tune three times with different random seeds (29, 30 and 31) and scored each on the same test set:

| Fine-tuned, 3 seeds | Right runbook | Right action |
|---|---|---|
| Qwen3.5-4B | 99.8% ± 0.3 | 81.9% ± 0.1 |
| Nemotron Nano 9B v2 | 99.4% ± 0.1 | 80.1% ± 0.3 |

On the held-out scenario types alone, the fine-tuned models picked the right runbook 100% (Qwen) and 98.3% (Nemotron) of the time, and the right action 80.2% and 76.7%.

What the tables say:

- Picking the runbook is easy for any capable model once the runbook definitions name the internal services. Every model scores above 93%, and fine-tuning closes the last gap.
- The action is where company knowledge matters. Meridian's response policy depends on the runbook and on how urgent the incident is. Nemotron 3 Ultra without the policy gets 45%. With the policy pasted into every prompt it gets 70%. The fine-tuned 4B and 9B models learned the policy from history and get 80% to 82%.
- Untuned small models are confidently wrong on the action. Untuned Nemotron Nano was wrong half the time when it was more than 91% sure. After fine-tuning, its error rate at that confidence fell to 1.9%.

Recompute any row from the packaged scores, for example:

```bash
python eval/analyze_results.py results/small-models/C-qwen-finetuned.action.rows.jsonl.gz \
  --calibration results/small-models/cal-C-qwen-finetuned.action.rows.jsonl.gz
```

## Inside the agent

We deployed the same Helix agent three times. Nemotron 3 Super runs its loop: triage the incident, check status, then open a ticket or page a team. The only difference between the three is the model behind its `route_incident` tool. The Helix Evals plugin scored each one end to end.

| Model behind the triage step | Incidents | Right runbook | Right action |
|---|---|---|---|
| Nemotron 3 Ultra, no Meridian notes | 50 | 100% | 60% |
| Nemotron 3 Ultra, notes in the prompt | 200 | 100% | 74% |
| **Qwen3.5-4B, fine-tuned** | **200** | **100%** | **81%** |

The agent carried out its triage step's choice in 149 of 150 cases we checked, so agent accuracy follows the model-level numbers. On the first 50 incidents alone the two 200-incident agents tied at 86%. Those 50 happen to be easier than average for Ultra, which is why we ran 150 more. Run 200 if you want to see a difference.

Speed inside the agent: a whole agent run took about the same time either way (12.3 s average with the fine-tuned model, 11.4 s with Ultra and the notes, 12.8 s with Ultra alone). The agent's own loop takes most of that time: about 16,000 prompt tokens per incident on Nemotron 3 Super, the same for every version. The triage step itself wasn't faster inside these runs either. Our tool server fetched an auth token and opened a new connection to a remote Helix cluster on every call, and the records can't separate model time from that overhead. The speed numbers below are for the model measured on its own.

Recompute the agent table:

```bash
python eval/analyze_agent.py results/agent/*.rows.jsonl.gz
```

## Speed and size, measured on the model alone

| Per triage answer | Typical (p50) | Slow (p95) | Prompt tokens | Output tokens |
|---|---|---|---|---|
| Nemotron 3 Ultra, no notes | 669 ms | 2,247 ms | 566 | 2 |
| Nemotron 3 Ultra, notes in the prompt | 670 ms | 2,268 ms | 1,436 | 2 |
| Qwen3.5-4B, fine-tuned | 301 ms | 560 ms | 551 | 1 |
| Strands Decider 2B, on a MacBook | 793 ms | 1,005 ms | 572 | 1 |

The fine-tuned model answers faster, measured on its own: 301 ms against 670 ms typical. Latency is end to end from the same laptop through each model's endpoint: a hosted endpoint for Ultra, the Helix gateway for the fine-tuned model and localhost for Strands. 9 of 50 calls to Ultra with notes hit the hosted endpoint's rate limit and are left out. Raw numbers: [`results/latency.json`](results/latency.json).

Throughput on one GPU at 16 concurrent requests: 29.5 answers a second (Qwen 4B) and 20.5 (Nemotron Nano 9B), measured from a laptop through the Helix gateway, so the GPU's own ceiling is higher ([`results/throughput.json`](results/throughput.json)).

## Cost per answer

Nemotron 3 Ultra at OpenRouter's public prices on 2026-10-07 ($0.50 per million input tokens, $2.20 per million output tokens):

| Per 1,000 answers | Cost |
|---|---|
| Nemotron 3 Ultra, no notes (566 tokens in) | $0.29 |
| Nemotron 3 Ultra, notes in the prompt (1,436 tokens in) | $0.72 |
| Fine-tuned Qwen3.5-4B on one A100 80GB, kept busy at 29.5 answers a second | $0.02 to $0.06 |

The fine-tuned range uses single A100 80GB listings on Brev from $1.66 to $6.03 an hour (October 2026). A dedicated GPU costs the same whether it's busy or idle, so the API is cheaper at low volume. Against Ultra with notes, the GPU pays for itself above roughly 2,300 to 8,400 answers an hour, depending on its price.

## The confidence threshold

Fine-tuned Qwen, action question, with the temperature fitted on the 240-incident calibration split:

| Act alone when confidence is at least | Share the small model handles | Its accuracy on that share |
|---|---|---|
| 0.5 | 99% | 82.2% |
| 0.7 | 86% | 86.4% |
| 0.8 | 74% | 89.5% |
| 0.9 | 45% | 97.9% |
| 0.95 | 35% | 100% |

Sending the low-confidence cases to Nemotron 3 Ultra with the notes lowered overall accuracy at every threshold: 82.2% for the small model alone, 75.7% at a 0.9 threshold with escalation. Ultra is worse than the fine-tuned model on those same hard cases. For this step, uncertain cases should go to a person. Nemotron Nano showed the same pattern.

```bash
python eval/cascade.py --small results/small-models/C-qwen-finetuned.action.rows.jsonl.gz \
  --cal results/small-models/cal-C-qwen-finetuned.action.rows.jsonl.gz \
  --large results/frontier/A-plus-frontier-with-context.action.rows.jsonl.gz
```

## Data quality gates

[`scripts/split_dataset.py`](scripts/split_dataset.py) drops rows in this order. Each dropped row counts once, at the first check it fails.

| Check | Training run (3,600 generated) | Test run (500 generated) |
|---|---|---|
| Judge: text doesn't support the label (score below 4) | 126 | 16 |
| Judge: text gives away the answer (score below 4) | 280 | 37 |
| Keyword check in the Data Designer config | 78 | 15 |
| Pattern sweep at split time | 136 | 16 |
| Unrealistic or empty | 0 | 0 |
| **Kept** | **2,980** (2,503 train, 237 validation, 240 calibration) | **416** (116 from held-out scenario types) |

620 of 3,600 training rows were dropped (17.2%), and 494 of those (80%) gave away the answer. 136 of them got past both the judge and the keyword check and were caught only by the pattern sweep. A row that names the runbook or action makes a fine-tune look better than it is, so check for this in your own data.

Kept training incidents per runbook: RB-01 255, RB-02 293, RB-03 324, RB-04 437, RB-05 203, RB-06 232, RB-07 179, RB-08 261, RB-09 253, RB-10 257, RB-11 205, RB-12 81.

## Time and cost of the whole run

From job records. Our runs used hosted models and a shared Helix cluster, so the dollar figures are what you would pay at public prices: A100 80GB at $1.66 to $6.03 an hour, and OpenRouter list prices for the hosted models.

| Stage | Time | Hardware | Cost |
|---|---|---|---|
| Data generation: 3,600 training and 500 test incidents, run in parallel | 20 min 40 s | Hosted writer and judge | $1.29 |
| Fine-tuning: 6 jobs (2 models × 3 seeds) plus 1 failed job | Qwen about 30 min per job (22 min of training), Nemotron about 41 min per job; 3.21 GPU-hours in all | 1 × A100 80GB per job | $5 to $19 |
| Serving: 2 vLLM deployments, up only for evaluation | 21 min (Qwen) and 30 min (Nemotron) from create to ready; about 3.2 GPU-hours in use | 1 GPU each | $5 to $19 |
| Model-level evaluation | Under a minute per small-model job; about 2.5 min per 416-row Ultra job | Serving GPUs, hosted Ultra | $0.63 for Ultra |
| Agent-level evaluation: 450 agent runs | 14 min per 50-incident job, 68 min in all | Agent on a laptop, hosted Super | about $1 |

**Everything we ran cost about $13 to $40 at public prices**, depending on the A100 price. Most of that is GPU time. Calendar time was longer (22 hours from first data job to last agent evaluation), mostly debugging, overnight gaps and seed reruns. Idle time on deployments isn't counted: a deployment holds its GPU until you delete it, so tear it down when you finish.

Gaps in our records: the deployments don't record the GPU type, so serving is priced as an A100 80GB. Four agent evaluation jobs hit the 1-hour job limit and lost their rows (see [KNOWN-ISSUES.md](KNOWN-ISSUES.md)); their cost is an upper bound.

## Where the numbers live

| Files | What |
|---|---|
| `results/frontier/*.rows.jsonl.gz` | Nemotron 3 Ultra with and without notes, and Strands Decider |
| `results/small-models/*.rows.jsonl.gz` | Untuned and fine-tuned Qwen and Nemotron, all seeds, plus the calibration split (`cal-*`) |
| `results/small-models/summary-*.json`, `seed-summary.json` | Summaries per model and question, and across seeds |
| `results/agent/*.rows.jsonl.gz`, `summary.json` | Whole-agent scores; `.b2` to `.b4` are incidents 51 to 200 |
| `results/latency.json`, `results/throughput.json` | Speed measured on the model alone |
