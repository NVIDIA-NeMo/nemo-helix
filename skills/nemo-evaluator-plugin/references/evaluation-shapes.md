# Evaluation Shapes

## Difference summary
Choose the evaluation shape from what produces the scored output. Metrics are
shared scorers; the input and evidence differ.

| Question | Dataset-driven | Task-driven | Retrieval-driven |
| --- | --- | --- | --- |
| What is the input? | Fixed dataset rows | Tasks with intent, inputs, and metrics | A BEIR corpus, queries, and qrels |
| What is scored? | One output per row | One or more trials per task | Ranked corpus IDs for every query |
| Which metrics apply? | The same metric set applies to every row | Each task can define its own metrics | Per-query nDCG, recall, precision, and MAP |
| What evidence is available? | Row fields, row scores, and aggregates | Final output, trajectory, tool calls, other trial evidence, per-task rewards, and summary | Per-query rankings and Range-averaged scores |
| Platform job | `evaluate submit` | `agent-evaluate submit` | `retrieve-eval submit` |

## Dataset-driven evaluation

Use dataset-driven evaluation for a fixed set of examples where the same
scoring rules apply to every row. Typical uses include model quality checks
and labeled-set benchmarks.

Each row contains the fields consumed by the metric, for example:

```python
{"question": "Capital of France?", "expected": "Paris", "output": "Paris"}
```

Then choose the scorer, validate its field mapping with the standalone SDK,
and submit only after the pass/fail smoke case behaves as expected.

## Task-driven evaluation

Use task-driven evaluation when the system performs work and the process can
matter as much as the final answer. A model, agent, or runner produces a trial
containing the final output and available execution evidence. Tasks can carry
different metrics, so one taskset can grade heterogeneous work.

Choose this shape for agent behavior, tool use, multi-step work, runner-based
benchmarks, or rescoring precomputed trials.

Define tasks, trials or a target, concurrency, and result handling before
submitting the task-driven job. Read the Agent Evaluation reference for details.

## Retrieval-driven evaluation

Use retrieval-driven evaluation when an embedding model (and optional reranker) ranks a corpus
for a set of queries. The dataset must use the BEIR test layout:
`corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. Pass it as `dataset=` with
`target=Retrieval(embeddings=..., reranker=...)`. Long documents are title-aware truncated to
65535 characters; they are not split into multi-vector passages.

This is not row-based RAG answer scoring. Do not convert qrels into artificial
answer rows or use an LLM judge for deterministic retrieval quality.

Submit the fileset and embedding target as references:

```bash
uv run nemo evaluator retrieve-eval submit --spec \
  '{"dataset":"default/eval-beir","target":"default/embed-nim","k":[1,5,10,100]}'
```

The `eval_results.json` artifact contains `ndcg_cut_k`, `recall_k`, `P_k`, and `map_cut_k`.
An optional `baseline` model reference adds relative `ndcg_cut_10` and `recall_10` to the job
output.

## Previous / Next / artifacts

| Direction | Skill or job | Artifact |
|---|---|---|
| Previous | Data Designer `retrieval-prepare` (frozen) and customizer output model entity or embed/ranking NIM | `eval_beir` fileset; tuned + base model refs |
| This stage | `nemo evaluator retrieve-eval submit` | `eval_results.json` |
| Next | `nemo-retrieval-recipes` deploy step, or stop after metrics | Optional NIM deploy (embed and rerank ONNX from Automodel) |

End-to-end domain recipes live in `nemo-retrieval-recipes`. Do not use an LLM judge for qrels.
