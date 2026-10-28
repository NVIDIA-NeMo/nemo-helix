<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hugging Face dataset conversion

Run from **nemo-helix** git root: `uv run python …` (the workspace environment has `datasets` and `huggingface_hub`).

## Chat-template check

Read the template from the tokenizer's config files. Unlike loading the tokenizer, this downloads one small file and runs no code from the repo.

```python
import json
from huggingface_hub import file_exists, hf_hub_download


def has_chat_template(repo: str) -> bool:
    """True when the repo's tokenizer ships a chat template."""
    if file_exists(repo, "tokenizer_config.json"):
        with open(hf_hub_download(repo, "tokenizer_config.json"), encoding="utf-8") as f:
            if json.load(f).get("chat_template"):
                return True
    # Newer tokenizers keep the template in its own file.
    return file_exists(repo, "chat_template.jinja")
```

A gated repo needs `HF_TOKEN` in the shell that runs this.

If the model entity already exists: `nemo models get <entity> --workspace default` → use `spec.is_chat` or `spec.chat_template` instead of downloading anything.

## Conversion script (adapt `to_chat` per dataset)

```python
from datasets import load_dataset
import json
import random
from pathlib import Path

HF_REPO = "<hf-repo>"
HF_DATASET = "<hf-dataset>"   # e.g. tau/commonsense_qa
DATASET_NAME = HF_DATASET.split("/")[-1].lower()   # fileset name, e.g. commonsense_qa

has_chat = has_chat_template(HF_REPO)   # from the check above

def to_chat(ex):
    # MCQA example (tau/commonsense_qa):
    labels, texts = ex["choices"]["label"], ex["choices"]["text"]
    choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    user = f"Question: {ex['question']}\nChoices:\n{choices}\nAnswer:"
    assistant = texts[labels.index(ex["answerKey"])]
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}

def to_sft(row):
    return {"prompt": row["messages"][0]["content"], "completion": row["messages"][1]["content"]}

train_format = (lambda row: row) if has_chat else to_sft

ds = load_dataset(HF_DATASET)

# Freeze a test split: about 10% of the training rows, at most 500, never trained on.
rows = [to_chat(ex) for ex in ds["train"]]
if len(rows) < 2:
    raise SystemExit("Need at least two rows: one to train on and one to test.")
random.Random(42).shuffle(rows)
n_test = min(500, max(1, len(rows) // 10))
splits = {
    Path("/tmp/train-data/train.jsonl"): [train_format(row) for row in rows[n_test:]],
    # CHAT rows whatever the training shape: the eval reads item.messages. Own directory and fileset.
    Path("/tmp/test-data/test.jsonl"): rows[:n_test],
}
if "validation" in ds:
    splits[Path("/tmp/train-data/validation.jsonl")] = [train_format(to_chat(ex)) for ex in ds["validation"]]
for path, rows in splits.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
```

Then upload `/tmp/train-data/` to the dataset fileset and `/tmp/test-data/test.jsonl` to `<DATASET_NAME>-test` (`SKILL.md` § **Workflow** step 2). Validate with `nemo files list <DATASET_NAME> --workspace default`.

## Mapping to job JSON

Only the **chat-template / `messages` (`to_chat`) output is backend-agnostic** — that JSONL feeds both SFT backends (automodel + unsloth). The `to_sft` (`prompt` / `completion`) shape is **automodel-only**; unsloth needs a `to_text` rendering instead (see below). Either way, the **dataset block in job JSON is shaped per backend**. (rl/DPO uses preference data, not this SFT output — see `dataset-formats.md` § NeMo-RL.)

| Backend | Row format used | Dataset block in job JSON |
|---------|----------------|---------------------------|
| automodel (CHAT) | `to_chat` output (`messages`) | `{ "training": "default/<DATASET_NAME>", "validation": "default/<DATASET_NAME>" }` — schema auto-detected from row 1 |
| automodel (SFT)  | `to_sft` output (`prompt` / `completion`) | same as above (no `prompt_template`) |
| **unsloth (preferred)** | `to_chat` output (`messages`) | `{ "path": "default/<DATASET_NAME>", "apply_chat_template": true }` (+ `validation_path` if present) |
| unsloth (no chat template) | **Custom `to_text` rendering**: emit `{"text": "<prompt>\n<completion>"}` rows (not the `to_sft` output directly) | `{ "path": "default/<DATASET_NAME>", "text_field": "text" }` |

**Note:** Unsloth does **not** read the automodel SFT shape `{"prompt": ..., "completion": ...}`. If `has_chat` is False *and* the user picked unsloth, swap `to_sft` for a `to_text` that renders one `text` column. Sketch:

```python
def to_text(row):
    user, assistant = row["messages"][0]["content"], row["messages"][1]["content"]
    return {"text": f"{user}\n{assistant}"}
```

For the chat path (`has_chat` True), the `to_chat` JSONL works unchanged across both SFT backends (automodel + unsloth) — only the job-JSON dataset block differs.
