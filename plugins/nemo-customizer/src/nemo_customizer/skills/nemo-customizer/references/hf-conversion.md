<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hugging Face dataset conversion

Run from **nemo-helix** git root: `uv run python …` (plugin brings `datasets` + `transformers`).

Do **not** ask the user for local paths when they gave an HF dataset id — convert and upload in the same session.

## Chat-template check

```python
from transformers import AutoTokenizer
has_chat = bool(getattr(AutoTokenizer.from_pretrained("<hf-repo>", trust_remote_code=True), "chat_template", None))
```

If the model entity already exists: `nemo models get <entity> --workspace default` → use `spec.is_chat` or `spec.chat_template` instead of re-downloading tokenizer weights.

## Conversion script (adapt `to_chat` per dataset)

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from pathlib import Path

HF_REPO = "<hf-repo>"
HF_DATASET = "<hf-dataset>"   # e.g. tau/commonsense_qa
DATASET_NAME = HF_DATASET.split("/")[-1].lower()   # fileset name, e.g. commonsense_qa

has_chat = bool(getattr(AutoTokenizer.from_pretrained(HF_REPO, trust_remote_code=True), "chat_template", None))

def to_chat(ex):
    # MCQA example (tau/commonsense_qa):
    labels, texts = ex["choices"]["label"], ex["choices"]["text"]
    choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    user = f"Question: {ex['question']}\nChoices:\n{choices}\nAnswer:"
    assistant = texts[labels.index(ex["answerKey"])]
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}

def to_sft(ex):
    row = to_chat(ex)
    return {"prompt": row["messages"][0]["content"], "completion": row["messages"][1]["content"]}

convert = to_chat if has_chat else to_sft

ds = load_dataset(HF_DATASET)
out = Path("/tmp/train-data")
out.mkdir(exist_ok=True)
for split in ("train", "validation"):
    if split in ds:
        with (out / f"{split}.jsonl").open("w") as f:
            for ex in ds[split]:
                f.write(json.dumps(convert(ex)) + "\n")
```

Then upload (see main skill). Validate with `nemo files list <DATASET_NAME> --workspace default`.

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
def to_text(ex):
    row = to_chat(ex)
    user, assistant = row["messages"][0]["content"], row["messages"][1]["content"]
    return {"text": f"{user}\n{assistant}"}
```

For the chat path (`has_chat` True), the `to_chat` JSONL works unchanged across both SFT backends (automodel + unsloth) — only the job-JSON dataset block differs.
