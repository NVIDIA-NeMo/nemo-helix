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

## Image rows

When a source row has an image, write the image next to the JSONL and emit a part list. A string `content` is still one text part. Text-only rows in this same file use `to_chat`, so every line is a `messages` row. An image file needs a chat model. The assistant label stays one text part (a string).

Automodel does not convert the Hugging Face dataset. Upload one directory that is already this shape. For CORD-v2, run the helper and upload the directory it writes. That fileset holds `train/` and `validation/`:

```bash
uv run --no-project --with datasets --with pillow \
  plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/import_cord.py \
  --output /tmp/cord-v2
nemo files filesets create cord-v2 --workspace default --purpose dataset
nemo files upload /tmp/cord-v2 cord-v2 --workspace default
```

`cord-v2` contains `train/train.jsonl` and `validation/validation.jsonl`, each with `images/` beside the JSONL. Image urls are fileset-relative (`train/images/00000.png`, `validation/images/00000.png`). Automodel selects training files from `train/` and validation files from `validation/`. Job JSON names this one fileset. The Evaluator scores `validation/validation.jsonl` in it: that official validation file is CORD's frozen split, and the script does not write `test.jsonl`.

`nemo customization automodel --upload-dataset /tmp/cord-v2` uploads that same directory.

Finance (TAT-DQA and BizBench) uses `scripts/import_finance.py`. Upload `visual_ocr/`, `reasoning/`, and `code_generation/` separately. OCR rows have a relative `image_url`. Reasoning and code rows are text. Each directory's `split_manifest.json` is the leakage audit.

Unsloth stays on string `content` or pre-rendered `text`. Do not give it image parts.

```python
def example_images(ex):
    image = ex.get("image")
    if image is not None:
        return [image]
    return list(ex.get("images") or [])

def to_multimodal_chat(image_urls, user, assistant):
    parts = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    parts.append({"type": "text", "text": user})
    return {
        "messages": [
            {"role": "user", "content": parts},
            {"role": "assistant", "content": assistant},
        ]
    }

def image_row(ex, split, index, images_dir):
    found = example_images(ex)
    chat = to_chat(ex)  # same user text and assistant label as a text row
    if not found:
        return chat
    user, assistant = chat["messages"][0]["content"], chat["messages"][1]["content"]
    urls = []
    for image_index, image in enumerate(found):
        name = f"{split}_{index}_{image_index}.png"
        image.save(images_dir / name)
        urls.append(f"images/{name}")
    return to_multimodal_chat(urls, user, assistant)

# Freeze a test split from train, the same rule as the text script above.
train_examples = list(ds["train"])
if len(train_examples) < 2:
    raise SystemExit("Need at least two rows: one to train on and one to test.")
random.Random(42).shuffle(train_examples)
n_test = min(500, max(1, len(train_examples) // 10))
held_out, train_examples = train_examples[:n_test], train_examples[n_test:]

train_images = Path("/tmp/train-data/images")
test_images = Path("/tmp/test-data/images")
train_images.mkdir(parents=True, exist_ok=True)
test_images.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, examples, split, images_dir):
    with path.open("w") as f:
        for index, ex in enumerate(examples):
            f.write(json.dumps(image_row(ex, split, index, images_dir)) + "\n")

write_jsonl(Path("/tmp/train-data/train.jsonl"), train_examples, "train", train_images)
write_jsonl(Path("/tmp/test-data/test.jsonl"), held_out, "test", test_images)
if "validation" in ds:
    write_jsonl(Path("/tmp/train-data/validation.jsonl"), ds["validation"], "validation", train_images)
```

Each `image_url.url` is that relative path, or a `data:` URL when the bytes stay inline. A row with several images gets one `image_url` part per image, then the text part. Upload `/tmp/train-data/` to the dataset fileset and `/tmp/test-data/` to `<DATASET_NAME>-test`. Each directory holds its JSONL and `images/`. Validate with `nemo files list <DATASET_NAME> --workspace default`.

A Gym row that needs an image uses Responses API parts on `responses_create_params.input[].content`: `{type: input_image, image_url: "<string>"}` followed by `{type: input_text, text}`. A string `content` stays one text part. The automodel fileset stays `{type: image_url, image_url: {url}}`. See the **nemo-gym-environments** skill, `references/rollout-rows.md`.

## Mapping to job JSON

Only the **chat-template / `messages` (`to_chat`) output is backend-agnostic** — that JSONL feeds both SFT backends (automodel + unsloth). The `to_sft` (`prompt` / `completion`) shape is **automodel-only**; unsloth needs a `to_text` rendering instead (see below). Either way, the **dataset block in job JSON is shaped per backend**. (rl/DPO uses preference data, not this SFT output — see `dataset-formats.md` § NeMo-RL.)

| Backend | Row format used | Dataset block in job JSON |
|---------|----------------|---------------------------|
| automodel (CHAT) | `to_chat` output (`messages`, string `content`) | `{ "training": "default/<DATASET_NAME>" }` — schema auto-detected from row 1 |
| automodel (MULTIMODAL) | `to_multimodal_chat` output (part list + sibling images) | same fileset ref. Text-only rows in the same file stay string `content`. |
| automodel (SFT)  | `to_sft` output (`prompt` / `completion`) | same as above (no `prompt_template`) |
| **unsloth (preferred)** | `to_chat` output (`messages`) | `{ "path": "default/<DATASET_NAME>", "apply_chat_template": true }` |
| unsloth (no chat template) | **Custom `to_text` rendering**: emit `{"text": "<prompt>\n<completion>"}` rows (not the `to_sft` output directly) | `{ "path": "default/<DATASET_NAME>", "text_field": "text" }` |

**Note:** Unsloth does **not** read the automodel SFT shape `{"prompt": ..., "completion": ...}`. If `has_chat` is False *and* the user picked unsloth, swap `to_sft` for a `to_text` that renders one `text` column. Sketch:

```python
def to_text(row):
    user, assistant = row["messages"][0]["content"], row["messages"][1]["content"]
    return {"text": f"{user}\n{assistant}"}
```

For the chat path (`has_chat` True), the `to_chat` JSONL works unchanged across both SFT backends (automodel + unsloth) — only the job-JSON dataset block differs.
