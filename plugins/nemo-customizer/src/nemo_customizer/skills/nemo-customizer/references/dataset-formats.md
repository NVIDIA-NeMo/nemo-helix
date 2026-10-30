<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dataset formats

All three backends read JSONL from a platform fileset, but the **row shape and the job-JSON dataset block differ**. Pick the section that matches your plugin (automodel, unsloth, or rl/DPO).

Upload the JSONL files at the **fileset root**, then reference the fileset from the job JSON `dataset` block. Name them `train.jsonl` and `validation.jsonl`, which every SFT and DPO backend reads; the job-JSON contracts differ:

- **Every backend takes one dataset fileset** holding the training file and, optionally, the validation file; there is no separate validation reference. Without a validation file, training holds out a share of the training rows (automodel and unsloth: `schedule.validation_split`, default 0.1; rl: 10%).
- **SFT (automodel, unsloth):** automodel points `dataset.training` at the fileset; unsloth uses `dataset.path`.
- **rl (DPO):** one `dataset` string — see § NeMo-RL (DPO).
- **rl (GRPO):** rows are **NeMo Gym rollout rows** written for the environment; the **nemo-gym-environments** skill covers them.

## Automodel

Automodel detects schema from the **first JSONL line**.

| Schema | JSONL shape | Job JSON |
|--------|-------------|----------|
| **CHAT** (preferred when model has chat template) | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` | (none) |
| **MULTIMODAL** | Same `messages` list. `content` is a string (one text part) or a list of `{type: text, text}` and `{type: image_url, image_url: {url}}`. `url` is a fileset-relative path or a `data:` URL. The assistant label stays one text part. | (none) |
| **SFT** | `{"prompt": "...", "completion": "..."}` | (none) |
| **CUSTOM** | Any two columns, e.g. `{"input": "...", "output": "..."}` | `"prompt_template": "{input} {output}"` on `dataset` |
| **EMBEDDING** | `{"query": "...", "pos_doc": "...", "neg_doc": ["...", "..."]}` | `training.recipe: bi_encoder` or `cross_encoder`. `neg_doc` must be a **non-empty** list; convert-only retrieval Stage 1 emits `[]` and training crashes unless you mine first. |

**Conversion preference:** CHAT if the model has a chat template (`hf-conversion.md` § **Chat-template check**, or model `spec.is_chat` / `spec.chat_template`) → else SFT. Use CUSTOM or EMBEDDING only when the user asks or the task requires it. String `content` stays CHAT. A row with an image saves that file next to the JSONL and sets the user message to an `image_url` part followed by a text part; that part list is MULTIMODAL (`hf-conversion.md` § Image rows). `scripts/import_cord.py` and `scripts/import_finance.py` are examples of that conversion (CORD-v2, and finance OCR, reasoning, and code). Upload the directory that is already this shape; Automodel does not convert the Hugging Face dataset.

MULTIMODAL rows are the fileset to upload. Automodel detects a `messages` row as CHAT, and `ChatDataset` requires string `content`, so an image row fails validation until multimodal training support lands.

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "images/receipt.png"}},
    {"type": "text", "text": "Describe this image."}
  ]},
  {"role": "assistant", "content": "<s_menu><s_nm>"}
]}
```

**Why CHAT for chat models:** SFT (and CUSTOM) rows train as plain text, `prompt + completion`, with no chat template and no end-of-sequence token. A chat model trained that way barely changes through `/v1/chat/completions`, and `/v1/completions` output runs past the answer. Put the exact user prompt you will send at inference in the `user` turn.

Retrieval Stage 1 already emits Automodel EMBEDDING `training.jsonl`. Do not convert those rows to CHAT `messages`.

For **CUSTOM**, placeholders in `prompt_template` must match column names exactly (two placeholders).

## Unsloth

Unsloth has no schema auto-detection — the row shape is controlled by two `dataset` fields in the job JSON. The training driver hands rows to `trl.SFTTrainer`, which only reads one column (`text_field`) per row.

| Mode | `dataset.apply_chat_template` | Required JSONL shape | What the trainer sees |
|------|------------------------------|----------------------|----------------------|
| **Messages (preferred)** | `true` | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` (same as automodel CHAT) | Each row's `messages` is rendered through `tokenizer.apply_chat_template(...)` at training time; the rendered string is written into `text_field` (default `"text"`). |
| **Pre-rendered text** | `false` (default) | `{"text": "<one fully-formed training string>"}` | The string in `text_field` is fed to SFTTrainer verbatim. |

Job JSON snippets:

```json
"dataset": { "path": "default/<dataset-fileset>", "apply_chat_template": true }
```

```json
"dataset": { "path": "default/<dataset-fileset>", "text_field": "text", "apply_chat_template": false }
```

Optional fields on the unsloth `dataset` block:

| Field | Default | Notes |
|-------|---------|-------|
| `text_field` | `"text"` | Column the trainer reads. In messages mode it's the column the rendered string is **written to** before training. |
| `apply_chat_template` | `false` | Set `true` only when each row has a `messages` array. |
| `packing` | `false` | `trl.SFTTrainer` packing — concatenates short rows up to `max_seq_length` for throughput. Needs short, compatible rows; safe to leave off. |

**Conversion guidance:**

- If the model has a chat template (`hf-conversion.md` § **Chat-template check**), use the same `to_chat` converter from `references/hf-conversion.md` and set `apply_chat_template: true`. This is the recommended path for instruction-tuned models. Unsloth stays on string `content` or pre-rendered `text`. Do not emit image parts on this path.
- If the model has **no** chat template, render each example to a single training string yourself (e.g. `f"{prompt}\n{completion}"`) and emit `{"text": "..."}` rows. Then set `apply_chat_template: false` and keep `text_field: "text"`.
- The automodel SFT format `{"prompt": "...", "completion": "..."}` is **not** directly consumable by unsloth — unsloth has no built-in `prompt`/`completion` concatenation. Convert to either messages or pre-rendered text before upload.

EMBEDDING and CUSTOM (automodel-only schemas) are not supported by unsloth today.

## Evaluation data

| Training JSONL | Eval rows | How it is scored |
|----------------|-----------|------------------|
| `messages` (single- or multi-turn) | A frozen `test.jsonl` in the same shape, in its own fileset | `post-training-eval.md` |
| EMBEDDING (`query` / `pos_doc` / `neg_doc`) | Frozen Stage 1 `eval_beir` fileset (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) | `nemo evals retrieve-eval`: nDCG@k / Recall@k in `eval_results.json` |

## NeMo-RL (DPO) — preference data

DPO trains on **preference pairs**, not prompt→completion examples. The `rl` backend takes a **single** dataset fileset, referenced by one `dataset` string, holding a training file and, optionally, a validation file, named like `train*.jsonl` and `val*.jsonl` or `dev*.jsonl` (such as `training.jsonl` and `validation.jsonl`), or placed directly in `train/` (or `training/`) and `val/` (or `validation/`, `dev/`). Without a validation file, 10% of the training rows are held out for validation.

The dataset-preparation step **auto-detects the row schema from the first line** and selects the matching NeMo-RL loader. **Three preference formats are supported** (platform schemas `BinaryPreferenceDatasetItemSchema` / `HelpSteer3DatasetItemSchema` / `Tulu3PreferenceDatasetItemSchema`):

### Binary preference (`BinaryPreferenceDataset`)

Simple `prompt` / `chosen` / `rejected` — the `prompt` may be a plain string **or** a list of chat messages:

```json
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "I'm not sure."}
```

| Key | Meaning |
|-----|---------|
| `prompt` | The input/context shown to the model (string or list of chat messages). |
| `chosen` | The preferred (higher-reward) response. |
| `rejected` | The dispreferred response. |

### HelpSteer3 (`HelpSteer3`)

A conversation `context` (string or chat messages), two candidate responses, and a signed `overall_preference` in **-3..3** — **negative** means `response1` is preferred, **positive** means `response2`, **0** is a tie. This is the **raw** schema of `nvidia/HelpSteer3` (the `preference` subset), so no conversion is needed:

```json
{"context": [{"role": "user", "content": "Explain how to use git rebase"}], "response1": "...", "response2": "...", "overall_preference": -2}
```

### Tulu3 preference (`Tulu3Preference`)

Full chat conversations for both branches — `chosen` and `rejected` are each a **list of messages** ending with the assistant turn:

```json
{"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "preferred"}], "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "dispreferred"}]}
```

Whichever format you upload, the job JSON dataset block is just the single fileset ref:

```json
"dataset": "default/<preference-fileset>"
```

**Notes**
- Upload both files to the **same** fileset (`--remote-path training.jsonl` and `--remote-path validation.jsonl`).
- `prompt` may be a plain string; the model's chat template is applied at training time (override with `training.chat_template` only when needed).
- DPO is **full-weight** — there is no LoRA/adapter dataset variant. The output is a full model checkpoint.

