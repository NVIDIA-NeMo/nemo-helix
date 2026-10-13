# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate customizer E2E test data from ``assets_manifest.json`` in this directory.

All dataset sources, sizes, and output paths are defined in the manifest. Run via
``publish_assets_to_s3.sh`` (off-CI) or directly:

    PYTHONPATH=. uv run --with datasets python e2e/customizer/generate_customizer_data.py

Produces, under the manifest ``generation.output_dir``:
  - prompt_completion/{training,validation}.jsonl   (SQuAD, from squad.outputs)
  - chat_format/{training,validation}.jsonl         (SQuAD, from squad.outputs)
  - dpo/{training,validation,eval}.jsonl            (HelpSteer3, from helpsteer3.outputs)

CI never pulls these from Hugging Face; it syncs the generated files from S3.
"""

import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from e2e.customizer.assets_manifest import generation_config, load, manifest_path

# Span-extraction system prompt, kept identical between training and eval so the
# eval label (final assistant turn) is what the fine-tuned model learns to produce.
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "Answer with the shortest exact span from the context."
)


def convert_to_prompt_completion(example: dict) -> dict:
    prompt = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
    completion = example["answers"]["text"][0]
    return {"prompt": prompt, "completion": completion}


def convert_to_chat(example: dict) -> dict:
    user_content = f"Context: {example['context']}\n\nQuestion: {example['question']}"
    assistant_content = example["answers"]["text"][0]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def convert_to_dpo_helpsteer(example: dict) -> dict:
    return {
        "context": example["context"],
        "response1": example["response1"],
        "response2": example["response2"],
        "overall_preference": example["overall_preference"],
    }


def _as_messages(context: Any) -> list[dict[str, str]]:
    if isinstance(context, str):
        return [{"role": "user", "content": context}]
    return [{"role": msg["role"], "content": msg["content"]} for msg in context]


def convert_dpo_to_eval_chat(example: dict) -> dict | None:
    preference = example.get("overall_preference", 0)
    if preference == 0:
        return None
    preferred = example["response1"] if preference < 0 else example["response2"]
    messages = _as_messages(example["context"])
    messages.append({"role": "assistant", "content": preferred})
    return {"messages": messages}


def write_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data)} samples to {path} ({path.stat().st_size:,} bytes)")


def _select(split: Any, n: int, seed: int) -> list[dict]:
    subset = split.shuffle(seed=seed).select(range(min(n, len(split))))
    return list(subset)


def _generate_squad(output_dir: Path, spec: dict[str, Any], n_train: int, n_val: int, seed: int) -> None:
    hf_dataset = spec["hf_dataset"]
    print(f"\nLoading {hf_dataset} dataset...")
    squad = load_dataset(hf_dataset)
    if not isinstance(squad, DatasetDict):
        raise ValueError(f"{hf_dataset} does not contain expected splits")

    train_examples = _select(squad["train"], n_train, seed)
    val_examples = _select(squad["validation"], n_val, seed)
    print(f"Selected {len(train_examples)} training and {len(val_examples)} validation SQuAD examples")

    outputs = spec["outputs"]
    if "prompt_completion" in outputs:
        print("\nGenerating prompt_completion format...")
        pc_dir = output_dir / "prompt_completion"
        write_jsonl(
            [convert_to_prompt_completion(ex) for ex in train_examples],
            pc_dir / "training.jsonl",
        )
        write_jsonl(
            [convert_to_prompt_completion(ex) for ex in val_examples],
            pc_dir / "validation.jsonl",
        )

    if "chat_format" in outputs:
        print("\nGenerating chat_format...")
        chat_dir = output_dir / "chat_format"
        write_jsonl([convert_to_chat(ex) for ex in train_examples], chat_dir / "training.jsonl")
        write_jsonl([convert_to_chat(ex) for ex in val_examples], chat_dir / "validation.jsonl")


def _generate_helpsteer3(output_dir: Path, spec: dict[str, Any], n_train: int, n_val: int, seed: int) -> None:
    hf_dataset = spec["hf_dataset"]
    hf_config = spec.get("hf_config")
    label = f"{hf_dataset} ({hf_config})" if hf_config else hf_dataset
    print(f"\nLoading {label}...")
    hs3 = load_dataset(hf_dataset, hf_config) if hf_config else load_dataset(hf_dataset)
    if not isinstance(hs3, DatasetDict):
        raise ValueError(f"{hf_dataset} does not contain expected splits")

    dpo_train_examples = _select(hs3["train"], n_train, seed)
    dpo_val_examples = _select(hs3["validation"], n_val, seed)
    print(f"Selected {len(dpo_train_examples)} DPO training and {len(dpo_val_examples)} DPO validation examples")

    print("\nGenerating dpo format...")
    dpo_dir = output_dir / "dpo"
    write_jsonl(
        [convert_to_dpo_helpsteer(ex) for ex in dpo_train_examples],
        dpo_dir / "training.jsonl",
    )
    write_jsonl(
        [convert_to_dpo_helpsteer(ex) for ex in dpo_val_examples],
        dpo_dir / "validation.jsonl",
    )

    eval_rows = [row for ex in dpo_val_examples if (row := convert_dpo_to_eval_chat(ex)) is not None]
    write_jsonl(eval_rows, dpo_dir / "eval.jsonl")


def main() -> None:
    manifest = load()
    cfg = generation_config(manifest)
    n_train, n_val, seed = cfg["training_size"], cfg["validation_size"], cfg["seed"]
    output_dir: Path = cfg["output_dir"]

    print(f"Manifest: {manifest_path()}")
    print(f"Output:   {output_dir}")
    print(f"Sizes:    {n_train} train / {n_val} val (seed={seed})")

    datasets = manifest["datasets"]
    if "squad" in datasets:
        _generate_squad(output_dir, datasets["squad"], n_train, n_val, seed)
    if "helpsteer3" in datasets:
        _generate_helpsteer3(output_dir, datasets["helpsteer3"], n_train, n_val, seed)

    print("\nDone!")


if __name__ == "__main__":
    main()
