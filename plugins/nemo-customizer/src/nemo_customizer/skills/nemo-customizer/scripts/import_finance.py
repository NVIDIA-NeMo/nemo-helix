#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the three finance capability filesets as OpenAI multimodal JSONL.

Run this before upload. Automodel does not convert TAT-DQA or BizBench.
Each directory you upload is already the training row. OCR rows carry a
relative ``image_url``. Reasoning and code rows are text. Train and validation
assistant text is the thinking-trace target. Test assistant text is the answer
only. ``split_manifest.json`` in each directory is the leakage audit.

Prep matches the Nemotron finance cookbook: seed 42, validation ratio 0.1,
150 DPI, TAT-DQA revision ``bc46208``, BizBench revision ``0a793f2``, tasks
CodeFinQA and CodeTAT-QA.

    uv run --no-project --with datasets --with pillow --with pymupdf --with huggingface_hub \
      plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/import_finance.py \
      --output /tmp/finance

Then upload each capability directory (new fileset names, no ``--exist-ok``):

    nemo files filesets create finance-ocr --workspace default --purpose dataset
    nemo files upload /tmp/finance/visual_ocr finance-ocr --workspace default
    nemo files filesets create finance-reasoning --workspace default --purpose dataset
    nemo files upload /tmp/finance/reasoning finance-reasoning --workspace default
    nemo files filesets create finance-code --workspace default --purpose dataset
    nemo files upload /tmp/finance/code_generation finance-code --workspace default

``nemo customization automodel --upload-dataset`` uploads one of those directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

SPLITS = ("train", "validation", "test")
TAT_REPO = "next-tat/TAT-DQA"
BIZ_REPO = "kensho/bizbench"
TAT_REVISION = "bc46208b22f0bfe47152c98fd0938d238ce868c2"
BIZ_REVISION = "0a793f2f886156902c72b4a22cab82bb9dceaecf"
BIZ_TASKS = ("CodeFinQA", "CodeTAT-QA")
TARGET_TEMPLATE = "<think>\n{trace}\n</think>\n{answer}"
JSONL_NAMES = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
CAPABILITIES = {
    "visual_ocr": {
        "tasks": ("TAT-DQA/multi-span", "TAT-DQA/span"),
        "task_family": "visual_ocr",
        "input_mode": "image",
    },
    "reasoning": {
        "tasks": ("TAT-DQA/arithmetic", "TAT-DQA/count"),
        "task_family": "financial_reasoning",
        "input_mode": "text",
    },
    "code_generation": {
        "tasks": ("CodeFinQA", "CodeTAT-QA"),
        "task_family": "program_synthesis",
        "input_mode": "text",
    },
}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def stable_hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8], "big")


def _answer_text(answer: object) -> str:
    if isinstance(answer, list):
        return " | ".join(str(value).strip() for value in answer)
    return str(answer).strip()


def _answer_with_scale(answer: object, scale: object) -> str:
    value = _answer_text(answer)
    unit = str(scale or "").strip().casefold()
    if not unit:
        return value
    if unit == "percent":
        return value if value.endswith("%") else f"{value}%"
    return value if normalize_text(value).endswith(unit) else f"{value} {unit}"


def _tat_trace(question: dict, answer: str) -> str:
    derivation = str(question.get("derivation") or "").strip()
    facts = [str(value).strip() for value in (question.get("facts") or []) if str(value).strip()]
    scale = str(question.get("scale") or "").strip() or "none"
    parts = []
    if facts:
        parts.append("Evidence values: " + " ; ".join(facts))
    if derivation:
        parts.append("Calculation: " + derivation)
    if not parts:
        parts.append("Read the document and extract the requested evidence.")
    parts.append(f"Answer scale: {scale}")
    parts.append(f"Final answer: {answer}")
    return "\n".join(parts)


def build_user_prompt(task: str, task_family: str, question: str, context: str = "") -> str:
    """Build the same task-tagged user text for training and evaluation."""
    instruction = (
        "Return a grounded reasoning trace inside <think>...</think>, followed by only the final answer. "
        "Preserve the requested financial unit or scale."
    )
    if task_family in {"visual_document_reasoning", "visual_ocr"}:
        return (
            f"Task: {task}\n{instruction}\n"
            "Read the financial document image and extract the requested evidence exactly.\n"
            f"Question: {question}"
        )
    if task_family == "financial_reasoning":
        return (
            f"Task: {task}\n{instruction}\n"
            "Use the supplied financial-document evidence to calculate or count the requested result.\n\n"
            f"Evidence:\n{context}\n\nQuestion:\n{question}"
        )
    return (
        f"Task: {task}\n{instruction}\n"
        "Write an executable Python calculation in the reasoning trace.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )


def finance_row(row: dict[str, Any]) -> dict[str, Any]:
    """One chat row. Text-only content is a string. An image is a part list."""
    prompt = build_user_prompt(row["task"], row["task_family"], row["question"], row.get("context") or "")
    image_url = str(row.get("image_path") or "")
    if image_url:
        user: str | list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]
    else:
        user = prompt
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["target"]},
        ]
    }


def project_row(row: dict[str, Any], capability: str, split: str) -> dict[str, Any]:
    """Apply the capability view. Test rows keep the answer and drop the trace."""
    spec = CAPABILITIES[capability]
    projected = dict(row)
    projected["capability"] = capability
    projected["task_family"] = spec["task_family"]
    if spec["input_mode"] == "text":
        projected["image_path"] = ""
    if split == "test":
        projected["reasoning"] = ""
        projected["program"] = ""
        projected["target"] = str(row["answer"]).strip()
    return projected


def _render_pdf_page(pdf_bytes: bytes, page_index: int, dpi: int) -> Image.Image:
    try:
        import fitz  # ty: ignore[unresolved-import]  # optional; pdftoppm is the fallback

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_index = min(max(page_index, 0), document.page_count - 1)
            pixmap = document.load_page(page_index).get_pixmap(dpi=dpi, alpha=False)
            return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        finally:
            document.close()
    except ImportError:
        executable = shutil.which("pdftoppm")
        if not executable:
            raise RuntimeError("High-resolution TAT-DQA rendering requires PyMuPDF or pdftoppm")
        with tempfile.TemporaryDirectory(prefix="tatdqa-render-") as temporary:
            pdf_path = Path(temporary) / "document.pdf"
            output_prefix = Path(temporary) / "page"
            pdf_path.write_bytes(pdf_bytes)
            subprocess.run(
                [
                    executable,
                    "-f",
                    str(page_index + 1),
                    "-singlefile",
                    "-r",
                    str(dpi),
                    "-png",
                    str(pdf_path),
                    str(output_prefix),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with Image.open(output_prefix.with_suffix(".png")) as rendered:
                return rendered.convert("RGB")


def _ocr_page(payload: dict, page_index: int) -> dict:
    pages = payload.get("pages") or []
    if not pages:
        return {"bbox": [0, 0, 1, 1], "blocks": []}
    return pages[min(max(page_index, 0), len(pages) - 1)]


def _evidence_crop(page: Image.Image, ocr_page: dict, question: dict) -> tuple[Image.Image, str]:
    blocks = ocr_page.get("blocks") or []
    by_uuid = {str(block.get("uuid")): block for block in blocks}
    mapped = []
    for item in question.get("block_mapping") or []:
        for block_id in item:
            if str(block_id) in by_uuid:
                mapped.append(by_uuid[str(block_id)])
    page_bbox = ocr_page.get("bbox") or [0, 0, page.width, page.height]
    source_height = max(float(page_bbox[3]) - float(page_bbox[1]), 1.0)
    if not mapped:
        evidence = "\n".join(str(block.get("text") or "").strip() for block in blocks if block.get("text"))
        return page.copy(), evidence

    top = max(0.0, min(float(block["bbox"][1]) for block in mapped) - 130.0)
    bottom = min(source_height, max(float(block["bbox"][3]) for block in mapped) + 100.0)
    if bottom - top < 420.0:
        extra = (420.0 - (bottom - top)) / 2.0
        top = max(0.0, top - extra)
        bottom = min(source_height, bottom + extra)
    selected = [
        block
        for block in blocks
        if block.get("bbox") and float(block["bbox"][3]) >= top and float(block["bbox"][1]) <= bottom
    ]
    evidence = "\n".join(str(block.get("text") or "").strip() for block in selected if block.get("text"))
    y0 = int(round(top / source_height * page.height))
    y1 = int(round(bottom / source_height * page.height))
    return page.crop((0, max(0, y0), page.width, min(page.height, max(y0 + 1, y1)))), evidence


def load_tat_records(source_dir: Path, media_dir: Path, *, render_dpi: int) -> list[dict]:
    records: list[dict] = []
    json_names = {
        "train": "tatdqa_dataset_train.json",
        "validation": "tatdqa_dataset_dev.json",
        "test": "tatdqa_dataset_test_gold.json",
    }
    archive_names = {"train": "train", "validation": "dev", "test": "test"}
    for split, json_name in json_names.items():
        path = source_dir / json_name
        if not path.is_file():
            raise FileNotFoundError(path)
        docs = json.loads(path.read_text())
        print(f"Preparing TAT-DQA {split}: {len(docs)} documents", flush=True)
        archive_split = archive_names[split]
        archive = source_dir / f"tatdqa_docs_{archive_split}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        destination = media_dir / archive_split
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for document_index, item in enumerate(docs, 1):
                if document_index == 1 or document_index % 100 == 0:
                    print(
                        f"TAT-DQA {split}: document {document_index}/{len(docs)}; "
                        f"{len(records)} records prepared across splits",
                        flush=True,
                    )
                doc = item["doc"]
                uid = str(doc["uid"])
                page_number = int(doc["page"])
                page_index = page_number - 1
                pdf_member = f"{archive_split}/{uid}.pdf"
                ocr_member = f"{archive_split}/{uid}.json"
                missing = {pdf_member, ocr_member} - names
                if missing:
                    raise FileNotFoundError(f"Missing TAT-DQA source assets: {sorted(missing)}")
                page_image = _render_pdf_page(zf.read(pdf_member), page_index, render_dpi)
                ocr = _ocr_page(json.loads(zf.read(ocr_member)), page_index)
                source = str(doc.get("source") or uid)
                for question in item["questions"]:
                    crop, ocr_context = _evidence_crop(page_image, ocr, question)
                    image_name = f"{uid}_{page_number}_{question['uid']}.png"
                    image_path = destination / image_name
                    if not image_path.is_file() or not image_path.stat().st_size:
                        crop.save(image_path, format="PNG", optimize=True)
                    crop.close()
                    if "answer" not in question:
                        raise ValueError(f"Gold answer missing for TAT-DQA {split} question {question.get('uid')}")
                    answer = _answer_with_scale(question["answer"], question.get("scale"))
                    if not answer or not str(question.get("question") or "").strip():
                        continue
                    trace = _tat_trace(question, answer)
                    records.append(
                        {
                            "sample_id": f"tatdqa:{question['uid']}",
                            "source_dataset": "TAT-DQA",
                            "source_split": split,
                            "task": f"TAT-DQA/{question.get('answer_type') or 'unknown'}",
                            "task_family": "visual_document_reasoning",
                            "lineage_key": f"tatdqa-report:{normalize_text(source)}",
                            "question_key": normalize_text(question["question"]),
                            "image_path": f"{archive_split}/{image_name}",
                            "context": ocr_context,
                            "question": str(question["question"]).strip(),
                            "reasoning": trace,
                            "answer": answer,
                            "scale": str(question.get("scale") or "").strip(),
                            "program": "",
                            "target": TARGET_TEMPLATE.format(trace=trace, answer=answer),
                            "assigned_split": split,
                        }
                    )
                page_image.close()
    return records


def _biz_lineage(context: str, question: str) -> str:
    material = normalize_text(context) or normalize_text(question)
    return "bizbench-context:" + hashlib.sha256(material.encode()).hexdigest()


def _bizbench_parquet(source_dir: Path, split: str) -> Path:
    normalized = source_dir / f"bizbench_{split}.parquet"
    if normalized.is_file():
        return normalized
    matches = sorted((source_dir / "data").glob(f"{split}-*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one BizBench {split} parquet under {source_dir / 'data'} or {normalized}; found {matches}"
        )
    return matches[0]


def load_biz_records(source_dir: Path) -> list[dict]:
    from datasets import load_dataset

    records: list[dict] = []
    for split in ("train", "test"):
        dataset = load_dataset("parquet", data_files=str(_bizbench_parquet(source_dir, split)), split="train")
        for index, row in enumerate(dataset):
            task = str(row.get("task") or "")
            if task not in BIZ_TASKS:
                continue
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            program = str(row.get("program") or "").strip()
            context_value = row.get("context")
            context = "" if context_value is None else str(context_value).strip()
            if not question or not answer or not program:
                continue
            prompt_context = context if context else "No external document context is required."
            records.append(
                {
                    "sample_id": f"bizbench:{split}:{task}:{index}",
                    "source_dataset": "BizBench",
                    "source_split": split,
                    "task": task,
                    "task_family": "program_synthesis",
                    "lineage_key": _biz_lineage(context, question),
                    "question_key": normalize_text(question),
                    "image_path": "",
                    "context": prompt_context,
                    "question": question,
                    "reasoning": program,
                    "answer": answer,
                    "scale": "",
                    "program": program,
                    "target": TARGET_TEMPLATE.format(trace=program, answer=answer),
                    "assigned_split": split,
                }
            )
    return records


def _keys(records: Iterable[dict]) -> tuple[set[str], set[str]]:
    rows = list(records)
    return ({row["question_key"] for row in rows}, {row["lineage_key"] for row in rows})


def make_splits(tat: list[dict], biz: list[dict], *, seed: int, validation_ratio: float) -> tuple[dict, dict]:
    """Regroup TAT-DQA by report and preserve BizBench's official test split."""
    output = {name: [] for name in SPLITS}
    excluded = Counter()

    fixed_test = [row for row in biz if row["assigned_split"] == "test"]
    output["test"].extend(fixed_test)

    tat_reports: dict[str, list[dict]] = defaultdict(list)
    for row in tat:
        tat_reports[row["lineage_key"]].append(row)
    for key, rows in tat_reports.items():
        score = stable_hash(key, seed) / 2**64
        if score < 0.8:
            split = "train"
        elif score < 0.9:
            split = "validation"
        else:
            split = "test"
        output[split].extend(rows)

    heldout_questions, heldout_lineages = _keys(output["validation"] + output["test"])
    eligible: list[dict] = []
    for row in biz:
        if row["assigned_split"] != "train":
            continue
        if row["question_key"] in heldout_questions or row["lineage_key"] in heldout_lineages:
            excluded["train_collision_with_official_heldout"] += 1
        else:
            eligible.append(row)

    by_task: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    biz_contexts: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        biz_contexts[row["lineage_key"]].append(row)
    for rows in biz_contexts.values():
        tasks = sorted({row["task"] for row in rows})
        component_key = min(row["lineage_key"] for row in rows)
        by_task[tasks[0]].append((component_key, rows))
    for task, groups in by_task.items():
        groups.sort(key=lambda pair: stable_hash(f"{task}\0{pair[0]}", seed))
        validation_groups = max(1, math.ceil(len(groups) * validation_ratio))
        for group_index, (_, rows) in enumerate(groups):
            output["validation" if group_index < validation_groups else "train"].extend(rows)

    seen_questions: set[str] = set()
    seen_lineages: set[str] = set()
    for split in ("test", "validation", "train"):
        split_questions = {row["question_key"] for row in output[split]}
        split_lineages = {row["lineage_key"] for row in output[split]}
        kept = []
        for row in output[split]:
            if row["question_key"] in seen_questions or row["lineage_key"] in seen_lineages:
                excluded[f"{split}_lower_priority_collision"] += 1
            else:
                kept.append(row)
        output[split] = kept
        seen_questions.update(split_questions)
        seen_lineages.update(split_lineages)

    for split in SPLITS:
        output[split].sort(key=lambda row: stable_hash(row["sample_id"], seed))
    audit = audit_splits(output)
    audit["excluded"] = dict(excluded)
    return output, audit


def audit_splits(
    splits,
    *,
    workspace: Path | None = None,
    trace_free_test: bool = False,
) -> dict:
    counts: dict[str, int] = {}
    sources: dict[str, dict] = {}
    questions: dict[str, set[str]] = {}
    lineages: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = list(splits[split])
        if not rows:
            raise AssertionError(f"Empty {split} split")
        counts[split] = len(rows)
        sources[split] = dict(Counter(f"{row['source_dataset']}/{row['task']}" for row in rows))
        questions[split] = {row["question_key"] for row in rows}
        lineages[split] = {row["lineage_key"] for row in rows}
        for row in rows:
            required = ("sample_id", "question", "answer", "target")
            if not all(str(row.get(key) or "").strip() for key in required):
                raise AssertionError(f"Incomplete row in {split}: {row.get('sample_id')}")
            if trace_free_test and split == "test":
                if str(row.get("reasoning") or "").strip() or str(row.get("program") or "").strip():
                    raise AssertionError(f"Test row exposes a reasoning trace: {row.get('sample_id')}")
                if str(row["target"]).strip() != str(row["answer"]).strip():
                    raise AssertionError(f"Test target exposes more than the final answer: {row.get('sample_id')}")
            elif not str(row.get("reasoning") or "").strip():
                raise AssertionError(f"Supervised row lacks a reasoning trace in {split}: {row.get('sample_id')}")
            if row["task_family"] in {"visual_document_reasoning", "visual_ocr"} and workspace is not None:
                image_path = workspace / row["image_path"]
                if not image_path.is_file():
                    raise AssertionError(f"Missing TAT-DQA image: {image_path}")
            if row["task_family"] == "financial_reasoning":
                if row.get("image_path"):
                    raise AssertionError(f"Reasoning row unexpectedly retains an image: {row.get('sample_id')}")
                if not str(row.get("context") or "").strip():
                    raise AssertionError(f"Reasoning row lacks evidence context: {row.get('sample_id')}")
    overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlaps[f"{left}__{right}"] = {
                "normalized_questions": len(questions[left] & questions[right]),
                "document_lineages": len(lineages[left] & lineages[right]),
            }
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise AssertionError(f"Cross-split leakage: {overlaps}")
    return {"counts": counts, "source_task_counts": sources, "overlaps": overlaps}


def _place_image(row: dict[str, Any], media_dir: Path, dest: Path) -> None:
    source = media_dir / row["image_path"]
    relative = f"images/{source.name}"
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        shutil.copyfile(source, target)
    row["image_path"] = relative


def write_capability(
    dest: Path,
    splits: dict[str, list[dict]],
    capability: str,
    media_dir: Path,
    *,
    seed: int,
    validation_ratio: float,
    render_dpi: int,
    parent_audit: dict,
) -> dict:
    """Write one capability fileset: JSONL, OCR images, and the leakage audit."""
    spec = CAPABILITIES[capability]
    allowed = set(spec["tasks"])
    projected = {split: [] for split in SPLITS}
    for split, rows in splits.items():
        for row in rows:
            if row["task"] not in allowed:
                continue
            copied = project_row(row, capability, split)
            if copied.get("image_path"):
                _place_image(copied, media_dir, dest)
            projected[split].append(copied)
    audit = audit_splits(projected, workspace=dest if spec["input_mode"] == "image" else None, trace_free_test=True)
    dest.mkdir(parents=True, exist_ok=True)
    for split, rows in projected.items():
        with (dest / JSONL_NAMES[split]).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(finance_row(row), ensure_ascii=False) + "\n")
    manifest = {
        "name": capability,
        "seed": seed,
        "bizbench_validation_ratio": validation_ratio,
        "render_dpi": render_dpi,
        "sources": {
            "TAT-DQA": {
                "repo": TAT_REPO,
                "revision": TAT_REVISION,
                "split_policy": "all public gold rows regrouped 80/10/10 by annual report",
            },
            "BizBench": {
                "repo": BIZ_REPO,
                "revision": BIZ_REVISION,
                "tasks": list(BIZ_TASKS),
                "split_policy": "official test; lineage-grouped validation from train",
            },
        },
        "tasks": list(spec["tasks"]),
        "input_mode": spec["input_mode"],
        "trace_policy": (
            "train and validation assistant content is the thinking-trace target; "
            "test assistant content is the answer only"
        ),
        "parent_audit": parent_audit,
        **audit,
    }
    (dest / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def download_sources(cache: Path) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    tat = snapshot_download(TAT_REPO, repo_type="dataset", revision=TAT_REVISION, local_dir=cache / "tatdqa")
    biz = snapshot_download(BIZ_REPO, repo_type="dataset", revision=BIZ_REVISION, local_dir=cache / "bizbench")
    return Path(tat), Path(biz)


def write_filesets(
    output: Path,
    tat_source: Path,
    biz_source: Path,
    *,
    seed: int = 42,
    validation_ratio: float = 0.1,
    render_dpi: int = 150,
) -> dict[str, dict]:
    """Write ``visual_ocr``, ``reasoning``, and ``code_generation`` under ``output``."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    media_dir = output / f"tatdqa_pages_{render_dpi}dpi_crops"
    tat = load_tat_records(tat_source, media_dir, render_dpi=render_dpi)
    biz = load_biz_records(biz_source)
    rows, parent_audit = make_splits(tat, biz, seed=seed, validation_ratio=validation_ratio)
    manifests = {}
    for capability in CAPABILITIES:
        manifests[capability] = write_capability(
            output / capability,
            rows,
            capability,
            media_dir,
            seed=seed,
            validation_ratio=validation_ratio,
            render_dpi=render_dpi,
            parent_audit=parent_audit,
        )
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tat-source-dir", type=Path, default=None)
    parser.add_argument("--biz-source-dir", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/finance-sources"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--render-dpi", type=int, default=150)
    args = parser.parse_args()
    tat_source = args.tat_source_dir
    biz_source = args.biz_source_dir
    if tat_source is None or biz_source is None:
        downloaded_tat, downloaded_biz = download_sources(args.cache)
        tat_source = tat_source or downloaded_tat
        biz_source = biz_source or downloaded_biz
    manifests = write_filesets(
        args.output,
        tat_source,
        biz_source,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        render_dpi=args.render_dpi,
    )
    print(json.dumps({name: item["counts"] for name, item in manifests.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
