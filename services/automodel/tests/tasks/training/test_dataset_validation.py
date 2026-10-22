# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from nhx.automodel.entities.values import TrainingType
from nhx.automodel.tasks.training.datasets import validation as validation_mod
from nhx.automodel.tasks.training.datasets.preparation import DatasetFormatError, DatasetSchema
from nhx.automodel.tasks.training.datasets.preparation import detect_dataset_schema as detect_prepared_schema
from nhx.automodel.tasks.training.datasets.validation import DatasetValidator
from nhx.customization_common.schemas.datasets import MULTIMODAL_DATASET


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_dataset_compiles_schema_once(tmp_path: Path, mocker) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"query": "q", "pos_doc": "p", "neg_doc": []} for _ in range(50)])
    compile_spy = mocker.spy(validation_mod, "_compile_validator")

    DatasetValidator(training_type=TrainingType.SFT).validate_dataset(str(path))

    assert compile_spy.call_count == 1


def test_validate_dataset_accepts_embedding_rows(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"query": "q", "pos_doc": "p", "neg_doc": ["n"]}])
    DatasetValidator(training_type=TrainingType.SFT).validate_dataset(str(path))


CORD_ROW = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "images/receipt.png"}},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {"role": "assistant", "content": "<s_menu><s_nm>"},
    ]
}

CHAT_ROW = {
    "messages": [
        {"role": "user", "content": "Describe this image."},
        {"role": "assistant", "content": "<s_menu>"},
    ]
}


def test_sft_schema_rejects_a_part_list(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [CORD_ROW])
    with pytest.raises(DatasetFormatError):
        DatasetValidator(training_type=TrainingType.SFT).validate_dataset(str(path))


def test_multimodal_schema_accepts_part_list_single_part_nested_message_and_string(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            CORD_ROW,
            {"messages": [{"role": "user", "content": {"type": "text", "text": "Describe this image."}}]},
            {"messages": [{"role": "assistant", "content": {"role": "assistant", "content": "<s_menu>"}}]},
            CHAT_ROW,
        ],
    )
    DatasetValidator(training_type=TrainingType.SFT).validate_dataset(str(path), dataset_type=MULTIMODAL_DATASET)


def test_multimodal_schema_rejects_an_unknown_part_type(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [{"messages": [{"role": "user", "content": [{"type": "image", "image": "images/receipt.png"}]}]}],
    )
    with pytest.raises(DatasetFormatError):
        DatasetValidator(training_type=TrainingType.SFT).validate_dataset(str(path), dataset_type=MULTIMODAL_DATASET)


def test_detect_keeps_string_chat_on_sft_and_labels_part_content_multimodal(tmp_path: Path) -> None:
    chat = tmp_path / "chat.jsonl"
    image = tmp_path / "image.jsonl"
    later = tmp_path / "later.jsonl"
    _write_jsonl(chat, [CHAT_ROW])
    _write_jsonl(image, [CORD_ROW])
    _write_jsonl(
        later,
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "user",
                        "content": {"type": "image_url", "image_url": {"url": "images/receipt.png"}},
                    },
                ]
            }
        ],
    )

    validator = DatasetValidator(training_type=TrainingType.SFT)
    assert validator.detect_dataset_schema(str(chat)) == TrainingType.SFT.value
    assert validator.detect_dataset_schema(str(image)) == MULTIMODAL_DATASET
    assert detect_prepared_schema(chat)[0] == DatasetSchema.CHAT
    assert detect_prepared_schema(image)[0] == DatasetSchema.MULTIMODAL
    assert detect_prepared_schema(later)[0] == DatasetSchema.MULTIMODAL
