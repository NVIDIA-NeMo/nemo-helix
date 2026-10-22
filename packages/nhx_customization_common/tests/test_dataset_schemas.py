# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFTMultimodalChatDatasetItemSchema accepts part content. SFT chat does not."""

import json
from typing import get_args

import jsonschema
import pytest
from nhx.customization_common.schemas.datasets import (
    MULTIMODAL_SCHEMA,
    SFTChatDatasetItemSchema,
    SFTChatMessage,
    SFTDatasetSchemaType,
    SFTMultimodalChatDatasetItemSchema,
    SFTMultimodalChatMessage,
    SFTMultimodalImagePart,
    SFTMultimodalTextPart,
    row_has_multimodal_content,
)
from pydantic import TypeAdapter, ValidationError

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


def test_multimodal_schema_accepts_each_content_shape() -> None:
    row = SFTMultimodalChatDatasetItemSchema.model_validate(
        {
            "messages": [
                *CORD_ROW["messages"],
                {"role": "user", "content": {"type": "text", "text": "Describe this image."}},
                {
                    "role": "user",
                    "content": {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                },
                {"role": "assistant", "content": {"role": "assistant", "content": "<s_menu>"}},
            ]
        }
    )

    text_part = row.messages[2].content
    image_part = row.messages[3].content
    nested = row.messages[4].content
    assert row.messages[1].content == "<s_menu><s_nm>"
    assert isinstance(row.messages[0].content, list)
    assert isinstance(text_part, SFTMultimodalTextPart)
    assert text_part.type == "text"
    assert isinstance(image_part, SFTMultimodalImagePart)
    assert image_part.type == "image_url"
    assert isinstance(nested, SFTChatMessage)
    assert nested.role == "assistant"


def test_chat_message_rejects_a_part_list() -> None:
    with pytest.raises(ValidationError):
        SFTChatMessage.model_validate({"role": "user", "content": [{"type": "text", "text": "Describe this image."}]})


def test_chat_dataset_rejects_a_multimodal_row_and_accepts_string_content() -> None:
    with pytest.raises(ValidationError):
        SFTChatDatasetItemSchema.model_validate(CORD_ROW)

    accepted = SFTChatDatasetItemSchema.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Describe this image."},
                {"role": "assistant", "content": "<s_menu>"},
            ]
        }
    )
    assert accepted.messages[0].content == "Describe this image."


def test_sft_dataset_union_does_not_include_the_multimodal_schema() -> None:
    union = get_args(SFTDatasetSchemaType)[0]
    names = [get_args(member)[0].__name__ for member in get_args(union)]
    assert "SFTMultimodalChatDatasetItemSchema" not in names

    with pytest.raises(ValidationError):
        TypeAdapter(SFTDatasetSchemaType).validate_python(CORD_ROW)


def test_multimodal_message_rejects_other_part_types_and_empty_content() -> None:
    with pytest.raises(ValidationError):
        SFTMultimodalChatMessage.model_validate({"role": "user", "content": {"type": "image", "image": "receipt.png"}})
    with pytest.raises(ValidationError):
        SFTMultimodalChatMessage.model_validate({"role": "user", "content": ""})
    with pytest.raises(ValidationError):
        SFTMultimodalChatMessage.model_validate({"role": "user", "content": []})


def test_chat_message_content_stays_a_string() -> None:
    content = json.dumps(SFTChatMessage.model_json_schema()["properties"]["content"])
    assert "image_url" not in content
    assert "array" not in content


def test_row_has_multimodal_content_ignores_string_messages() -> None:
    assert row_has_multimodal_content({"messages": [{"role": "user", "content": "Describe this image."}]}) is False
    assert row_has_multimodal_content(CORD_ROW) is True
    assert row_has_multimodal_content(
        {"messages": [{"role": "user", "content": {"type": "text", "text": "Describe this image."}}]}
    )
    assert row_has_multimodal_content(
        {"messages": [{"role": "assistant", "content": {"role": "assistant", "content": "<s_menu>"}}]}
    )


def test_multimodal_json_schema_accepts_a_part_list_and_rejects_an_unknown_part() -> None:
    schema = MULTIMODAL_SCHEMA()
    jsonschema.validate(CORD_ROW, schema)
    jsonschema.validate({"messages": [{"role": "user", "content": "Describe this image."}]}, schema)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(
            {"messages": [{"role": "user", "content": [{"type": "image", "image": "images/receipt.png"}]}]},
            schema,
        )
