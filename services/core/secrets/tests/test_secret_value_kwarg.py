# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the data -> value field rename in secret request/response models.

Users intuitively reach for `value=` or `secret=` when creating Secrets through
the typed client, but the field was named `data`. This verifies the rename is
complete and consistent.
"""

import pytest
from nhx.core.secrets.api.v2.secrets.schemas import (
    HelixSecretAccessResponse,
    HelixSecretCreateRequest,
    HelixSecretUpdateRequest,
)
from pydantic import ValidationError


class TestCreateRequestValueField:
    def test_accepts_value_kwarg(self):
        model = HelixSecretCreateRequest(name="test-secret", value="my-secret-value")
        assert model.value.get_secret_value() == "my-secret-value"

    def test_no_data_field_exists(self):
        assert "data" not in HelixSecretCreateRequest.model_fields

    def test_json_body_with_value(self):
        model = HelixSecretCreateRequest.model_validate({"name": "test", "value": "secret"})
        assert model.value.get_secret_value() == "secret"

    def test_json_body_with_data_rejected(self):
        with pytest.raises(ValidationError, match="value"):
            HelixSecretCreateRequest.model_validate({"name": "test", "data": "secret"})

    def test_empty_value_rejected(self):
        with pytest.raises(ValidationError):
            HelixSecretCreateRequest(name="test-secret", value="")


class TestUpdateRequestValueField:
    def test_accepts_value_kwarg(self):
        model = HelixSecretUpdateRequest(value="new-value")
        assert model.value.get_secret_value() == "new-value"

    def test_no_data_field_exists(self):
        assert "data" not in HelixSecretUpdateRequest.model_fields

    def test_empty_value_rejected(self):
        with pytest.raises(ValidationError):
            HelixSecretUpdateRequest.model_validate({"value": ""})


class TestAccessResponseValueField:
    def test_uses_value_field(self):
        model = HelixSecretAccessResponse(name="test-secret", workspace="default", value="my-secret-value")
        assert model.value == "my-secret-value"

    def test_no_data_field_exists(self):
        assert "data" not in HelixSecretAccessResponse.model_fields
