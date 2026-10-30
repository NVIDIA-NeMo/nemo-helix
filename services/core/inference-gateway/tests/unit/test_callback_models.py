# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A VirtualModel names the VirtualModels its provider calls back through the gateway."""

import pytest
from nemo_helix_plugin.virtual_models.types import CreateVirtualModelRequest, UpdateVirtualModelRequest
from pydantic import ValidationError


def test_callback_models_are_kept_as_given() -> None:
    request = CreateVirtualModelRequest(name="routed", callback_models=["default/strong", "default/judge"])

    assert request.callback_models == ["default/strong", "default/judge"]


@pytest.mark.parametrize("ref", ["strong", "default/", "/strong", "default/strong/extra"])
@pytest.mark.parametrize("request_type", [CreateVirtualModelRequest, UpdateVirtualModelRequest])
def test_a_callback_model_must_be_a_workspace_name_reference(request_type: type, ref: str) -> None:
    fields = {"callback_models": [ref]} | ({"name": "routed"} if request_type is CreateVirtualModelRequest else {})

    with pytest.raises(ValidationError, match="workspace/name"):
        request_type(**fields)
