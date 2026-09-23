# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nhx.guardrails.api.schemas import BaseRequest

logger = logging.getLogger(__name__)


def add_custom_init_to_base_request():
    def __init__(__pydantic_self__, **data):
        if "streaming" in data:
            data["stream"] = data.pop("streaming")

            # convert string to boolean, it is the side-effect forwarding response from the backend thru llm_output
            if isinstance(data["stream"], str):
                data["stream"] = data["stream"].lower() == "true"

        # Explicitly call the superclass's __init__ method
        super(BaseRequest, __pydantic_self__).__init__(**data)

    setattr(BaseRequest, "__init__", __init__)
    logger.debug("Injected __init__ into BaseRequest")


add_custom_init_to_base_request()
