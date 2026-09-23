# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import unittest

from nemoguardrails.rails.llm.options import ActivatedRail, GenerationLog, GenerationResponse
from nhx.guardrails.app.schemas.utils.generation_options import (
    get_activated_rails_logging_options,
    is_activated_rails_logging_enabled,
)
from nhx.guardrails.app.schemas.utils.response_transformers import (
    create_guardrail_check_response_from_generation_response,
)
from nhx.guardrails.entities.enums import (
    StatusEnum,
)

logger = logging.getLogger(__name__)


class TestGuardrailCheckResponseCreation(unittest.TestCase):
    """Test the creation of GuardrailCheckResponse from GenerationResponse for various rail exceptions."""

    def test_content_safety_check_input_exception(self):
        """Test with content safety check input rail blocked."""
        # an activated rail that has stop=True
        activated_rail = ActivatedRail(
            type="input",
            name="content safety check input",
            decisions=["detect", "stop"],
            executed_actions=[],
            stop=True,
        )

        response = GenerationResponse(
            response=[
                {
                    "role": "exception",
                    "content": {
                        "type": "ContentSafetyCheckInputException",
                        "uid": "c8711d09-680f-40c7-8add-4869547d308e",
                        "message": "Input not allowed.",
                    },
                }
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=[activated_rail]),
            state=None,
        )
        # Only activated rails will be included in rails_status
        rails = [
            "jailbreak detection heuristics",
            "llama guard check input",
            "content safety check input",
            "self check facts",
        ]

        result = create_guardrail_check_response_from_generation_response(response, rails, True)

        # the overall status should be blocked
        self.assertEqual(result.status, StatusEnum.BLOCKED)

        # Only the activated rail should be in rails_status
        self.assertEqual(result.rails_status["content safety check input"].status, StatusEnum.BLOCKED)

        # Rails that were not activated should not be included
        self.assertNotIn("jailbreak detection heuristics", result.rails_status)
        self.assertNotIn("llama guard check input", result.rails_status)
        self.assertNotIn("self check facts", result.rails_status)

    def test_exclude_activated_rails_option(self):
        """Test with exclude_activated_rails_options=True."""
        activated_rail = ActivatedRail(
            type="input",
            name="content safety check input",
            decisions=["detect", "stop"],
            executed_actions=[],
            stop=True,
        )

        response = GenerationResponse(
            response=[
                {
                    "role": "exception",
                    "content": {
                        "type": "ContentSafetyCheckInputException",
                        "uid": "c8711d09-680f-40c7-8add-4869547d308e",
                        "message": "Input not allowed.",
                    },
                }
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=[activated_rail]),
            state=None,
        )
        rails = ["content safety check input"]

        # activated_rails is not empty
        self.assertEqual(len(response.log.activated_rails), 1)

        # exclude_activated_rails_options=True
        create_guardrail_check_response_from_generation_response(response, rails, exclude_activated_rails_options=True)

        # activated_rails is now an empty list
        self.assertEqual(len(response.log.activated_rails), 0)

    def test_multiple_stopped_rails(self):
        """Test when multiple rails have stop=True."""
        # create multiple rails with stop=True
        activated_rails = [
            ActivatedRail(
                type="input",
                name="content safety check input",
                decisions=["detect", "stop"],
                executed_actions=[],
                stop=True,
            ),
            ActivatedRail(
                type="input",
                name="jailbreak detection heuristics",
                decisions=["detect", "stop"],
                executed_actions=[],
                stop=True,
            ),
        ]

        response = GenerationResponse(
            response=[
                {
                    "role": "exception",
                    "content": {
                        "type": "InputRailException",
                        "uid": "c8711d09-680f-40c7-8add-4869547d308e",
                        "message": "Input not allowed.",
                    },
                }
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=activated_rails),
            state=None,
        )
        rails = ["content safety check input", "jailbreak detection heuristics"]

        result = create_guardrail_check_response_from_generation_response(response, rails, True)

        # only the first rail should be marked as blocked due to the break in the loop
        self.assertEqual(result.status, StatusEnum.BLOCKED)
        self.assertEqual(result.rails_status["content safety check input"].status, StatusEnum.BLOCKED)
        self.assertNotIn("jailbreak detection heuristics", result.rails_status)

    def test_multiple_with_no_stoppers_but_exception_present(self):
        """Test when multiple rails have stop=False but exception is present."""
        # create multiple rails with stop=False
        activated_rails = [
            ActivatedRail(
                type="input",
                name="content safety check input",
                decisions=["detect"],
                executed_actions=[],
                stop=False,
            ),
            ActivatedRail(
                type="input",
                name="jailbreak detection heuristics",
                decisions=["detect"],
                executed_actions=[],
                stop=False,
            ),
        ]

        response = GenerationResponse(
            response=[
                {
                    "role": "exception",
                    "content": {
                        "type": "InputRailException",
                        "uid": "c8711d09-680f-40c7-8add-4869547d308e",
                        "message": "Input not allowed.",
                    },
                }
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=activated_rails),
            state=None,
        )
        rails = ["content safety check input", "jailbreak detection heuristics"]

        result = create_guardrail_check_response_from_generation_response(response, rails, True)

        # Overall status is BLOCKED due to exception, but rails themselves show UNKNOWN
        self.assertEqual(result.status, StatusEnum.BLOCKED)
        self.assertEqual(result.rails_status["content safety check input"].status, StatusEnum.UNKNOWN)
        self.assertEqual(result.rails_status["jailbreak detection heuristics"].status, StatusEnum.UNKNOWN)

    def test_only_one_rail_blocked(self):
        """Test when there is only one blocked rail."""

        activated_rails = [
            ActivatedRail(
                type="output",
                name="self check output",
                decisions=["execute self_check_output"],
                executed_actions=[],
                stop=True,
            ),
        ]

        response = GenerationResponse(
            response=[
                {
                    "role": "exception",
                    "content": {
                        "type": "OutputRailException",
                        "uid": "5abba554-8278-4fb6-a95c-f22b22195a85",
                        "event_created_at": "2025-04-03T18:25:57.590337+00:00",
                        "source_uid": "NeMoGuardrails",
                        "message": "Output not allowed. The output was blocked by the 'self check output' flow.",
                    },
                }
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=activated_rails),
            state=None,
        )

        rails = ["self check output"]
        result = create_guardrail_check_response_from_generation_response(response, rails, True)
        self.assertEqual(result.status, StatusEnum.BLOCKED)
        self.assertEqual(result.rails_status["self check output"].status, StatusEnum.BLOCKED)

    def test_no_blocked_rails(self):
        """Test when there are no blocked rails."""
        activated_rails = [
            ActivatedRail(
                type="input", name="content safety check input", decisions=["pass"], executed_actions=[], stop=False
            ),
            ActivatedRail(
                type="input", name="jailbreak detection heuristics", decisions=["pass"], executed_actions=[], stop=False
            ),
        ]

        response = GenerationResponse(
            response=[
                {
                    "role": "assistant",  # it is not an exception
                    "content": "This is a normal response",
                }
            ],
            llm_output=None,
            output_data=None,
            log=GenerationLog(activated_rails=activated_rails),
            state=None,
        )
        rails = ["content safety check input", "jailbreak detection heuristics"]

        result = create_guardrail_check_response_from_generation_response(response, rails, True)

        # all of the rails should be marked as success
        self.assertEqual(result.status, StatusEnum.SUCCESS)
        self.assertEqual(result.rails_status["content safety check input"].status, StatusEnum.SUCCESS)
        self.assertEqual(result.rails_status["jailbreak detection heuristics"].status, StatusEnum.SUCCESS)

    def test_no_log(self):
        """Test when there is no log in the response."""
        response = GenerationResponse(
            response=[
                {
                    "role": "exception",
                    "content": {
                        "type": "ContentSafetyCheckInputException",
                        "uid": "c8711d09-680f-40c7-8add-4869547d308e",
                        "message": "Input not allowed.",
                    },
                }
            ],
            llm_output=None,
            output_data=None,
            log=None,  # no log
            state=None,
        )
        rails = ["content safety check input"]

        exclude_activated_rails_options = get_activated_rails_logging_options(response.log)
        exclude_activated_rails_logging_options = is_activated_rails_logging_enabled(exclude_activated_rails_options)

        result = create_guardrail_check_response_from_generation_response(
            response, rails, exclude_activated_rails_options=exclude_activated_rails_logging_options
        )

        # Overall status is BLOCKED due to exception
        self.assertEqual(result.status, StatusEnum.BLOCKED)
        # No activated rails means rails_status should be empty
        self.assertEqual(len(result.rails_status), 0)
        self.assertNotIn("content safety check input", result.rails_status)
