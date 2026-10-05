# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from nhx.guardrails.app.schemas.utils.generation_options import (
    get_activated_rails_logging_options,
    is_activated_rails_logging_enabled,
    update_generation_options,
)
from nhx.guardrails.entities.values._private import GenerationOptions
from nhx.guardrails.entities.values.common import GuardrailsDataInput


class TestGenerationOptions(unittest.TestCase):
    def setUp(self):
        self.guardrails_data = GuardrailsDataInput()
        self.options = {"log": {"activated_rails": True}}

    def test_update_generation_options(self):
        updated_data = update_generation_options(self.guardrails_data, self.options)
        self.assertTrue(hasattr(updated_data, "options"))
        self.assertEqual(updated_data.options.log["activated_rails"], True)

    def test_get_activated_rails_logging_options(self):
        self.guardrails_data.options = GenerationOptions(**self.options)
        result = get_activated_rails_logging_options(self.guardrails_data)
        self.assertEqual(result, {"log": {"activated_rails": True}})

    def test_is_activated_rails_logging_enabled(self):
        result = is_activated_rails_logging_enabled(self.options)
        self.assertTrue(result)

    def test_is_activated_rails_logging_disabled(self):
        options = {"log": {"activated_rails": False}}
        result = is_activated_rails_logging_enabled(options)
        self.assertFalse(result)
