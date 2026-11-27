# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_user_sim_plugin.model_check import check_models


async def test_each_role_reports_whether_its_model_answers():
    asked = []

    async def ask(role, model):
        asked.append(model)
        if model == "down":
            raise RuntimeError("Error code: 500 - {'detail': '500'}")

    statuses = await check_models({"user_model": "up", "judge_model": "down", "evaluator_model": "down"}, ask)

    assert [(s.role, s.model, s.ok) for s in statuses] == [
        ("user_model", "up", True),
        ("judge_model", "down", False),
        ("evaluator_model", "down", False),
    ]
    assert statuses[1].error == "Error code: 500 - {'detail': '500'}"
    # A model shared by two roles is asked once.
    assert sorted(asked) == ["down", "up"]
