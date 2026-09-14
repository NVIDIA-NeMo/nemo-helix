# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from nemo_platform import AsyncNeMoPlatform


@pytest.mark.parametrize(
    "uri",
    [
        "nemo-evaluator-taskset://default/suite",
        "nemo-evaluator-taskset://default/suite#latest",
        f"nemo-evaluator-taskset://other/suite#{'a' * 64}",
        f"nemo-evaluator-taskset://default/%2e%2e#{'a' * 64}",
        f"nemo-evaluator-taskset://default/suite?extra=1#{'a' * 64}",
        f"nemo-evaluator-taskset://default/suite/extra#{'a' * 64}",
        f"nemo-evaluator-taskset://default/su\nite#{'a' * 64}",
        f"\tnemo-evaluator-taskset://default/suite#{'a' * 64}",
    ],
)
async def test_adapter_rejects_invalid_uri_before_client_or_storage(uri, tmp_path):
    from nemo_evaluator.harbor.taskset_source import EvaluatorTasksetSourceAdapter

    adapter = EvaluatorTasksetSourceAdapter(client=MagicMock(spec=AsyncNeMoPlatform), workspace="default")
    with pytest.raises(ValueError):
        await adapter.materialize(uri, destination_root=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()
