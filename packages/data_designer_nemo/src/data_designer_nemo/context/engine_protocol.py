# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shape the Data Designer engine expects from a context."""

from typing import Protocol

from data_designer.engine.resources.person_reader import PersonReader
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.engine.secret_resolver import SecretResolver


class DataDesignerEngineContext(Protocol):
    """What ``create_data_designer`` needs from a context.

    Implemented by
    :class:`~data_designer_nemo.context.execution.DataDesignerExecutionContext`
    for real workloads, and by
    :class:`~data_designer_nemo.context.check_models.DataDesignerCheckModelsContext`
    for model probes, which need none of these components to do real work.
    """

    def get_secret_resolver(self) -> SecretResolver: ...

    def get_seed_readers(self) -> list[SeedReader]: ...

    def get_person_reader(self) -> PersonReader | None: ...
