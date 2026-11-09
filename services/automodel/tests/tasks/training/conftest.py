# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import the training backend modules once, under a ``nemo_automodel`` stub that is then removed.

``config.py`` imports ``nemo_automodel`` at module level, and that package exists only in the
training image. The stub is in ``sys.modules`` just long enough for that import. Left behind, it
would make every later test on the xdist worker see a fake ``nemo_automodel``; for example,
``pytest.importorskip("nemo_automodel")`` would stop skipping. The imported modules stay cached
and keep their references to the stub.
"""

import importlib
import importlib.util
import sys
from unittest.mock import MagicMock

_STUBBED = ("nemo_automodel", "nemo_automodel._transformers", "nemo_automodel._transformers.registry")
_IMPORTED_UNDER_STUB = (
    "nhx.automodel.tasks.training.backends.config",
    "nhx.automodel.tasks.training.backends.backend",
)

if importlib.util.find_spec("nemo_automodel") is None:
    stubbed = [name for name in _STUBBED if name not in sys.modules]
    for name in stubbed:
        sys.modules[name] = MagicMock()
    try:
        for module in _IMPORTED_UNDER_STUB:
            importlib.import_module(module)
    finally:
        for name in stubbed:
            del sys.modules[name]
