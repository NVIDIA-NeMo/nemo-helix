# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from sandboxed_gym.runtime import gym_host_runtime as runtime


@pytest.fixture
def isolated_gym_host_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests that execute the real ``_install_wheels_v1_dependencies`` helper.

    Use this fixture whenever a test reaches that helper directly or through
    ``bootstrap_gym_host``; tests that replace the helper with a mock do not need it.
    """
    # The helper writes its uv wheel search path, uv offline switch and Python import path straight
    # into os.environ. setenv first so monkeypatch records each key even when it was unset, and
    # teardown restores the caller's value or removes the helper's.
    for key in (runtime.UV_FIND_LINKS_ENV_KEY, runtime.UV_OFFLINE_ENV_KEY, "PYTHONPATH"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    # Let the helper extend a copy of the import path instead of this process's sys.path.
    import_path = runtime.sys.path.copy()
    monkeypatch.setattr(runtime, "_import_path", lambda: import_path)
