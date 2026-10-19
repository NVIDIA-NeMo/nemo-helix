# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from nemo_anonymizer_plugin.app.upstream_logging import preserve_root_logging
from nemo_anonymizer_plugin.functions import _preview_worker as worker_module


def test_preserve_root_logging_restores_root_handlers_and_level() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_filters = list(root.filters)
    original_level = root.level
    original_disabled = root.disabled
    original_propagate = root.propagate

    sentinel_handler = logging.NullHandler()
    sentinel_filter = logging.Filter("sentinel")

    try:
        root.handlers = [sentinel_handler]
        root.filters = [sentinel_filter]
        root.setLevel(logging.DEBUG)
        root.disabled = False
        root.propagate = True

        with preserve_root_logging():
            root.handlers.clear()
            root.addHandler(logging.StreamHandler())
            root.filters.clear()
            root.setLevel(logging.ERROR)
            root.disabled = True
            root.propagate = False

        assert root.handlers == [sentinel_handler]
        assert root.filters == [sentinel_filter]
        assert root.level == logging.DEBUG
        assert root.disabled is False
        assert root.propagate is True
    finally:
        root.handlers = original_handlers
        root.filters = original_filters
        root.setLevel(original_level)
        root.disabled = original_disabled
        root.propagate = original_propagate


def test_make_anonymizer_preserves_root_logging_when_upstream_clobbers_it(monkeypatch) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    sentinel_handler = logging.NullHandler()

    class ClobberingAnonymizer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            root.handlers.clear()
            root.addHandler(logging.StreamHandler())
            root.setLevel(logging.ERROR)

    try:
        root.handlers = [sentinel_handler]
        root.setLevel(logging.INFO)
        # ``_make_anonymizer`` builds via ``build_gliner_anonymizer``; stand in for it
        # with a factory that clobbers root logging, to prove the wrapper restores it.
        monkeypatch.setattr(
            worker_module,
            "build_gliner_anonymizer",
            lambda *, model_configs_yaml, dd_providers: ClobberingAnonymizer(
                model_configs=model_configs_yaml, model_providers=dd_providers
            ),
        )

        anonymizer = worker_module._make_anonymizer(model_configs_yaml="model_configs: []", dd_providers=None)

        assert isinstance(anonymizer, ClobberingAnonymizer)
        assert root.handlers == [sentinel_handler]
        assert root.level == logging.INFO
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)
