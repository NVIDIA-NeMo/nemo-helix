# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exist_ok post-generation transformer."""

import textwrap
from pathlib import Path

import libcst as cst
from nemo_helix_sdk_tools.sdk.post_generation_exist_ok import (
    _ConflictErrorImportAdder,
    _ExistOkInjector,
    _should_process_file,
)


def _transform(source: str) -> str:
    """Run both the injector and import adder on source code, return the result."""
    tree = cst.parse_module(textwrap.dedent(source))
    injector = _ExistOkInjector()
    tree = tree.visit(injector)
    if injector.modified:
        adder = _ConflictErrorImportAdder()
        tree = tree.visit(adder)
    return tree.code


def _injector_modified(source: str) -> bool:
    """Return whether the injector marked the source as modified."""
    tree = cst.parse_module(textwrap.dedent(source))
    injector = _ExistOkInjector()
    tree.visit(injector)
    return injector.modified


# --- Eligibility tests ---


class TestEligibility:
    def test_class_with_create_and_retrieve_is_eligible(self):
        source = """\
        from ..._types import Body, Query, Headers, NotGiven, not_given
        class MyResource:
            def create(self, *, name: str, extra_headers=None):
                return self._post()
            def retrieve(self, name: str, *, workspace=None):
                return self._get()
        """
        assert _injector_modified(source)

    def test_class_without_retrieve_is_skipped(self):
        source = """\
        class MyResource:
            def create(self, *, name: str):
                return self._post()
        """
        assert not _injector_modified(source)

    def test_class_without_create_is_skipped(self):
        source = """\
        class MyResource:
            def retrieve(self, name: str):
                return self._get()
        """
        assert not _injector_modified(source)

    def test_create_without_name_param_is_skipped(self):
        """Role bindings have create(principal=, role=) but no name param."""
        source = """\
        class RoleBindingsResource:
            def create(self, *, principal: str, role: str):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        assert not _injector_modified(source)

    def test_already_has_exist_ok_is_skipped(self):
        source = """\
        class MyResource:
            def create(self, *, name: str, exist_ok: bool = False):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        assert not _injector_modified(source)


# --- Sync transform tests ---


class TestSyncTransform:
    def test_adds_exist_ok_param(self):
        source = """\
        class MyResource:
            def create(self, *, name: str, extra_headers=None):
                return self._post()
            def retrieve(self, name: str, *, workspace=None):
                return self._get()
        """
        result = _transform(source)
        assert "exist_ok: bool = False" in result

    def test_wraps_body_in_try_except(self):
        source = """\
        class MyResource:
            def create(self, *, name: str, extra_headers=None):
                return self._post()
            def retrieve(self, name: str, *, workspace=None):
                return self._get()
        """
        result = _transform(source)
        assert "try:" in result
        assert "except ConflictError:" in result

    def test_except_handler_reraises_when_not_exist_ok(self):
        source = """\
        class MyResource:
            def create(self, *, name: str):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        assert "if not exist_ok:" in result
        assert "raise" in result

    def test_fallback_calls_retrieve_with_name_and_workspace(self):
        source = """\
        class MyResource:
            def create(self, *, workspace=None, name: str):
                return self._post()
            def retrieve(self, name: str, *, workspace=None):
                return self._get()
        """
        result = _transform(source)
        assert "self.retrieve(name" in result
        assert "workspace" in result.split("self.retrieve")[1].split(")")[0]

    def test_fallback_calls_retrieve_without_workspace_when_absent(self):
        """Workspaces resource: retrieve takes name only, no workspace."""
        source = """\
        class WorkspacesResource:
            def create(self, *, name: str):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        assert "self.retrieve(name" in result
        assert "workspace" not in result.split("self.retrieve")[1].split(")")[0]

    def test_docstring_preserved_outside_try(self):
        source = '''\
        class MyResource:
            def create(self, *, name: str):
                """Create a resource."""
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        '''
        result = _transform(source)
        lines = result.splitlines()
        docstring_line = next(i for i, line in enumerate(lines) if "Create a resource." in line)
        try_line = next(i for i, line in enumerate(lines) if "try:" in line)
        assert docstring_line < try_line

    def test_exist_ok_inserted_before_extra_headers(self):
        source = """\
        class MyResource:
            def create(self, *, name: str, extra_headers=None, extra_query=None):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        exist_ok_pos = result.index("exist_ok")
        extra_headers_pos = result.index("extra_headers")
        assert exist_ok_pos < extra_headers_pos


# --- Async transform tests ---


class TestAsyncTransform:
    def test_async_create_gets_exist_ok(self):
        source = """\
        class AsyncMyResource:
            async def create(self, *, name: str):
                return await self._post()
            async def retrieve(self, name: str, *, workspace=None):
                return await self._get()
        """
        result = _transform(source)
        assert "exist_ok: bool = False" in result

    def test_async_fallback_uses_await(self):
        source = """\
        class AsyncMyResource:
            async def create(self, *, name: str):
                return await self._post()
            async def retrieve(self, name: str, *, workspace=None):
                return await self._get()
        """
        result = _transform(source)
        assert "await self.retrieve(name" in result
        assert "workspace" in result.split("await self.retrieve")[1].split(")")[0]

    def test_async_wraps_in_try_except(self):
        source = """\
        class AsyncMyResource:
            async def create(self, *, name: str):
                return await self._post()
            async def retrieve(self, name: str):
                return await self._get()
        """
        result = _transform(source)
        assert "try:" in result
        assert "except ConflictError:" in result


# --- Both sync and async in same file ---


class TestSyncAndAsyncInSameFile:
    def test_both_variants_transformed(self):
        source = """\
        class MyResource:
            def create(self, *, name: str):
                return self._post()
            def retrieve(self, name: str):
                return self._get()

        class AsyncMyResource:
            async def create(self, *, name: str):
                return await self._post()
            async def retrieve(self, name: str):
                return await self._get()
        """
        result = _transform(source)
        assert result.count("exist_ok: bool = False") == 2
        assert result.count("except ConflictError:") == 2


# --- Overload tests ---


class TestOverloadHandling:
    def test_overload_stub_gets_param_but_no_try_except(self):
        source = """\
        from typing import overload
        class MyResource:
            @overload
            def create(self, *, name: str, type: str) -> Response:
                ...
            def create(self, *, name: str, **kwargs):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        assert result.count("exist_ok: bool = False") == 2
        assert result.count("except ConflictError:") == 1

    def test_multiple_overloads_no_body_wrapping(self):
        source = """\
        from typing import overload
        class MyResource:
            @overload
            def create(self, *, name: str, type: str) -> ResponseA:
                ...
            @overload
            def create(self, *, name: str, mode: int) -> ResponseB:
                ...
            def create(self, *, name: str, **kwargs):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        assert result.count("exist_ok: bool = False") == 3
        assert result.count("except ConflictError:") == 1


# --- Import adder tests ---


class TestConflictErrorImportAdder:
    def test_adds_import_when_missing(self):
        source = """\
        from ..._types import Body
        class MyResource:
            def create(self, *, name: str):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        assert "from ..._exceptions import ConflictError" in result

    def test_does_not_duplicate_import(self):
        source = """\
        from ..._exceptions import ConflictError
        class MyResource:
            def create(self, *, name: str):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        import_lines = [line for line in result.splitlines() if "import ConflictError" in line]
        assert len(import_lines) == 1

    def test_detects_relative_depth_from_existing_imports(self):
        source = """\
        from .._utils import something
        class MyResource:
            def create(self, *, name: str):
                return self._post()
            def retrieve(self, name: str):
                return self._get()
        """
        result = _transform(source)
        assert "from .._exceptions import ConflictError" in result


# --- File filtering tests ---


class TestShouldProcessFile:
    def test_normal_resource_file(self):
        assert _should_process_file(Path("projects/projects.py"))

    def test_skips_jobs(self):
        assert not _should_process_file(Path("audit/jobs/jobs.py"))

    def test_skips_completions_directory(self):
        assert not _should_process_file(Path("guardrail/completions/completions.py"))

    def test_skips_underscore_files(self):
        assert not _should_process_file(Path("__init__.py"))

    def test_skips_benchmark_jobs(self):
        assert not _should_process_file(Path("evaluation/benchmark_jobs/benchmark_jobs.py"))

    def test_skips_metric_jobs(self):
        assert not _should_process_file(Path("evaluation/metric_jobs/metric_jobs.py"))

    def test_skips_gateway(self):
        assert not _should_process_file(Path("inference/gateway/gateway.py"))

    def test_skips_members(self):
        assert not _should_process_file(Path("members/members.py"))

    def test_skips_adapters_directory(self):
        assert not _should_process_file(Path("models/adapters/adapters.py"))

    def test_skips_results(self):
        assert not _should_process_file(Path("jobs/results.py"))
