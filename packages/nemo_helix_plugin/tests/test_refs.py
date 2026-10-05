# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`nemo_helix_plugin.refs`.

Covers the generic ref machinery (the :class:`StrRef` Pydantic hook,
the path-shape classifier, and :data:`OutputTarget` round-tripping) that
plugin authors reuse when building union-typed spec fields.

Plugin-specific subclasses (e.g. ``nemo_agents_plugin.refs.AgentRef``)
are exercised in their respective plugin test suites; here we stick to
the building blocks shipped from this package so the suite is hermetic
and doesn't require any plugin to be installed.
"""

from __future__ import annotations

import pytest
from nemo_helix_plugin.refs import (
    EndpointURL,
    FilesetRef,
    LocalDir,
    OutputTarget,
    StrRef,
    classify_output_target,
)
from pydantic import BaseModel


class TestStrRefSubclasses:
    """The shipped subclasses behave like plain strings everywhere strings are used."""

    def test_endpoint_url_is_a_str(self) -> None:
        assert isinstance(EndpointURL("http://x"), str)

    def test_local_dir_is_a_str(self) -> None:
        assert isinstance(LocalDir("./out"), str)

    def test_fileset_ref_is_a_str(self) -> None:
        assert isinstance(FilesetRef("eval-results"), str)

    def test_subclass_identity_preserved(self) -> None:
        """``isinstance`` checks let resolvers branch on which arm was instantiated."""
        assert isinstance(LocalDir("./out"), LocalDir)
        assert not isinstance(LocalDir("./out"), FilesetRef)

    def test_equality_with_plain_str(self) -> None:
        assert FilesetRef("eval-results") == "eval-results"
        assert "eval-results" == FilesetRef("eval-results")


class TestStrRefMetavars:
    """Each shipped subclass advertises the placeholder the CLI generator picks up."""

    def test_base_class_has_no_metavar(self) -> None:
        assert StrRef.__cli_metavar__ is None

    def test_endpoint_url_metavar(self) -> None:
        assert EndpointURL.__cli_metavar__ == "URL"

    def test_local_dir_metavar(self) -> None:
        assert LocalDir.__cli_metavar__ == "PATH"

    def test_fileset_ref_metavar(self) -> None:
        assert FilesetRef.__cli_metavar__ == "FILESET_REF"


class TestClassifyOutputTarget:
    """Path-shape markers route to :class:`LocalDir`; bare names to :class:`FilesetRef`."""

    @pytest.mark.parametrize(
        "value",
        [
            "/abs/out",
            "./out",
            "../parent/out",
            "~/home/out",
            "~",
            r"C:\Users\me\out",
            r"out\sub",
            # Windows absolute paths with forward slashes — common when
            # users paste paths that have already been normalised (e.g. by
            # WSL or pathlib's PurePosixPath repr) but still carry a
            # drive letter.
            "C:/Users/me/out",
            "d:/tmp",
        ],
    )
    def test_path_shaped_classify_as_local_dir(self, value: str) -> None:
        assert classify_output_target(value) is LocalDir

    @pytest.mark.parametrize(
        "value",
        [
            "eval-results",
            "default/eval-results",
            "my-fileset",
            "ws-1/calc-eval",
        ],
    )
    def test_bare_names_classify_as_fileset_ref(self, value: str) -> None:
        assert classify_output_target(value) is FilesetRef


class TestPydanticIntegration:
    """The :class:`StrRef` hook lets Pydantic accept any string for a ref-typed field.

    Disambiguation is intentionally *not* validated at the model layer
    — both arms of :data:`OutputTarget` accept any string value.
    Resolvers inspect the shape at run time when they have the runtime
    context to give an actionable error.
    """

    def test_concrete_subclass_field_round_trips(self) -> None:
        """A field typed as the concrete subclass coerces ``str`` -> subclass.

        ``ty`` doesn't know that the :class:`StrRef` Pydantic core schema
        accepts a bare ``str`` and wraps it in the subclass, so the
        literal-arg call needs an explicit ignore — at runtime the
        coercion is what we're actually asserting on the next line.
        """

        class _M(BaseModel):
            path: LocalDir

        m = _M(path="./out")  # ty: ignore[invalid-argument-type]
        assert isinstance(m.path, LocalDir)
        assert m.path == "./out"

    def test_union_field_accepts_local_path(self) -> None:
        class _M(BaseModel):
            out: OutputTarget | None = None

        assert _M(out="./eval-out").out == "./eval-out"  # ty: ignore[invalid-argument-type]

    def test_union_field_accepts_fileset_ref(self) -> None:
        class _M(BaseModel):
            out: OutputTarget | None = None

        assert _M(out="default/eval-results").out == "default/eval-results"  # ty: ignore[invalid-argument-type]

    def test_union_field_accepts_none(self) -> None:
        class _M(BaseModel):
            out: OutputTarget | None = None

        assert _M().out is None
