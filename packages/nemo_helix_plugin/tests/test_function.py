# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Class-level tests for :class:`nemo_helix_plugin.function.NemoFunction`.

Covers:

- :class:`_NamedPlugin` identity enforcement (empty / missing ``name``).
- ``run`` must be ``async def`` — a sync ``def run`` raises at class
  definition time so plugin authors get the error during import, not
  at request time.
- Instantiation produces a runnable object whose ``run`` method
  returns either a value or an async iterator (the route adapter
  branches on this in C2; here we just validate the predicate).
- The optional ``endpoint`` override, ``description`` ClassVar, and
  streaming response-start flag default to the documented values.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from nemo_helix_plugin.function import NemoFunction, returns_async_iterator
from pydantic import BaseModel


class GreetSpec(BaseModel):
    name: str


class GreetResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Identity / class-level enforcement
# ---------------------------------------------------------------------------


class TestNemoFunctionIdentity:
    def test_concrete_subclass_requires_name(self) -> None:
        with pytest.raises(TypeError, match="non-empty string class variable 'name'"):

            class _Bad(NemoFunction[GreetSpec]):  # noqa: N801 — local-only
                spec_schema = GreetSpec

                async def run(self, spec: GreetSpec) -> dict:
                    return {"name": spec.name}

    def test_empty_name_string_rejected(self) -> None:
        with pytest.raises(TypeError, match="non-empty string class variable 'name'"):

            class _Empty(NemoFunction[GreetSpec]):  # noqa: N801 — local-only
                name = ""
                spec_schema = GreetSpec

                async def run(self, spec: GreetSpec) -> dict:
                    return {"name": spec.name}

    def test_description_defaults_to_empty(self) -> None:
        class _Greet(NemoFunction[GreetSpec]):
            name = "greet"
            spec_schema = GreetSpec

            async def run(self, spec: GreetSpec) -> dict:
                return {"name": spec.name}

        assert _Greet.description == ""
        assert _Greet.endpoint is None
        assert _Greet.send_headers_before_first_frame is False

    def test_endpoint_override_passes_through(self) -> None:
        class _Greet(NemoFunction[GreetSpec]):
            name = "greet"
            description = "Greet a name."
            spec_schema = GreetSpec
            endpoint = "/apis/example/v2/workspaces/{workspace}/legacy-greet"

            async def run(self, spec: GreetSpec) -> dict:
                return {"name": spec.name}

        assert _Greet.endpoint == "/apis/example/v2/workspaces/{workspace}/legacy-greet"
        assert _Greet.description == "Greet a name."


# ---------------------------------------------------------------------------
# `run` must be async
# ---------------------------------------------------------------------------


class TestRunMustBeAsync:
    def test_sync_def_run_rejected_at_class_definition(self) -> None:
        with pytest.raises(TypeError, match="must be `async def`"):

            class _Bad(NemoFunction[GreetSpec]):  # noqa: N801 — local-only
                name = "bad"
                spec_schema = GreetSpec

                def run(self, spec: GreetSpec) -> dict:
                    return {"name": spec.name}

    def test_async_def_run_accepted(self) -> None:
        class _Good(NemoFunction[GreetSpec]):
            name = "good"
            spec_schema = GreetSpec

            async def run(self, spec: GreetSpec) -> dict:
                return {"name": spec.name}

        assert _Good is not None  # class definition must not raise


# ---------------------------------------------------------------------------
# Streaming detection — predicate used by the route adapter
# ---------------------------------------------------------------------------


class TestStreamingDetection:
    @pytest.mark.asyncio
    async def test_returning_value_is_not_iterator(self) -> None:
        class _Sync(NemoFunction[GreetSpec]):
            name = "sync"
            spec_schema = GreetSpec

            async def run(self, spec: GreetSpec) -> GreetResponse:
                return GreetResponse(message=f"Hello, {spec.name}!")

        result = await _Sync().run(GreetSpec(name="world"))
        assert not returns_async_iterator(result)
        assert isinstance(result, GreetResponse)
        assert result.message == "Hello, world!"

    @pytest.mark.asyncio
    async def test_async_generator_is_iterator(self) -> None:
        class _Stream(NemoFunction[GreetSpec]):
            name = "stream"
            spec_schema = GreetSpec

            async def run(self, spec: GreetSpec) -> AsyncIterator[GreetResponse]:
                yield GreetResponse(message=f"Hello, {spec.name}!")
                yield GreetResponse(message="Goodbye!")

        gen = _Stream().run(GreetSpec(name="world"))
        assert returns_async_iterator(gen)
        collected = [m.message async for m in gen]
        assert collected == ["Hello, world!", "Goodbye!"]


# ---------------------------------------------------------------------------
# Signature introspection — opt-in DI parameters
# ---------------------------------------------------------------------------


class TestRunSignature:
    def test_signature_exposes_kwonly_di_parameters(self) -> None:
        class _Greet(NemoFunction[GreetSpec]):
            name = "greet"
            spec_schema = GreetSpec

            async def run(self, spec: GreetSpec, *, sdk: object | None = None) -> dict:
                return {"name": spec.name, "have_sdk": sdk is not None}

        sig = _Greet.run_signature()
        # ``self`` + ``spec`` + ``sdk``.
        assert list(sig.parameters) == ["self", "spec", "sdk"]
        assert sig.parameters["sdk"].default is None

    def test_signature_minimal_when_no_di(self) -> None:
        class _Greet(NemoFunction[GreetSpec]):
            name = "greet"
            spec_schema = GreetSpec

            async def run(self, spec: GreetSpec) -> dict:
                return {"name": spec.name}

        assert list(_Greet.run_signature().parameters) == ["self", "spec"]
