# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin function interface — what plugin authors implement for in-process functions.

Plugin authors subclass :class:`NemoFunction` and register the class
under the ``nemo.functions`` entry-point group. The platform mounts a
``POST`` route per function on the plugin service (auto-derived path),
and the CLI exposes ``nemo <plugin> <fn> run`` (in-process) and
``nemo <plugin> <fn> submit`` (HTTP POST) for each. Functions never
dispatch through a backend — they run in the plugin service's event
loop, in the same request that triggered them.

Mental model — *Function = spec → response | stream[frame]*:

- **``spec``** — the inputs (plugin-authored Pydantic model). Validated
  against :attr:`spec_schema` before :meth:`run` is called.
- non-streaming → ``return`` a JSON-serializable value (typically a
  Pydantic model).
- streaming     → declare ``run`` as an ``async def`` returning an
  ``AsyncIterator``/``AsyncGenerator`` and ``yield`` frames. The
  route adapter detects the iterator return type and wraps it in a
  ``StreamingResponse`` with ``application/x-ndjson`` (one
  ``model_dump_json()`` line per frame).

Single class, single method colour. ``run`` is **always**
``async def`` — there is no sync twin and there is no class-level
streaming flag. The only required ``ClassVar`` is :attr:`spec_schema`.

Sync work inside :meth:`run` is a localised escape via
``await asyncio.to_thread(...)`` (or anyio's ``to_thread.run_sync``);
``def run`` is **not accepted** so plugin authors can't accidentally
block the request loop.

Example::

    # my_plugin/functions/greet.py
    from nemo_helix_plugin.function import NemoFunction
    from pydantic import BaseModel

    class GreetSpec(BaseModel):
        name: str

    class GreetResponse(BaseModel):
        message: str

    class GreetFunction(NemoFunction):
        name        = "greet"
        description = "Say hello to a name."
        spec_schema = GreetSpec

        async def run(self, spec: GreetSpec) -> GreetResponse:
            return GreetResponse(message=f"Hello, {spec.name}!")

    # pyproject.toml:
    # [project.entry-points."nemo.functions"]
    # my-plugin.greet = "my_plugin.functions.greet:GreetFunction"

Entry-point key convention: ``<plugin-name>.<function-name>``, e.g.
``example.greet``. The platform's
:func:`~nemo_helix_plugin.discovery.discover_functions` resolves functions
unambiguously across plugins; default-path mounting picks them up at
service startup.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any, ClassVar, Generic, TypeVar

from nemo_helix_plugin._base import _NamedPlugin
from pydantic import BaseModel

SpecT = TypeVar("SpecT", bound=BaseModel)
"""Per-function spec type. Bound to ``BaseModel`` so subclasses get a
typed ``spec`` parameter (`async def run(self, spec: GreetSpec) -> ...`)
without violating Liskov on the abstract base — see :class:`NemoFunction`.
"""


class NemoFunction(_NamedPlugin, Generic[SpecT]):
    """Abstract base class for plugin-contributed in-process functions.

    Subclasses declare their identity via class variables and implement
    :meth:`run`. The platform auto-derives:

    - A CLI subcommand tree: ``nemo <plugin> <fn> run|submit``.
    - A FastAPI route on the plugin service:
      ``POST /apis/<plugin>/v2/workspaces/{workspace}/<name>``
      (override per-class via :attr:`endpoint`).

    Identity:

    .. attribute:: name
        :type: str

        Unique function name within the plugin (e.g. ``"greet"``).
        Combined with the plugin name for the full entry-point key
        (``"my-plugin.greet"``). The class declares only the suffix.

    .. attribute:: description
        :type: str

        Human-readable description. Surfaces in CLI ``--help`` and as
        the route's OpenAPI summary.

    Spec schema:

    .. attribute:: spec_schema
        :type: type[SpecT]

        Pydantic model for the function's inputs. The route adapter
        validates the request body against it before invocation; the
        local ``run`` verb validates the CLI-supplied spec the same
        way. **Required** — the route factory and the CLI both need it
        to generate a working surface. Tied to :data:`SpecT` so a
        subclass like ``NemoFunction[GreetSpec]`` constrains
        ``spec_schema`` to ``type[GreetSpec]`` for type-checkers (and
        IDEs) without runtime cost.

    Endpoint override:

    .. attribute:: endpoint
        :type: str | None

        Override of the **trailing path segment** appended after
        ``/apis/<plugin>/v2/workspaces/{workspace}``. Default is
        ``/{name}``, producing the canonical
        ``POST /apis/<plugin>/v2/workspaces/{workspace}/{name}``
        route. Only the ``{name}`` placeholder is substituted; the
        workspace placeholder stays as a FastAPI route parameter
        because requests carry it per-call. Useful when a plugin
        needs to preserve a legacy suffix (e.g. ``/{name}/v1`` or
        ``/{name}-stream``) — it does **not** let a function
        relocate itself outside its plugin's URL namespace. Leave
        ``None`` to use the default.

    .. attribute:: generate_legacy_verbs
        :type: bool

        Temporary CLI compatibility knob. ``True`` keeps the generated
        ``<function> run|submit`` command group. ``False`` registers
        ``<function>`` itself as the remote submit command and omits the
        local ``run`` and legacy ``submit`` verbs.

    Stream response start:

    .. attribute:: send_headers_before_first_frame
        :type: bool

        For streaming functions, return ``StreamingResponse`` without
        waiting for the first yielded frame. The default ``False`` keeps
        exceptions raised before the first frame in the normal FastAPI
        error path, so validation/setup failures can still become HTTP
        ``4xx``/``5xx`` responses instead of failing after response
        headers have been sent. Enable only for streams that
        intentionally perform long pre-first-frame work and need
        immediate response headers plus heartbeat injection during that
        initial quiet period.

    Lifecycle:

    .. py:method:: run(spec, *, sdk=None, async_sdk=None, ctx=None)
        :async:

        Execute the function. ``spec`` is a validated
        :attr:`spec_schema` instance. Optional keyword-only parameters
        opt into framework-managed dependencies — see
        :meth:`run_signature` for the contract.
    """

    # ------------------------------------------------------------------ #
    # Identity                                                           #
    # ------------------------------------------------------------------ #

    name: ClassVar[str]
    description: ClassVar[str] = ""

    # ------------------------------------------------------------------ #
    # Spec schema (required)                                             #
    # ------------------------------------------------------------------ #

    # ``ClassVar[type[SpecT]]`` ties the declared schema to the same
    # type variable the class is generic over, so a subclass written as
    # ``class GreetFunction(NemoFunction[GreetSpec])`` constrains
    # ``spec_schema`` to ``type[GreetSpec]`` for type-checkers and IDEs.
    # PEP 526 forbids type variables inside ``ClassVar``, so the suppression
    # here is the standard escape hatch for the otherwise-correct shape.
    spec_schema: ClassVar[type[SpecT]]  # ty: ignore[invalid-type-form]

    # ------------------------------------------------------------------ #
    # Endpoint override                                                  #
    # ------------------------------------------------------------------ #

    endpoint: ClassVar[str | None] = None

    generate_legacy_verbs: ClassVar[bool] = True

    send_headers_before_first_frame: ClassVar[bool] = False

    # ------------------------------------------------------------------ #
    # Frame schema (streaming functions only)                            #
    # ------------------------------------------------------------------ #

    # The frame type a streaming function yields, usually a ``kind``-discriminated
    # union. Without it the route documents an empty schema and generated clients
    # hand back ``unknown``, so every consumer re-declares the frames.
    frame_schema: ClassVar[Any] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    # The abstract signature uses ``(*args, **kwargs) -> Any`` —
    # the universal "callable" supertype — so every concrete override
    # is automatically Liskov-compatible regardless of which framework
    # DI parameters it opts into (``ctx`` / ``sdk`` / ``async_sdk``).
    # Three reasons we don't tighten the base:
    #
    # - Subclasses come in two runtime flavours — coroutine functions
    #   (non-streaming) and async generator functions (streaming) —
    #   which Python types differently. A tighter base would make one
    #   shape look incompatible to strict type checkers on every
    #   override.
    # - We still enforce async-ness at class-definition time in
    #   :meth:`__init_subclass__`, so plugin authors can't accidentally
    #   ship a sync ``def run`` that would block the request loop.
    # - The actual contract (``spec`` is the first positional, kwargs
    #   are framework-managed DI) is documented in the method
    #   docstring and in :meth:`run_signature`. Plugin authors get IDE
    #   help from their concrete ``async def run(self, spec: MySpec, ...)``
    #   signature, not from the abstract.
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function and return a value or yield NDJSON frames.

        Functions widen the signature with keyword-only parameters
        (``ctx: FunctionContext``, ``sdk``, ``async_sdk``, ``is_local``) — the route
        adapter and the local CLI both resolve those by parameter name.
        Parameters that aren't resolvable fall back to whatever default
        the signature declares; required parameters with no available
        binding raise at call time, not at import.

        ``run`` **must be ``async def``**. The class enforces this in
        :meth:`__init_subclass__`. Sync work goes through
        ``await asyncio.to_thread(...)``.

        Args:
            spec: Validated :attr:`spec_schema` instance.

        Returns:
            Either a JSON-serialisable value (typically a Pydantic
            model — the route adapter calls ``model_dump_json`` on
            ``BaseModel`` returns) **or** an
            ``AsyncIterator``/``AsyncGenerator`` of frames (typically
            Pydantic models — each frame is emitted as a single NDJSON
            line on the wire). The shape is detected at request time
            from the actual return value, not from a class-level flag.
        """

    # ------------------------------------------------------------------ #
    # Class-level enforcement                                            #
    # ------------------------------------------------------------------ #

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # ``_NamedPlugin.__init_subclass__`` skips intermediate ABCs
        # (anything that still has unimplemented abstract methods); we
        # do the same here so abstract intermediate bases can declare
        # ``run`` without forcing it to be coroutine-typed.
        run_attr = cls.__dict__.get("run")
        if run_attr is None:
            return
        if getattr(run_attr, "__isabstractmethod__", False):
            return
        # Both ``async def`` (coroutine function) and ``async def``
        # with ``yield`` (async generator function) are accepted —
        # streaming functions are async generators, non-streaming ones
        # are coroutines that return a value. Plain ``def`` for either
        # would block the request loop and is rejected here so plugin
        # authors get the error at import, not at request time.
        if not (inspect.iscoroutinefunction(run_attr) or inspect.isasyncgenfunction(run_attr)):
            raise TypeError(
                f"{cls.__qualname__}.run must be `async def` — NemoFunction "
                f"requires an async coroutine or async generator (use "
                f"`await asyncio.to_thread(...)` for sync work). See "
                f"nemo_helix_plugin.function for the contract."
            )

    @classmethod
    def run_signature(cls) -> inspect.Signature:
        """Return :meth:`run`'s :class:`inspect.Signature`.

        Cheap helper used by the route adapter and local CLI to
        discover which framework-managed parameters the function opts
        into (``ctx``, ``sdk``, ``async_sdk``, ``is_local``). Plugin authors don't
        normally call this — it's part of the framework contract.
        """
        return inspect.signature(cls.run)


def returns_async_iterator(value: object) -> bool:
    """Return ``True`` when *value* is an async iterator/generator.

    Used by :func:`nemo_helix_plugin.functions.routes.add_function_routes` to
    branch between ``StreamingResponse`` (NDJSON) and a plain JSON
    response. Kept on the public module so callers (e.g. tests) can
    exercise the same predicate the route uses.

    Recognises both ``AsyncIterator`` (anything with ``__aiter__``) and
    bare async generators.
    """
    return isinstance(value, AsyncIterator) or inspect.isasyncgen(value)


__all__ = ["NemoFunction", "returns_async_iterator"]
