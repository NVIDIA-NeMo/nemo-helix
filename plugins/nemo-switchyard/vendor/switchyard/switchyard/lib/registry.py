# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``MiddlewareFactory`` registry — typed config -> request/response pipelines.

A ``MiddlewareFactory`` bridges declarative config (an entity-store
payload, a NeMo Helix VirtualModel descriptor, a CLI flag bundle) to a pair of
:class:`RequestPipeline` / :class:`ResponsePipeline`. Hosts that own
their own LLM backend — NeMo Helix IGW today, possibly Studio / Evaluator
later — look up a factory by name, validate their raw config through it,
and slot the resulting pipelines around their backend.

Factories are deliberately host-agnostic: they live in
``switchyard/lib/factories/``, they don't know about IGW or NeMo Helix, and
the same registry is usable from the Switchyard CLI / standalone
server. NeMo Helix-specific concerns (IGW ``ModelProvider`` resolution,
request-id keying, etc.) belong to the bridge layer, not here.

Discovery uses the ``switchyard.middlewares`` entry-point group.
Each entry-point is a module path; importing the module triggers a
``register(...)`` side-effect inside the module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from switchyard.lib.request_pipeline import RequestPipeline
    from switchyard.lib.response_pipeline import ResponsePipeline
    from switchyard.lib.roles import LLMBackend, ResponseTranslator
    from switchyard.lib.switchyard import Switchyard


# ``ConfigT`` is contravariant in the part-builders (the factory consumes a
# config) and covariant in ``validate`` (the factory produces a config).
# Pyright / ty accept this when ConfigT is bound to ``BaseModel`` and used
# at both call sites with the same factory's concrete subclass.
ConfigT = TypeVar("ConfigT", bound="BaseModel")


@runtime_checkable
class MiddlewareFactory(Protocol[ConfigT]):
    """Granular builder protocol — typed config to pipeline parts.

    The protocol is split into four part-builders so consumers only pay
    for what they use:

    * NeMo Helix IGW supplies its own backend — it calls
      :meth:`build_request_pipeline` / :meth:`build_response_pipeline`
      and never instantiates an LLM client.
    * Standalone Switchyard (CLI, recipes) calls
      :meth:`MiddlewareBundle.from_factory` which invokes all four
      part-builders to assemble a full chain.

    Implementations should:

    * Set ``name`` to a unique string — the registry key.
    * Set ``config_class`` to the pydantic model that describes their
      configuration.
    * Implement ``validate(raw)`` to coerce a dict / pre-validated model
      into the typed config (raises ``pydantic.ValidationError`` on
      malformed input).
    * Implement :meth:`build_request_pipeline` and
      :meth:`build_response_pipeline` — every factory ships processors
      (the pipeline may be empty).
    * Optionally implement :meth:`build_backend` and
      :meth:`build_translator` — return ``None`` when the host owns
      these slots. :class:`BaseMiddlewareFactory` provides the
      ``return None`` defaults so pure middleware factories don't need
      to write them by hand.
    """

    name: ClassVar[str]
    config_class: ClassVar[type[BaseModel]]

    def validate(self, raw: Any) -> ConfigT: ...

    def build_request_pipeline(self, config: ConfigT) -> RequestPipeline: ...

    def build_response_pipeline(self, config: ConfigT) -> ResponsePipeline: ...

    def build_backend(self, config: ConfigT) -> LLMBackend | None: ...

    def build_translator(self, config: ConfigT) -> ResponseTranslator | None: ...


class BaseMiddlewareFactory(Generic[ConfigT]):
    """Default implementations for the optional part-builders.

    Concrete factories that ship a full chain override
    :meth:`build_backend` / :meth:`build_translator`; pure middleware
    factories (RouteLLM-as-processor, FormatTranslate, future
    StatsFactory) inherit the ``return None`` defaults and only
    implement the two pipeline builders.

    Subclasses must still set ``name`` / ``config_class`` and implement
    ``validate``, ``build_request_pipeline``, ``build_response_pipeline``.
    Not a ``Protocol`` — instances are concrete and the registry checks
    structural conformance via :class:`MiddlewareFactory`.
    """

    def build_backend(self, config: ConfigT) -> LLMBackend | None:
        return None

    def build_translator(self, config: ConfigT) -> ResponseTranslator | None:
        return None


_FACTORIES: dict[str, MiddlewareFactory[Any]] = {}
_SWITCHYARDS: dict[str, Switchyard] = {}
_DEFAULT_SWITCHYARD: Switchyard | None = None


def register(factory: MiddlewareFactory[Any], *, replace: bool = False) -> None:
    """Register a factory under ``factory.name``.

    Args:
        factory: A :class:`MiddlewareFactory` instance.
        replace: When ``True``, silently overwrite an existing
            registration. When ``False`` (default), a duplicate name
            raises ``ValueError``. ``replace=True`` is intended for tests
            that swap a factory in / out; production code should never
            need it.

    Raises:
        TypeError: ``factory`` does not satisfy the
            :class:`MiddlewareFactory` protocol.
        ValueError: another factory is already registered under
            ``factory.name`` and ``replace`` is ``False``.
    """
    if not isinstance(factory, MiddlewareFactory):
        raise TypeError(
            f"register() expected a MiddlewareFactory, got {type(factory).__name__}"
        )
    name = factory.name
    if not replace and name in _FACTORIES:
        raise ValueError(
            f"MiddlewareFactory name {name!r} is already registered "
            f"(by {type(_FACTORIES[name]).__name__})"
        )
    _FACTORIES[name] = factory


def unregister(name: str) -> None:
    """Drop the factory registered under ``name``. No-op if absent.

    Used primarily by tests that swap factories in and out so they don't
    leak registrations across cases.
    """
    _FACTORIES.pop(name, None)


def lookup(name: str) -> MiddlewareFactory[Any]:
    """Return the factory registered under ``name``.

    Raises:
        KeyError: no factory registered under ``name``. The error
            message lists the currently registered names so the caller
            can spot typos.
    """
    try:
        return _FACTORIES[name]
    except KeyError:
        raise KeyError(
            f"No MiddlewareFactory registered under {name!r}; "
            f"available: {sorted(_FACTORIES)}"
        ) from None


def registered_names() -> list[str]:
    """Return all currently registered factory names, sorted."""
    return sorted(_FACTORIES)


def discover() -> list[str]:
    """Load every entry-point in the ``switchyard.middlewares`` group.

    Each entry-point is a module path; importing the module is expected
    to call ``register(...)`` at module load time (see
    ``factories/passthrough.py`` for the canonical pattern).

    Returns:
        Names of all currently-registered factories after discovery,
        sorted. Useful for tests and CLI ``--list-middlewares``-style
        outputs.
    """
    from importlib.metadata import entry_points

    for ep in entry_points(group="switchyard.middlewares"):
        ep.load()
    return registered_names()


# ============================================================================
# Switchyard Registry — model name → pre-built chains
# ============================================================================


def register_switchyard(model: str, switchyard: Switchyard) -> None:
    """Register a Switchyard instance for a model name.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-opus-4-6").
        switchyard: Pre-built Switchyard instance.

    Raises:
        ValueError: model is already registered.
    """
    if model in _SWITCHYARDS:
        raise ValueError(
            f"Switchyard for model {model!r} is already registered"
        )
    _SWITCHYARDS[model] = switchyard


def lookup_switchyard(model: str) -> Switchyard:
    """Retrieve the Switchyard for a model, with fallback to default.

    Args:
        model: Model identifier.

    Returns:
        The registered Switchyard for this model, or the default if set.

    Raises:
        KeyError: model is not registered and no default is set.
    """
    if model in _SWITCHYARDS:
        return _SWITCHYARDS[model]
    if _DEFAULT_SWITCHYARD is not None:
        return _DEFAULT_SWITCHYARD
    raise KeyError(
        f"No Switchyard registered for model {model!r}; "
        f"available models: {sorted(_SWITCHYARDS.keys())}. "
        f"Set a default with set_default_switchyard()."
    )


def set_default_switchyard(switchyard: Switchyard | None) -> None:
    """Set the default Switchyard for unregistered models.

    Args:
        switchyard: Switchyard instance, or None to clear the default.
    """
    global _DEFAULT_SWITCHYARD
    _DEFAULT_SWITCHYARD = switchyard


def registered_models() -> list[str]:
    """Return all currently registered model names, sorted."""
    return sorted(_SWITCHYARDS.keys())


def unregister_switchyard(model: str) -> None:
    """Drop the Switchyard registered for ``model``. No-op if absent.

    Used primarily by tests.
    """
    _SWITCHYARDS.pop(model, None)


def clear_switchyards() -> None:
    """Clear all registered Switchyard instances and the default.

    Used primarily by tests.
    """
    global _DEFAULT_SWITCHYARD
    _SWITCHYARDS.clear()
    _DEFAULT_SWITCHYARD = None
