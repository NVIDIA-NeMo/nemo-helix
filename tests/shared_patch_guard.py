# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop tests patching stdlib or third-party modules and classes.

A patch replaces an attribute for the whole process, not just for the test that made it. Patching
``time.sleep``, ``httpx.Client.send`` or ``kubernetes.client.CoreV1Api`` therefore also changes them
for every other thread in the xdist worker, such as an in-process platform's controllers or a thread
left running by an earlier test. Those threads then consume the test's scripted values or spin on a
fake sleep, and the test fails intermittently (#2669, #2853, #3743).

Patch the name where the code under test looks it up instead. Have the module bind what it uses
(``from time import sleep``) and patch ``that_module.sleep``, or replace ``that_module.time`` with a
fake object, or pass the dependency in.

Every patch API ends up in one of three hooks, so these all go through the check:

- ``MonkeyPatch.setattr``/``delattr`` with an object or a dotted string, and ``setitem`` on a
  module's ``__dict__``.
- ``unittest.mock._patch.__enter__``: ``patch`` and ``patch.object`` as decorators, as context
  managers, through ``start()`` or ``ExitStack``, ``patch.multiple``, and pytest-mock's ``mocker``.
- ``unittest.mock._patch_dict._patch_dict``: ``patch.dict`` on a module's ``__dict__``.

The check resolves the object each patch actually lands on, however the target was spelled. A patch
is refused when that object is a module or a class defined outside this repository, and the code
that requested it is in this repository. These are allowed:

- patches of instances, and of first-party modules and classes;
- ``os.environ``, ``sys.argv`` and ``tempfile.tempdir``, and test helpers such as ``typer.testing``;
- setting a name to the value it already holds, or adding a name the target doesn't define;
- patches a third-party tool makes on a test's behalf (``respx``, pytest's own fixtures).

Only unit tests are refused for now. Patches in tests marked ``integration`` or ``e2e`` are reported
but allowed.

Modes, from ``--shared-patch-guard`` or ``NHX_SHARED_PATCH_GUARD``:

- ``strict`` (the default): refuse every such patch.
- ``report``: refuse nothing, and append one JSON line per checked patch to ``--shared-patch-report``.
- ``off``: install nothing.

To list every patch the guard would refuse without failing any test::

    uv run pytest -m unit --shared-patch-guard=report --shared-patch-report=/tmp/patches.jsonl
    jq -r 'select(.decision == "refused") | .key' /tmp/patches.jsonl | sort -u

``@pytest.mark.allow_shared_patch("httpx.Client.send", ...)`` allows the named patches for one test, and
with no arguments allows any.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import sysconfig
import threading
import types
import unittest.mock
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest
from _pytest import monkeypatch as _monkeypatch

MODES = ("off", "report", "strict")

# Process-wide on purpose: environment, arguments and temporary-directory setup.
ALLOWED_TARGETS = frozenset({"os.environ", "sys.argv", "tempfile.tempdir"})

# Test helpers such as ``typer.testing.CliRunner`` or ``starlette.testclient``: only tests call them.
_TEST_HELPER_SEGMENTS = frozenset({"testing", "testclient"})

_UNKNOWN = object()

# Frames inside these modules belong to the patching machinery, not to the code that asked for the patch.
_MACHINERY = ("unittest.mock", "_pytest.monkeypatch", "contextlib", "pytest_mock", __name__)
_STDLIB = Path(sysconfig.get_paths()["stdlib"]).resolve()


class SharedPatchError(AssertionError):
    """A test patched a module or class that the whole process shares."""


@dataclass(frozen=True)
class PatchRecord:
    key: str
    decision: str
    target: str
    kind: str
    origin: str
    api: str
    requester: str
    test: str | None


@dataclass
class _State:
    mode: str = "off"
    root: Path = Path.cwd()
    first_party: frozenset[str] = frozenset()
    report: Path | None = None
    test: str | None = None
    allowed: frozenset[str] | None = frozenset()
    enforce: bool = True


_state = _State()
_local = threading.local()


def _path_origin(path: str | None) -> str:
    if not path:
        return "stdlib"
    resolved = Path(path).resolve()
    if "site-packages" in resolved.parts or "dist-packages" in resolved.parts:
        return "third-party"
    if resolved.is_relative_to(_STDLIB):
        return "stdlib"
    if resolved.is_relative_to(_state.root) and ".venv" not in resolved.relative_to(_state.root).parts:
        return "first-party"
    return "third-party"


def _module_origin(module: types.ModuleType) -> str:
    top = module.__name__.split(".")[0]
    if top in _state.first_party:
        return "first-party"
    if top in sys.stdlib_module_names:
        return "stdlib"
    path = getattr(module, "__file__", None)
    if path is None:
        paths = list(getattr(module, "__path__", []))
        path = paths[0] if paths else None
    return _path_origin(path)


def _describe(target: object) -> tuple[str, str, str]:
    """Return (kind, origin, dotted name) for the object a patch lands on."""
    if isinstance(target, types.ModuleType):
        return "module", _module_origin(target), target.__name__
    if isinstance(target, type):
        module = sys.modules.get(target.__module__)
        origin = _module_origin(module) if module is not None else "stdlib"
        # Private module segments move between Python versions (pathlib._local.Path is pathlib.Path
        # before 3.13), so leave them out to keep names the same across the CI matrix.
        public = ".".join(part for part in target.__module__.split(".") if not part.startswith("_"))
        return "class", origin, f"{public or target.__module__}.{target.__qualname__}"
    return "instance", "", f"<{type(target).__module__}.{type(target).__qualname__} instance>"


def _requester() -> str | None:
    """The file of the code that asked for the patch, skipping the patching machinery.

    A ``@patch`` decorator applies its patch from mock's ``patched`` wrapper, before the decorated
    function runs, so the requester is that function's file.
    """
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module == "unittest.mock" and frame.f_code.co_name == "patched":
            func = frame.f_locals.get("func")
            try:
                return inspect.getsourcefile(inspect.unwrap(func)) if func is not None else None
            except TypeError:
                return None
        if not any(module == name or module.startswith(f"{name}.") for name in _MACHINERY):
            return frame.f_code.co_filename
        frame = frame.f_back
    return None


def _check(api: str, target: object, attribute: str | None, new: object = _UNKNOWN) -> None:
    if _state.mode == "off" or getattr(_local, "busy", False):
        return
    kind, origin, name = _describe(target)
    if kind == "instance" or origin == "first-party":
        return
    dotted = f"{name}.{attribute}" if attribute else name
    if dotted in ALLOWED_TARGETS or _TEST_HELPER_SEGMENTS.intersection(name.split(".")):
        return
    if attribute is not None:
        current = inspect.getattr_static(target, attribute, _UNKNOWN)
        # Setting a name to what it already holds, or adding one the target doesn't have, changes
        # nothing other code can see.
        if current is _UNKNOWN or current is new:
            return
    requester = _requester()
    if requester is None or _path_origin(requester) != "first-party":
        decision = "tool"
    else:
        relative = Path(requester).resolve().relative_to(_state.root).as_posix()
        key = f"{relative} {dotted}"
        allowed = _state.allowed is None or dotted in _state.allowed
        if allowed:
            decision = "marker"
        else:
            decision = "refused"
    record = PatchRecord(
        key=key if decision != "tool" else f"{requester} {dotted}",
        decision=decision,
        target=dotted,
        kind=kind,
        origin=origin,
        api=api,
        requester=requester or "",
        test=_state.test,
    )
    if _state.report is not None:
        with _state.report.open("a") as report:
            report.write(json.dumps(asdict(record)) + "\n")
    if decision == "refused" and _state.enforce and _state.mode == "strict":
        raise SharedPatchError(
            f"{api} patches {dotted}, which every thread in the test process shares, so other threads "
            f"see the fake too. Patch the name where the code under test looks it up instead: bind it "
            f"in that module (`from {name} import {attribute}`) and patch `that_module.{attribute}`, "
            f"replace the module reference there with a fake, or pass the dependency in. "
            f"See tests/shared_patch_guard.py."
        )


@contextmanager
def _busy() -> Generator[None]:
    """Don't check patches the guard's own resolution triggers."""
    previous = getattr(_local, "busy", False)
    _local.busy = True
    try:
        yield
    finally:
        _local.busy = previous


def _resolve_dotted(path: str) -> tuple[object, str] | None:
    try:
        with _busy():
            attribute, target = _monkeypatch.derive_importpath(path, True)
    except Exception:
        return None
    return target, attribute


def _module_of_dict(mapping: object) -> types.ModuleType | None:
    if not isinstance(mapping, dict) or "__spec__" not in mapping:
        return None
    module = sys.modules.get(mapping.get("__name__", ""))
    return module if module is not None and module.__dict__ is mapping else None


_original_setattr = _monkeypatch.MonkeyPatch.setattr
_original_delattr = _monkeypatch.MonkeyPatch.delattr
_original_setitem = _monkeypatch.MonkeyPatch.setitem
_original_enter = unittest.mock._patch.__enter__
# Private, and absent from the stubs; the guard's own tests fail if a Python release moves it.
_original_patch_dict = getattr(unittest.mock._patch_dict, "_patch_dict")


def _guarded_setattr(self: pytest.MonkeyPatch, target: Any, *args: Any, **kwargs: Any) -> None:
    if isinstance(target, str):
        resolved = _resolve_dotted(target)
        if resolved is not None:
            # With a dotted string, the second argument (`name`) is the new value.
            _check("monkeypatch.setattr", *resolved, args[0] if args else kwargs.get("name", _UNKNOWN))
    else:
        name = args[0] if args else kwargs.get("name")
        if isinstance(name, str):
            _check("monkeypatch.setattr", target, name, args[1] if len(args) > 1 else kwargs.get("value", _UNKNOWN))
    return _original_setattr(self, target, *args, **kwargs)


def _guarded_delattr(self: pytest.MonkeyPatch, target: Any, *args: Any, **kwargs: Any) -> None:
    if isinstance(target, str):
        resolved = _resolve_dotted(target)
        if resolved is not None:
            _check("monkeypatch.delattr", *resolved)
    else:
        name = args[0] if args else kwargs.get("name")
        if isinstance(name, str):
            _check("monkeypatch.delattr", target, name)
    return _original_delattr(self, target, *args, **kwargs)


def _guarded_setitem(self: pytest.MonkeyPatch, dic: Any, name: Any, value: Any) -> None:
    module = _module_of_dict(dic)
    if module is not None and isinstance(name, str):
        _check("monkeypatch.setitem", module, name, value)
    return _original_setitem(self, dic, name, value)


def _guarded_enter(self: Any) -> Any:
    try:
        with _busy():
            target = self.getter()
    except Exception:
        target = None
    if target is not None:
        _check("mock.patch", target, self.attribute, self.new)
    return _original_enter(self)


def _guarded_patch_dict(self: Any) -> Any:
    in_dict = self.in_dict
    if isinstance(in_dict, str):
        import pkgutil

        try:
            with _busy():
                in_dict = pkgutil.resolve_name(in_dict)
        except Exception:
            in_dict = None
    module = _module_of_dict(in_dict)
    if module is not None:
        _check("mock.patch.dict", module, None)
    return _original_patch_dict(self)


def _install() -> None:
    setattr(_monkeypatch.MonkeyPatch, "setattr", _guarded_setattr)
    setattr(_monkeypatch.MonkeyPatch, "delattr", _guarded_delattr)
    setattr(_monkeypatch.MonkeyPatch, "setitem", _guarded_setitem)
    setattr(unittest.mock._patch, "__enter__", _guarded_enter)
    setattr(unittest.mock._patch_dict, "_patch_dict", _guarded_patch_dict)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "shared_patch_first_party",
        "Top-level packages from this repository that are installed outside it, such as a wheel under test.",
        type="args",
        default=[],
    )
    group = parser.getgroup("shared-patch-guard")
    group.addoption(
        "--shared-patch-guard",
        choices=MODES,
        default=None,
        help="Refuse patches of stdlib or third-party modules and classes (default: strict).",
    )
    group.addoption("--shared-patch-report", default=None, help="Append one JSON line per checked patch here.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "allow_shared_patch(*targets): allow patching these shared names, e.g. 'time.sleep'; none = any"
    )
    mode = config.getoption("--shared-patch-guard") or os.environ.get("NHX_SHARED_PATCH_GUARD", "strict")
    if mode not in MODES:
        raise pytest.UsageError(f"--shared-patch-guard must be one of {', '.join(MODES)}, not {mode!r}")
    report = config.getoption("--shared-patch-report")
    _state.mode = mode
    _state.root = config.rootpath.resolve()
    _state.first_party = frozenset(config.getini("shared_patch_first_party"))
    _state.report = Path(report) if report else None
    if mode != "off":
        _install()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    marker = item.get_closest_marker("allow_shared_patch")
    _state.test = item.nodeid
    _state.allowed = (frozenset(marker.args) if marker.args else None) if marker else frozenset()
    _state.enforce = item.get_closest_marker("integration") is None and item.get_closest_marker("e2e") is None
    try:
        return (yield)
    finally:
        _state.test = None
        _state.allowed = frozenset()
        _state.enforce = True
