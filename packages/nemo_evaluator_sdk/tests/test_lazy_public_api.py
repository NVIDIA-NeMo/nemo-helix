# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards the lazy (PEP 562) public surfaces of the evaluator SDK.

Two barrels re-export lazily, and importing any submodule runs both in turn, so a single
convenience ``from … import …`` added at either module scope silently re-drags the whole
backend/benchmark/metric stack into every consumer that only wanted ``agent_eval``:

* ``nemo_evaluator_sdk/__init__.py`` (AALGO-429) — the execution/backend and metric stack.
* ``nemo_evaluator_sdk/values/__init__.py`` (AALGO-311) — pyarrow, numpy, jinja2 and jsonschema,
  together with the deferred pyarrow import in ``values/results.py``.

Both are covered here in their source form and in the ``nemo_platform.beta.evaluator`` mirror the
vendoring tool generates.

Every *assertion* runs out-of-process. Resolving a whole public surface imports openai, sacrebleu,
ragas and the execution stack; doing that in-process would leave them in ``sys.modules`` for every
test that runs after it in the session, so a future in-process "module X must not be imported"
check — the natural way someone would extend this file — would depend on collection order. The one
in-process call is the ``find_spec`` availability probe below, which is deliberately restricted to
a top-level name so that it imports nothing.
"""

import asyncio
import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_VENDORED_MIRROR = "nemo_platform.beta.evaluator"

# Imported out-of-process on purpose: by the time this module runs under pytest, sibling suites
# have already pulled the execution stack into sys.modules, so an in-process check proves nothing.
_IMPORT_SURFACE_PROBE = """
import json, sys
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner
assert HarborAgentTaskRunner is not None
print(json.dumps(sorted(sys.modules)))
"""

# Resolves every declared name for real. This is also the only thing left that validates each
# re-exported submodule still imports at all: before the package went lazy, `import
# nemo_evaluator_sdk` executed all of them, so a syntax error or a broken third-party import
# failed immediately everywhere. Narrowing this loop, skipping this test, or dropping a name from
# `__all__` silently gives that guarantee up.
_PUBLIC_SURFACE_PROBE = """
import json, sys

name, submodule_name = sys.argv[1], sys.argv[2]
module = __import__(name, fromlist=["__all__"])

for attribute in module.__all__:      # AttributeError/ImportError names the offender in stderr
    getattr(module, attribute)

try:
    module.NoSuchName
except AttributeError:
    pass
else:
    raise AssertionError("unknown attribute did not raise AttributeError")

# The `from pkg import submodule` fallback, which only fires when __getattr__ raises
# AttributeError rather than KeyError or ImportError.
submodule = getattr(__import__(name, fromlist=[submodule_name]), submodule_name)
assert submodule.__name__ == f"{name}.{submodule_name}", submodule.__name__

print(json.dumps({
    "resolved": len(module.__all__),
    "missing_from_dir": sorted(set(module.__all__) - set(dir(module))),
}))
"""


def _is_harbor_module(name: str) -> bool:
    return name == "harbor" or name.startswith("harbor.")


def _block_harbor_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_harbor_import(name, *args, **kwargs):
        if _is_harbor_module(name):
            raise ModuleNotFoundError("blocked Harbor import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_harbor_import)


def test_agent_eval_import_does_not_pull_the_execution_stack() -> None:
    """The optimizer imports only ``agent_eval``; it must not pay for backends and benchmarks.

    Beyond start-up cost, every module loaded here is a package whose import failure becomes an
    SDK-path failure at evaluation time. Keeping the boundary tight is what lets the consumer's
    deferred SDK import actually contain a broken install.
    """
    proc = subprocess.run([sys.executable, "-c", _IMPORT_SURFACE_PROBE], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr

    # Last line only: the child inherits the environment, so a sitecustomize banner or an
    # import-time print() would otherwise turn this into an opaque JSONDecodeError.
    modules = set(json.loads(proc.stdout.strip().splitlines()[-1]))

    leaked = sorted(name for name in modules if name.startswith("nemo_evaluator_sdk.execution"))
    assert leaked == [], f"the package __init__ re-drags the execution stack: {leaked}"

    # Every one of these was on this path before the two barrels went lazy, so each goes red if it
    # returns: openai/sacrebleu/zstandard came from the root __init__, and pyarrow/numpy/jinja2/
    # jsonschema from the values/ barrel plus values/results.py's module-scope pyarrow import.
    # rouge_score is deliberately absent: metrics/rouge.py already defers it into a
    # cached_property, so it was never on this path and asserting it would prove nothing.
    heavy = {"openai", "sacrebleu", "zstandard", "pyarrow", "numpy", "jinja2", "jsonschema"} & modules
    assert heavy == set(), f"heavy dependencies pulled into the agent_eval path: {sorted(heavy)}"
    harbor_modules = sorted(name for name in modules if _is_harbor_module(name))
    assert harbor_modules == [], f"the lazy Harbor result validator was imported eagerly: {harbor_modules}"

    # A canary, not a spec: measured at 300 modules once both barrels went lazy, down from 1416.
    # The bound is deliberately close — the assertions above enumerate known offenders, so only
    # this catches a re-drag through some other route. Raise it only for a dependency agent_eval
    # genuinely needs, and say which in the commit message.
    assert len(modules) < 380, f"agent_eval import surface grew to {len(modules)} modules"


def test_harbor_adapter_invocation_without_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import build_trials_from_job_dir

    _block_harbor_import(monkeypatch)
    with pytest.raises(ModuleNotFoundError, match=r"optional `harbor` extra on Python >=3\.12") as exc_info:
        build_trials_from_job_dir(".", [])

    assert "repository root" in str(exc_info.value)
    assert "uv sync --frozen --package nemo-evaluator-sdk --extra harbor" in str(exc_info.value)


def test_harbor_execution_without_extra_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborRuntimeConfig, _build_native_job

    _block_harbor_import(monkeypatch)
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs")
    _job_dir, run_job = _build_native_job(config, tmp_path / "dataset", None)

    async def invoke_run_job() -> None:
        await run_job()

    with pytest.raises(ModuleNotFoundError, match=r"optional `harbor` extra on Python >=3\.12") as exc_info:
        asyncio.run(invoke_run_job())

    assert "repository root" in str(exc_info.value)
    assert "uv sync --frozen --package nemo-evaluator-sdk --extra harbor" in str(exc_info.value)


@pytest.mark.parametrize(
    ("module_name", "submodule_name"),
    [
        ("nemo_evaluator_sdk", "values"),
        ("nemo_evaluator_sdk.values", "models"),
        (_VENDORED_MIRROR, "values"),
        (f"{_VENDORED_MIRROR}.values", "models"),
    ],
)
def test_every_public_name_resolves(module_name: str, submodule_name: str) -> None:
    """``__all__`` and ``_LAZY_ATTRS`` must not drift apart, in the source or the vendored mirror.

    A typo in the lazy table is invisible until a consumer hits that one attribute, so resolve
    the whole surface in one pass.

    The mirror leg is what keeps the relative module paths in ``_LAZY_ATTRS`` honest. They are
    relative precisely so the vendoring tool has nothing to rewrite; if someone switches them to
    an f-string (which the rewriter does not touch) or to absolute literals under a changed
    rewriter, the mirror resolves to the wrong package. Existing tests import *through* the mirror
    but never resolve its surface, so nothing else would notice.
    """
    # Probe the top-level package only. `find_spec` on a dotted name RAISES ModuleNotFoundError
    # when a parent is missing rather than returning None, so probing the full path would error
    # in exactly the case this guard exists for (no nemo-platform installed). It also imports
    # every parent in-process, which would defeat this module's isolation.
    root_package = module_name.partition(".")[0]
    if importlib.util.find_spec(root_package) is None:
        pytest.skip(f"{root_package} is not installed in this environment")

    proc = subprocess.run(
        [sys.executable, "-c", _PUBLIC_SURFACE_PROBE, module_name, submodule_name],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["missing_from_dir"] == [], f"__all__ names absent from dir(): {result['missing_from_dir']}"
    assert result["resolved"] > 0

    # `version` is deliberately not asserted here: this workspace installs nemo-evaluator-sdk
    # itself, and its declared version is literally "0.0.0", so the distribution-fallback in
    # _resolve_version() only changes behaviour in a built wheel, where the SDK is absent.
