# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loads ``nemo_opensandbox_ext`` into a stock OpenSandbox server.

The extension ships as plain source files in a ConfigMap, mounted into the stock server
image, so there is no custom image to build. What's needed is a way to run our code inside
the server process at the right moment. Python imports ``sitecustomize`` at startup when its
directory is on ``PYTHONPATH``, which the server's Helm values set.

This module only installs an import hook, so other Python processes in the image are
unaffected. When the server imports ``opensandbox_server.main``, the hook:

1. registers the services provider, before that module builds the sandbox service (which
   is when the server picks its provider);
2. runs the module;
3. adds the extension's routes to the FastAPI app the module created.

A failure in any step stops the server, instead of starting it without the extension.

The extension hooks into server internals, so it only loads on ``TESTED_SERVER_VERSIONS``.
On any other version the server starts without it and logs why; clients see that through
the extension's health route. To support a new server version, bump the ``opensandbox-server``
pin in the root ``pyproject.toml``, run the extension tests, and add the version here.

Putting this directory first on ``PYTHONPATH`` shadows any other ``sitecustomize`` in the image.
"""

import importlib.abc
import importlib.machinery
import importlib.metadata
import logging
import sys

# The server module the hook wraps: it builds the provider and the app when imported.
TARGET_MODULE = "opensandbox_server.main"

# Server versions the extension's tests run against; see the module docstring.
TESTED_SERVER_VERSIONS = frozenset({"0.2.1"})


def server_version() -> str | None:
    """The installed opensandbox-server version, or None if the package metadata is missing."""
    try:
        return importlib.metadata.version("opensandbox-server")
    except importlib.metadata.PackageNotFoundError:
        return None


class _Loader(importlib.abc.Loader):
    """Wraps the loader of ``opensandbox_server.main`` to install the extension around its execution."""

    def __init__(self, inner: importlib.abc.Loader) -> None:
        """Wrap the loader Python would have used."""
        self._inner = inner

    def create_module(self, spec):
        """Delegate module creation unchanged."""
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        """Register the provider, run the server's main module, then add the routes to the app it built.

        On an untested server version, only runs the main module, so the server starts unchanged.
        """
        version = server_version()

        if version not in TESTED_SERVER_VERSIONS:
            self._inner.exec_module(module)

            # Logged after main ran, so the server's logging config is in place and the line shows up.
            logging.getLogger("opensandbox_server.nemo_ext").warning(
                "nemo services: extension not loaded; opensandbox-server %s is not a tested version (%s)",
                version or "<unknown>",
                ", ".join(sorted(TESTED_SERVER_VERSIONS)),
            )
            return

        # Imported here so the extension (and the server it imports) only load in the server process.
        import nemo_opensandbox_ext

        nemo_opensandbox_ext.register_provider()
        self._inner.exec_module(module)
        nemo_opensandbox_ext.attach_routes(module.app)


class _Finder(importlib.abc.MetaPathFinder):
    """Finds ``opensandbox_server.main`` as usual, but hands it the wrapping ``_Loader``."""

    def find_spec(self, fullname, path, target=None):
        """Return a wrapped spec for the server's main module; None (no opinion) for every other module."""
        if fullname != TARGET_MODULE:
            return None

        # Let the normal path-based finder locate the module, then swap in our loader.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


# Install the hook once, ahead of the standard finders, even if this module is imported twice.
if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
