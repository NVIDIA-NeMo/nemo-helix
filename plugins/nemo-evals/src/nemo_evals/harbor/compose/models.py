# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What reading a compose file raises: its error, and the collector of problems that becomes it."""

from __future__ import annotations


class ComposeError(ValueError):
    """A task whose compose file can't run as services in a sandbox. ``code`` is stable and shown to users."""

    def __init__(self, code: str, detail: str, *, compose_path: str | None = None, paths: tuple[str, ...] = ()) -> None:
        """Build the message ``<code>: <detail> (<paths>) in <compose_path>``, and keep each part on the error."""
        message = f"{code}: {detail}"
        if paths:
            message += f" ({', '.join(paths)})"
        if compose_path:
            message += f" in {compose_path}"
        super().__init__(message)

        self.code = code
        self.detail = detail
        self.compose_path = compose_path
        self.paths = paths


class ComposeProblems:
    """Everything wrong with one compose file, collected while it is parsed and raised as one :class:`ComposeError`.

    Collecting rather than raising at the first problem lets an author fix everything in one pass. A problem is
    *unsupported* (valid Compose that a pod can't run) or *invalid* (something Compose itself would refuse).
    """

    def __init__(self) -> None:
        """Start with nothing found."""
        self._unsupported: list[str] = []
        self._invalid: list[str] = []

    def unsupported(self, path: str) -> None:
        """Record a feature at ``path`` that Compose has and a pod doesn't."""
        self._unsupported.append(path)

    def invalid(self, path: str, why: str) -> None:
        """Record a value at ``path`` that Compose itself would refuse, and why."""
        self._invalid.append(f"{path}: {why}")

    def raise_if_any(self, compose_path: str) -> None:
        """Raise one error for everything recorded; do nothing when nothing was.

        Any invalid value makes the error ``compose_invalid``, since the file is broken whatever runs it. Unsupported
        features found alongside are still listed in its paths, so the author sees every problem in one pass.
        """
        if not self._invalid and not self._unsupported:
            return

        unsupported = tuple(dict.fromkeys(self._unsupported))
        if not self._invalid:
            raise ComposeError(
                "compose_unsupported",
                "these can't run as sandbox services",
                compose_path=compose_path,
                paths=unsupported,
            )

        detail = "; ".join(self._invalid)
        if unsupported:
            detail += "; and these can't run as sandbox services"
        raise ComposeError("compose_invalid", detail, compose_path=compose_path, paths=unsupported)
