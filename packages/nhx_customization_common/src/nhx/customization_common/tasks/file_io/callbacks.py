# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom fsspec callbacks for progress reporting during file I/O operations."""

import logging
import threading
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fsspec.callbacks import Callback, TqdmCallback
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.customization_common.schemas.file_io import DownloadStats, TaskPhase, UploadStats
from nhx.customization_common.tasks.file_io_progress_reporter import ProgressReporter
from nhx.customization_common.tasks.file_io_utils import list_local_files as _list_local_files

logger = logging.getLogger(__name__)


def get_percentage(current: int, total: int) -> int:
    """Get integer percentage 0-100, clamped to the valid range.

    Progress accounting must never abort the underlying transfer. Inputs
    can fall outside ``[0, total]`` for benign reasons — most commonly when
    the pre-transfer file listing under-counts a source that contains nested
    directories, so the live ``current`` count exceeds ``total`` by the
    number of nested files. Clamp rather than raise so a cosmetic progress
    number can't fail a multi-GB download.
    """
    if total <= 0:
        return 0
    if current > total or current < 0:
        logger.debug("get_percentage clamping out-of-range progress: current=%s total=%s", current, total)
    current = max(0, min(current, total))
    return int((current / total) * 100)


@dataclass
class FileInfo:
    """A dataclass for file information."""

    path: str
    size: int


class TqdmPerFileUploadCallback(Callback):
    """A callback that creates a separate tqdm progress bar for each file upload."""

    def __init__(self, src_path: Path, **kwargs: Any):
        self.src_path = src_path
        super().__init__(**kwargs)

    def branched(self, full_src_path: str, full_dest_path: str, **kwargs: Any) -> TqdmCallback:
        if self.src_path.is_file():
            relative_path = self.src_path.name
        else:
            relative_path = Path(full_src_path).relative_to(self.src_path)
        return TqdmCallback(
            tqdm_kwargs={
                "desc": f"Uploading {relative_path!s}",
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
                "miniters": 1,
            },
        )


class TqdmPerFileDownloadCallback(Callback):
    """A callback that creates a separate tqdm progress bar for each file download.

    Accepts a ``file_sizes`` dict (relative path -> byte size) so each
    progress bar can show percent-complete even when the SDK streams the
    file without a Content-Length header.
    """

    def __init__(self, dest_path: Path, fileset_path: str, file_sizes: dict[str, int] | None = None, **kwargs: Any):
        self.dest_path = dest_path
        self.fileset_path = fileset_path.rstrip("/")
        self.file_sizes = file_sizes or {}
        super().__init__(**kwargs)

    def branched(self, full_src_path: str, full_dest_path: str, **kwargs: Any) -> TqdmCallback:
        dest_full_path = Path(full_dest_path)
        if self.dest_path.is_file():
            relative_path = dest_full_path.name
        else:
            try:
                relative_path = dest_full_path.relative_to(self.dest_path)
            except ValueError:
                relative_path = dest_full_path.name

        relative_file_path = full_src_path
        if full_src_path.startswith(self.fileset_path):
            relative_file_path = full_src_path[len(self.fileset_path) :].lstrip("/")

        file_size = self.file_sizes.get(relative_file_path)

        callback = TqdmCallback(
            tqdm_kwargs={
                "desc": f"Downloading {relative_path!s}",
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
                "miniters": 1,
            },
        )

        if file_size is not None:
            callback.set_size(file_size)

        return callback


class BaseProgressCallback(Callback):
    """Base class for file upload/download progress callbacks."""

    progress_reporter: ProgressReporter
    fileset_name: str
    total_files: int
    total_size: int
    stats: UploadStats | DownloadStats
    _lock: threading.Lock

    def __init__(
        self,
        progress_reporter: ProgressReporter,
        fileset_name: str,
        total_files: int,
        total_size: int,
        stats: UploadStats | DownloadStats,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.progress_reporter = progress_reporter
        self.fileset_name = str(fileset_name)
        self.total_files = total_files
        self.total_size = total_size
        self.stats = stats
        self._lock = threading.Lock()

    @staticmethod
    def list_local_files(src_path: Path) -> list[FileInfo]:
        """List all files under *src_path* (see shared ``list_local_files``)."""
        return [FileInfo(path=f.path, size=f.size) for f in _list_local_files(src_path)]

    @abstractmethod
    def branched(self, source_path: str, dest_path: str, **kwargs: Any) -> "BaseSingleFileCallback":
        """Create a child callback for a single file transfer."""
        ...


class BaseSingleFileCallback(Callback):
    """Base class for per-file callbacks within a batch operation."""

    parent: BaseProgressCallback
    source_path: str
    dest_path: str
    _completed: bool

    def __init__(
        self,
        parent: BaseProgressCallback,
        source_path: str,
        dest_path: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.parent = parent
        self.source_path = source_path
        self.dest_path = dest_path
        self._completed = False

    @abstractmethod
    def _get_phase(self) -> str:
        """Return the TaskPhase for this operation."""
        ...

    @abstractmethod
    def _get_file_display_path(self) -> str:
        """Return the path to use for display/logging."""
        ...

    @abstractmethod
    def _update_stats(self) -> None:
        """Update the parent's stats for this operation (called within lock)."""
        ...

    @abstractmethod
    def _get_files_count(self) -> int:
        """Return the current files count from stats (called within lock)."""
        ...

    @abstractmethod
    def _build_status_details(self, files_count: int, total_bytes: int, current_file: str) -> dict[str, Any]:
        """Build the status_details dict for progress reporting."""
        ...

    def close(self) -> None:
        """Called when the file transfer completes."""
        if self._completed:
            return

        self._completed = True
        parent = self.parent
        current_file = self._get_file_display_path()

        with parent._lock:
            self._update_stats()
            files_count = self._get_files_count()
            total_bytes = parent.stats.total_bytes

        logger.debug(f"File transferred: {current_file} ({files_count}/{parent.total_files})")

        parent.progress_reporter.update_progress(
            status=HelixJobStatus.ACTIVE,
            status_details=self._build_status_details(files_count, total_bytes, current_file),
        )

    def __enter__(self) -> "BaseSingleFileCallback":
        return self

    def __exit__(self, *exc_args: object) -> None:
        self.close()


class FileUploadProgressCallback(BaseProgressCallback):
    """Callback for tracking file upload progress and reporting to the Jobs service."""

    stats: UploadStats

    def __init__(
        self,
        progress_reporter: ProgressReporter,
        src_path: Path,
        fileset_name: str,
        stats: UploadStats,
        **kwargs: Any,
    ):
        files = self.list_local_files(src_path)
        if not files:
            logger.warning(f"Source path {src_path} contains no files")
        total_files = len(files)
        total_size = sum(f.size for f in files)

        super().__init__(
            progress_reporter=progress_reporter,
            fileset_name=fileset_name,
            total_files=total_files,
            total_size=total_size,
            stats=stats,
            **kwargs,
        )

        logger.info(f"Uploading {total_files} files ({total_size} bytes) to {self.fileset_name}")

        progress_reporter.update_progress(
            status=HelixJobStatus.ACTIVE,
            status_details={
                "phase": TaskPhase.UPLOADING,
                "fileset": self.fileset_name,
                "total_files": total_files,
                "total_size": total_size,
                "uploaded_files": 0,
                "uploaded_bytes": 0,
            },
        )

    def branched(self, source_path: str, dest_path: str, **kwargs: Any) -> "SingleFileUploadCallback":
        return SingleFileUploadCallback(
            parent=self,
            source_path=source_path,
            dest_path=dest_path,
            **kwargs,
        )


class SingleFileUploadCallback(BaseSingleFileCallback):
    """Per-file upload callback. Notifies parent on completion."""

    parent: FileUploadProgressCallback

    def _get_phase(self) -> str:
        return TaskPhase.UPLOADING

    def _get_file_display_path(self) -> str:
        return self.dest_path.split("/")[-1] if "/" in self.dest_path else self.dest_path

    def _update_stats(self) -> None:
        self.parent.stats.files_uploaded += 1
        if self.size is not None:
            self.parent.stats.total_bytes += self.size

    def _get_files_count(self) -> int:
        return self.parent.stats.files_uploaded

    def _build_status_details(self, files_count: int, total_bytes: int, current_file: str) -> dict[str, Any]:
        return {
            "phase": TaskPhase.UPLOADING,
            "fileset": self.parent.fileset_name,
            "total_files": self.parent.total_files,
            "total_size": self.parent.total_size,
            "uploaded_files": files_count,
            "uploaded_bytes": total_bytes,
            "current_file": current_file,
            "progress_pct": get_percentage(files_count, self.parent.total_files),
        }


class FileDownloadProgressCallback(BaseProgressCallback):
    """Callback for tracking file download progress and reporting to the Jobs service."""

    stats: DownloadStats

    def __init__(
        self,
        progress_reporter: ProgressReporter,
        fileset_name: str,
        total_files: int,
        total_size: int,
        stats: DownloadStats,
        **kwargs: Any,
    ):
        super().__init__(
            progress_reporter=progress_reporter,
            fileset_name=fileset_name,
            total_files=total_files,
            total_size=total_size,
            stats=stats,
            **kwargs,
        )

        logger.info(f"Downloading {total_files} files ({total_size} bytes) from {self.fileset_name}")

        progress_reporter.update_progress(
            status=HelixJobStatus.ACTIVE,
            status_details={
                "phase": TaskPhase.DOWNLOADING,
                "fileset": self.fileset_name,
                "total_files": total_files,
                "total_size": total_size,
                "downloaded_files": 0,
                "downloaded_bytes": 0,
            },
        )

    def branched(self, source_path: str, dest_path: str, **kwargs: Any) -> "SingleFileDownloadCallback":
        return SingleFileDownloadCallback(
            parent=self,
            source_path=source_path,
            dest_path=dest_path,
            **kwargs,
        )


class SingleFileDownloadCallback(BaseSingleFileCallback):
    """Per-file download callback. Notifies parent on completion."""

    parent: FileDownloadProgressCallback

    def _get_phase(self) -> str:
        return TaskPhase.DOWNLOADING

    def _get_file_display_path(self) -> str:
        return self.source_path.split("/")[-1] if "/" in self.source_path else self.source_path

    def _update_stats(self) -> None:
        self.parent.stats.files_downloaded += 1
        if self.size is not None:
            self.parent.stats.total_bytes += self.size

    def _get_files_count(self) -> int:
        return self.parent.stats.files_downloaded

    def _build_status_details(self, files_count: int, total_bytes: int, current_file: str) -> dict[str, Any]:
        return {
            "phase": TaskPhase.DOWNLOADING,
            "fileset": self.parent.fileset_name,
            "total_files": self.parent.total_files,
            "total_size": self.parent.total_size,
            "downloaded_files": files_count,
            "downloaded_bytes": total_bytes,
            "current_file": current_file,
            "progress_pct": get_percentage(files_count, self.parent.total_files),
        }


class CompositeCallback(Callback):
    """A callback that delegates to multiple child callbacks."""

    def __init__(self, *callbacks: Callback, **kwargs: Any):
        super().__init__(**kwargs)
        self.callbacks = list(callbacks)

    def set_size(self, size: int) -> None:
        self.size = size
        for cb in self.callbacks:
            cb.set_size(size)

    def absolute_update(self, value: int) -> None:
        self.value = value
        for cb in self.callbacks:
            cb.absolute_update(value)

    def relative_update(self, inc: int = 1) -> None:
        self.value += inc
        for cb in self.callbacks:
            cb.relative_update(inc)

    def branched(self, source_path: str, dest_path: str, **kwargs: Any) -> "CompositeCallback":
        child_callbacks = [cb.branched(source_path, dest_path, **kwargs) for cb in self.callbacks]
        return CompositeCallback(*child_callbacks)

    def call(self, hook_name: str | None = None, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.call(hook_name, **kwargs)

    def close(self) -> None:
        for cb in self.callbacks:
            cb.close()

    def __enter__(self) -> "CompositeCallback":
        for cb in self.callbacks:
            cb.__enter__()
        return self

    def __exit__(self, *exc_args: object) -> None:
        for cb in self.callbacks:
            cb.__exit__(*exc_args)
