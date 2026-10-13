# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fsspec.callbacks import Callback


def _create_rich_progress(transient: bool = False, console=None):
    """Create a Rich Progress instance with adaptive columns.

    The progress bar uses custom columns that render differently based on
    whether the task is the main (file count) task or a file (bytes) task.
    Tasks should set `is_main=True` in their fields for file count display.
    """
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        Task,
        TextColumn,
        TimeElapsedColumn,
        TransferSpeedColumn,
    )
    from rich.table import Column
    from rich.text import Text

    class FilesOrBytesColumn(ProgressColumn):
        """Shows 'N/M files' for main task, bytes for file tasks."""

        def __init__(self):
            super().__init__(table_column=Column(min_width=14))
            self._files_col = MofNCompleteColumn()
            self._bytes_col = DownloadColumn()

        def render(self, task: Task) -> Text:
            if task.fields.get("is_main"):
                result = self._files_col.render(task)
                return Text(f"{result.plain} files", style="progress.download")
            return self._bytes_col.render(task)

    class AdaptiveSpeedColumn(ProgressColumn):
        """Shows '-' for main task, transfer speed for file tasks."""

        def __init__(self):
            super().__init__(table_column=Column(min_width=10))
            self._speed_col = TransferSpeedColumn()

        def render(self, task: Task) -> Text:
            if task.fields.get("is_main"):
                return Text("-", style="progress.data.speed")
            result = self._speed_col.render(task)
            if result.plain == "?":
                return Text("-", style="progress.data.speed")
            return result

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", table_column=Column(min_width=32)),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        FilesOrBytesColumn(),
        TextColumn("•"),
        AdaptiveSpeedColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=transient,
        console=console,
    )


class RichProgressCallback(Callback):
    """fsspec Callback that displays progress using Rich progress bars.

    Requires the `rich` package to be installed. If not installed,
    raises ImportError when instantiated.

    Shows:
    - Overall progress bar at top (file count)
    - Individual progress bar per file being transferred (bytes)

    Example:
        >>> with RichProgressCallback(description="Uploading") as callback:
        ...     fs.put(local_path, remote_path, callback=callback)
    """

    def __init__(
        self,
        description: str = "Transferring",
        transient: bool = False,
        console=None,
    ):
        """Initialize the Rich progress callback.

        Args:
            description: Text to display next to the progress bar.
            transient: If True, remove the progress bar after completion.
            console: Rich console to draw on. Defaults to Rich's global console
                (stdout); pass a stderr console when stdout carries output
                that scripts parse.
        """
        try:
            import rich  # noqa: F401
            from rich.progress import TaskID
        except ImportError as e:
            raise ImportError(
                "RichProgressCallback requires the 'rich' package. Install it with: uv pip install rich"
            ) from e

        super().__init__()
        self._description = description
        self._progress = _create_rich_progress(transient=transient, console=console)
        self._main_task_id: TaskID | None = None

    def set_size(self, size: int | None) -> None:
        """Set the total number of files."""
        if self._main_task_id is None or size is None:
            return
        self._progress.update(self._main_task_id, total=size)

    def absolute_update(self, value: int) -> None:
        """Set file progress to an absolute value."""
        if self._main_task_id is None:
            return
        self._progress.update(self._main_task_id, completed=value)

    def relative_update(self, inc: int = 1) -> None:
        """Increment file progress (called when a file completes)."""
        if self._main_task_id is None:
            return
        self._progress.advance(self._main_task_id, advance=inc)

    def branched(self, path_1: str, path_2: str, **kwargs) -> RichFileProgressCallback:
        """Create a child callback for an individual file transfer."""
        # Names match fsspec's ``Callback.branched`` so keyword callers still work.
        del path_2, kwargs
        filename = path_1.rsplit("/", 1)[-1] if "/" in path_1 else path_1
        return RichFileProgressCallback(filename, progress=self._progress)

    def __enter__(self):
        """Start the progress bar display."""
        self._progress.start()
        self._main_task_id = self._progress.add_task(self._description, total=None, is_main=True)
        return self

    def __exit__(self, *_exc_args):
        """Stop the progress bar display."""
        self._progress.stop()


class RichFileProgressCallback(Callback):
    """Callback that shows progress for a single file transfer.

    Can be used standalone with get_file() or as a child of RichProgressCallback
    for tracking individual files in a batch transfer.

    Example (standalone):
        >>> with RichFileProgressCallback() as callback:
        ...     fs.get_file(remote_path, local_path, callback=callback)

    Example (as child of RichProgressCallback):
        >>> # Typically created automatically via RichProgressCallback.branched()
        >>> callback = RichFileProgressCallback("myfile.txt", progress=parent._progress)
    """

    def __init__(
        self,
        filename: str = "Transferring",
        *,
        progress=None,
        transient: bool = False,
    ):
        """Initialize the file progress callback.

        Args:
            filename: Name/description to display. Defaults to "Transferring".
            progress: Optional Rich Progress instance. If not provided, creates its own.
            transient: If True and creating own progress bar, remove it after completion.
        """
        try:
            import rich  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "RichFileProgressCallback requires the 'rich' package. Install it with: uv pip install rich"
            ) from e

        from rich.progress import TaskID

        super().__init__()
        self._filename = filename
        self._owns_progress = progress is None
        self._progress = progress if progress is not None else _create_rich_progress(transient=transient)
        self._task_id: TaskID | None = None

    def set_size(self, size: int | None) -> None:
        """Create the task with the file size."""
        if size is None:
            return
        # Truncate long filenames
        display_name = self._filename
        if len(display_name) > 30:
            display_name = "..." + display_name[-27:]
        # Use different formatting for standalone vs child mode
        if self._owns_progress:
            task_description = display_name
        else:
            task_description = f"[dim]{display_name}[/dim]"
        self._task_id = self._progress.add_task(task_description, total=size, is_main=False)

    def relative_update(self, inc: int = 1) -> None:
        """Update bytes transferred for this file."""
        if self._task_id is None:
            return
        self._progress.advance(self._task_id, advance=inc)

    def absolute_update(self, value: int) -> None:
        """Set bytes transferred to absolute value."""
        if self._task_id is None:
            return
        self._progress.update(self._task_id, completed=value)

    def __enter__(self):
        """Start the progress bar display (only when used standalone)."""
        if self._owns_progress:
            self._progress.start()
        return self

    def __exit__(self, *_exc_args):
        """Stop the progress bar display (only when used standalone)."""
        if self._owns_progress:
            self._progress.stop()
        return False
