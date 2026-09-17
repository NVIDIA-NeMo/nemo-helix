# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIRenderer implementations for ``nemo data-designer`` verbs.

The renderers wrap the existing ``data_designer.cli.ui`` helpers
(``print_header``, ``print_success``, ``print_error``, ``wait_for_navigation_key``)
so the Nemo CLI's preview / create UX stays consistent with the library's
own ``data-designer preview`` / ``data-designer create`` commands —
including the interactive per-record browser the library's
``GenerationController`` ships.

The interactive browser needs the original :class:`DataDesignerConfigBuilder`
to construct :class:`PreviewResults`. The override wrappers in
``cli/inputs.py`` stash the builder in a contextvar before invoking the
original callback; this renderer reads it back via
:func:`get_current_config_builder`. When the builder isn't available
(e.g. a renderer instantiated outside the override path) the renderer
falls back to a plain dataframe table.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from data_designer.cli.ui import console, print_error, print_header, print_success, wait_for_navigation_key
from data_designer.cli.utils.sample_records_pager import PAGER_FILENAME, create_sample_records_pager
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.preview_results import PreviewResults
from data_designer.errors import DataDesignerError
from data_designer_nemo.errors import NDDError
from nemo_data_designer_plugin.cli.inputs import get_current_config_builder
from nemo_data_designer_plugin.functions._types import (
    AnalysisFrame,
    DatasetFrame,
    DatasetMetadataFrame,
    LogFrame,
    PreviewFrame,
    ProcessorOutputFrame,
)
from nemo_platform_plugin.cli_renderer import CLIRenderer, RendererContext
from nemo_platform_plugin.functions.frames import Done, Error, Heartbeat
from pydantic import BaseModel, TypeAdapter

_PREVIEW_FRAME_ADAPTER: TypeAdapter[PreviewFrame] = TypeAdapter(PreviewFrame)

# Play nicely with the upstream library's padding
_SUMMARY_INDENT = " " * 6


def _coerce_preview_frame(frame: Any) -> BaseModel | None:
    """Turn a raw frame into a typed frame.

    Returns ``None`` if the frame can't be parsed (unknown ``kind`` etc.) so
    the renderer can silently skip it instead of raising.
    """
    if isinstance(frame, BaseModel):
        return frame
    if isinstance(frame, dict):
        try:
            return _PREVIEW_FRAME_ADAPTER.validate_python(frame)
        except Exception:
            return None
    return None


class PreviewRenderer(CLIRenderer):
    """Renderer for ``nemo data-designer preview``.

    Streams log frames as colored Rich output during execution; on completion
    shows the dataset, analysis report, and a success summary. On error,
    prints a single error line.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._dataset_metadata: DatasetMetadata | None = None
        self._analysis: DatasetProfilerResults | None = None
        self._processor_outputs: dict[str, list[dict]] = {}
        self._error_message: str | None = None
        self._log_levels_seen: set[str] = set()

    def on_start(self, *, ctx: RendererContext) -> None:
        print_header("Data Designer Preview")

    def on_frame(self, frame: Any, *, ctx: RendererContext) -> None:
        typed = _coerce_preview_frame(frame)
        if typed is None:
            return
        if isinstance(typed, LogFrame):
            self._log_levels_seen.add(typed.level)
            _print_log(typed)
        elif isinstance(typed, DatasetFrame):
            self._records.extend(typed.records)
        elif isinstance(typed, DatasetMetadataFrame):
            self._dataset_metadata = typed.metadata
        elif isinstance(typed, AnalysisFrame):
            self._analysis = typed.analysis
        elif isinstance(typed, ProcessorOutputFrame):
            self._processor_outputs[typed.processor_name] = typed.records
        elif isinstance(typed, Error):
            self._error_message = typed.message
        elif isinstance(typed, (Heartbeat, Done)):
            pass

    def on_complete(self, *, ctx: RendererContext) -> None:
        if self._error_message:
            print_error(f"Preview failed: {self._error_message}")
            return
        if not self._records:
            print_error("Preview completed without generating any records. Check the log lines above.")
            return

        df = pd.DataFrame(self._records).convert_dtypes(dtype_backend="pyarrow")
        total = len(df)
        results = self._build_preview_results(df)

        console.print()
        save_requested = bool(ctx.cli_kwargs.get("save_results"))
        if save_requested and results is None:
            print_error("--save-results requires a config builder; falling back to in-terminal output.")

        if save_requested and results is not None:
            self._save_results(results, df, total, ctx)
        else:
            self._render_to_terminal(results, df, total, ctx)

        console.print()
        self._print_final_status(total)

    def _build_preview_results(self, df: pd.DataFrame) -> PreviewResults | None:
        """Assemble a PreviewResults from the stashed builder, if available.

        The override wrapper in ``cli/inputs.py`` stashes the builder so the
        renderer can unlock the library's per-record HTML rendering via
        ``display_sample_record(...)``. When the builder isn't available,
        callers fall back to a plain dataframe table.
        """
        builder = get_current_config_builder()
        if builder is None:
            return None
        try:
            return PreviewResults(
                config_builder=builder,
                dataset=df,
                dataset_metadata=self._dataset_metadata,
                analysis=self._analysis,
                processor_artifacts=self._processor_outputs or None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[yellow]Could not assemble PreviewResults: {exc}[/yellow]")
            return None

    def _render_to_terminal(
        self,
        results: PreviewResults | None,
        df: pd.DataFrame,
        total: int,
        ctx: RendererContext,
    ) -> None:
        if results is not None and _can_browse_interactively(ctx, total):
            self._browse_interactively(results, total)
        else:
            console.print(f"  [bold]{total} record(s) generated[/bold]")
            console.print()
            self._display_all_records(results, df, total)

        analysis = self._analysis
        if analysis is not None and hasattr(analysis, "to_report"):
            try:
                console.print()
                analysis.to_report()
            except Exception as exc:  # pragma: no cover - defensive
                console.print(f"[yellow]Could not render analysis report: {exc}[/yellow]")

    def _print_final_status(self, total: int) -> None:
        if "error" in self._log_levels_seen:
            print_error("Preview completed with errors. See the log lines above.")
        elif "warning" in self._log_levels_seen or "warn" in self._log_levels_seen:
            console.print("[yellow]⚠ Preview completed with warnings.[/yellow]")
        else:
            print_success(f"Preview complete — {total} record(s) generated")

    def _save_results(
        self,
        results: PreviewResults,
        df: pd.DataFrame,
        total: int,
        ctx: RendererContext,
    ) -> None:
        """Write preview artifacts to disk, mirroring upstream's --save-results layout."""
        artifact_path_raw = ctx.cli_kwargs.get("artifact_path")
        artifact_path = Path(artifact_path_raw) if artifact_path_raw else Path.cwd() / "artifacts"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = artifact_path / f"preview_results_{timestamp}"
        sample_records_dir = results_dir / "sample_records"
        sample_records_dir.mkdir(parents=True, exist_ok=True)

        df.to_parquet(results_dir / "dataset.parquet")

        analysis = self._analysis
        if analysis is not None and hasattr(analysis, "to_report"):
            try:
                analysis.to_report(save_path=results_dir / "report.html")
            except Exception as exc:  # pragma: no cover - defensive
                console.print(f"[yellow]Could not save analysis report: {exc}[/yellow]")

        for i in range(total):
            results.display_sample_record(
                index=i,
                save_path=sample_records_dir / f"record_{i}.html",
            )
        create_sample_records_pager(
            sample_records_dir=sample_records_dir,
            num_records=total,
            num_columns=len(df.columns),
        )

        console.print(f"  [bold]Results path:[/bold] {results_dir}")
        console.print(f"  [bold]Browser path:[/bold] {sample_records_dir / PAGER_FILENAME}")

    @staticmethod
    def _browse_interactively(results: PreviewResults, total: int) -> None:
        """Per-record browser, mirroring data_designer.cli.controllers.GenerationController."""
        current = 0
        PreviewRenderer._show_record(results, current, total)
        while True:
            console.print()
            action = wait_for_navigation_key()
            if action == "q":
                console.print("  [dim]Done browsing.[/dim]")
                break
            if action == "p":
                current = (current - 1) % total
            else:
                current = (current + 1) % total
            PreviewRenderer._show_record(results, current, total)

    @staticmethod
    def _show_record(results: PreviewResults, index: int, total: int) -> None:
        console.print(f"  [bold]Record {index + 1} of {total}[/bold]")
        results.display_sample_record(index=index)

    @staticmethod
    def _display_all_records(results: PreviewResults | None, df: pd.DataFrame, total: int) -> None:
        if results is not None:
            for i in range(total):
                PreviewRenderer._show_record(results, i, total)
        else:
            console.print(df.to_string(index=False, max_cols=10))

    def on_error(self, error: BaseException, *, ctx: RendererContext) -> None:
        print_error(f"Preview failed: {error}")
        _handle_error(error)


class CreateRenderer(CLIRenderer):
    """Renderer for ``nemo data-designer create``.

    Prints a compact submission summary — the job name, its record count,
    and the follow-up commands needed to track it — instead of dumping the
    whole job entity.

    The job submit path is non-streaming. The framework's single-value
    driver calls :meth:`on_frame` exactly once, with the decoded
    ``PlatformJobResponse``, so there is no frame-kind ambiguity to resolve
    here (unlike :class:`PreviewRenderer`, which reads an NDJSON stream of
    mixed frames). The only guard is on the response *shape*: an unexpected
    payload falls back to the raw dump so the job's identity is never lost.
    """

    def __init__(self) -> None:
        self._frame: Any = None

    def on_start(self, *, ctx: RendererContext) -> None:
        print_header("Data Designer Create")

    def on_frame(self, frame: Any, *, ctx: RendererContext) -> None:
        # Jobs are non-streaming; on_frame fires exactly once with the
        # submission response. Buffer it for the summary in on_complete.
        self._frame = frame

    def on_complete(self, *, ctx: RendererContext) -> None:
        console.print()
        print_success("Create submitted.")
        console.print()
        job_name = _job_name(self._frame)
        if job_name is None:
            self._render_fallback()
        else:
            self._render_summary(job_name, ctx)

    def _render_summary(self, job_name: str, ctx: RendererContext) -> None:
        frame: dict = self._frame
        rows: list[tuple[str, str]] = [("Job", job_name)]

        num_records = _num_records(frame, ctx)
        if num_records is not None:
            rows.append(("Records", str(num_records)))

        output_location = frame.get("output_location")
        if isinstance(output_location, str) and output_location:
            # Only set when the caller supplied a fileset; None when auto-created.
            rows.append(("Output", output_location))

        suffix = _workspace_suffix(frame, ctx)
        rows.extend(
            [
                ("Track", f"nemo jobs get {job_name}{suffix}"),
                ("Watch", f"nemo jobs watch {job_name}{suffix}"),
                ("Results", f"nemo jobs results list {job_name}{suffix}"),
            ]
        )

        width = max(len(label) for label, _ in rows) + 1
        for label, value in rows:
            # soft_wrap keeps the follow-up commands on one logical line;
            # Rich's default wrapping would split them mid-flag and break
            # the copy-paste these lines exist for.
            console.print(f"{_SUMMARY_INDENT}[bold]{f'{label}:':<{width}}[/bold]  {value}", soft_wrap=True)

        error_details = frame.get("error_details")
        if error_details:
            console.print()
            console.print(f"{_SUMMARY_INDENT}[yellow]⚠ Error details: {error_details}[/yellow]", soft_wrap=True)

    def _render_fallback(self) -> None:
        """Dump the raw payload when the response shape isn't what we expect.

        Better a wall of text than a job the user can no longer find.
        """
        console.print(self._frame)
        console.print()
        console.print(
            f"{_SUMMARY_INDENT}[dim]Could not read a job name from the response; "
            f"find the job with: nemo jobs list[/dim]"
        )

    def on_error(self, error: BaseException, *, ctx: RendererContext) -> None:
        print_error(f"Create failed: {error}")
        _handle_error(error)


def _job_name(frame: Any) -> str | None:
    """Return the job name when *frame* looks like a job entity, else ``None``."""
    if not isinstance(frame, dict):
        return None
    name = frame.get("name")
    return name if isinstance(name, str) and name else None


def _num_records(frame: dict, ctx: RendererContext) -> int | None:
    """Pull the record count from the stored spec, falling back to the CLI spec.

    The job entity stores a ``DataDesignerStepConfig``, so the count nests
    under ``spec.job_config``. The wrapper in ``cli/inputs.py`` builds the
    same count into the ``--spec`` JSON, which is the fallback when the
    response omits or restructures the spec.
    """
    spec = frame.get("spec")
    if isinstance(spec, dict):
        job_config = spec.get("job_config")
        if isinstance(job_config, dict):
            value = job_config.get("num_records")
            if isinstance(value, int):
                return value

    raw_spec = ctx.cli_kwargs.get("spec")
    if isinstance(raw_spec, str) and raw_spec:
        try:
            parsed = json.loads(raw_spec)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            value = parsed.get("num_records")
            if isinstance(value, int):
                return value
    return None


def _workspace_suffix(frame: dict, ctx: RendererContext) -> str:
    """Return ``" --workspace <ws>"`` for non-default workspaces, else ``""``.

    Without it the printed commands are silently wrong for anyone who
    passed ``-w``/``--workspace`` to ``create``.
    """
    workspace = frame.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        candidate = ctx.cli_kwargs.get("workspace")
        workspace = candidate if isinstance(candidate, str) else None
    if not workspace or workspace == "default":
        return ""
    return f" --workspace {workspace}"


def _print_log(frame: LogFrame) -> None:
    """Print a LogFrame to the console with level-appropriate coloring."""
    level = frame.level
    if level == "error":
        prefix = "[red][error][/red]"
    elif level in ("warning", "warn"):
        prefix = "[yellow][warn][/yellow]"
    elif level == "info":
        prefix = "[cyan][info][/cyan]"
    else:
        prefix = "[dim][debug][/dim]"
    console.print(f"{prefix} {frame.message}")


def _can_browse_interactively(ctx: RendererContext, total: int) -> bool:
    """Mirror the upstream library's TTY-and-multi-record gate.

    Honors a ``--non-interactive`` flag if the override wrapper exposed one,
    falling back to TTY checks otherwise.
    """
    if total <= 1:
        return False
    if ctx.cli_kwargs.get("non_interactive"):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _handle_error(error: BaseException) -> None:
    """Raises a typer.Exit error if the error is a first-class
    Data Designer (library or platform plugin) error. This ensures
    the stacktrace doesn't leak out, and instead we only present
    the custom error message.
    """
    if isinstance(error, (DataDesignerError, NDDError)):
        raise typer.Exit(code=1) from error
