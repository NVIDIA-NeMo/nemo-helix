# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute copy/paste examples without contacting a running platform."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import yaml
from nemo_anonymizer_plugin.app.task_config import PreviewRequest
from nemo_helix_plugin.files.client import FilesClient

from docs._scripts.lint_python_snippets import extract_python_snippets

DOCS = Path(__file__).resolve().parents[1]


def test_file_progress_example_runs_in_isolation(tmp_path: Path, monkeypatch) -> None:
    snippets = extract_python_snippets(DOCS / "get-started/concepts/manage-files.mdx")
    source = next(snippet.source for snippet in snippets if 'description="Uploading dataset"' in snippet.source)
    files = Mock(spec=FilesClient)
    filesystem = Mock()
    monkeypatch.chdir(tmp_path)
    with (
        patch("nemo_helix_plugin.client.client.NemoClient.from_config", return_value=Mock(files=files)),
        patch("filesets.FilesetFileSystem", return_value=filesystem),
        patch("filesets.RichProgressCallback", return_value=MagicMock()),
    ):
        exec(source, {})
    assert (tmp_path / "large_dataset/sample.jsonl").read_text() == '{"text": "Example training record"}\n'
    filesystem.put.assert_called_once()
    filesystem.get.assert_called_once()


def test_anonymizer_preview_spec_is_created_and_round_trips(tmp_path: Path) -> None:
    snippets = extract_python_snippets(DOCS / "anonymizer/tutorials/preview.mdx")
    build = next(snippet.source for snippet in snippets if "request = PreviewRequest(" in snippet.source)
    export = next(snippet.source for snippet in snippets if "yaml.safe_dump(" in snippet.source)
    namespace = {}
    with patch("nemo_helix_plugin.client.client.NemoClient.from_config") as from_config:
        from_config.return_value.resolve_workspace.return_value = "example-workspace"
        exec(build, namespace)
    output = tmp_path / "anonymizer-preview.yaml"
    # Redirect only the documented output path, keeping the snippet's real I/O.
    exec(export.replace("/tmp/anonymizer-preview.yaml", str(output)), namespace)
    restored = PreviewRequest.model_validate(yaml.safe_load(output.read_text()))
    assert restored == namespace["request"]
    assert restored.data.source == "fileset://example-workspace/anonymizer-inputs#anonymizer-input.csv"
    assert restored.num_records == 2
    assert restored.model_configs
