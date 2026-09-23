# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for creating Customizer inputs from local paths and HuggingFace ids."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nmp.customization_common.cli.uploads import (
    FilesUploadClient,
    SpecRefs,
    UploadError,
    _upload_source,
    create_resources,
    fileset_name_for,
    find_conflicts,
    is_huggingface_id,
    write_refs,
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def data(self) -> dict[str, Any]:
        return self._payload


class _FakeFiles:
    """Records calls; `existing` names the filesets the platform already has."""

    workspace = "default"

    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = existing or set()
        self.created: list[dict[str, Any]] = []
        self.uploaded: list[tuple[str, str]] = []
        #: Per uploaded path: the blocks received, and the headers the call carried.
        self.blocks: dict[str, list[bytes]] = {}
        self.headers: dict[str, dict[str, str]] = {}
        self._pending_headers: dict[str, str] = {}

    def with_headers(self, headers: Mapping[str, str]) -> _FakeFiles:
        self._pending_headers = dict(headers)
        return self

    def get_fileset(self, *, workspace: str, name: str) -> _Response:
        if name not in self.existing:
            raise NotFoundError(httpx.Response(404, json={"detail": f"no fileset {name}"}))
        return _Response({"name": name})

    def create_fileset(self, *, workspace: str, body: Any, exist_ok: bool) -> _Response:
        self.created.append({"name": body.name, "purpose": body.purpose, "storage": body.storage})
        return _Response({"name": body.name})

    def upload_file(self, *, workspace: str, name: str, path: str, content: Iterable[bytes]) -> _Response:
        assert not isinstance(content, bytes), "file content must be streamed, not read into memory"
        self.uploaded.append((name, path))
        self.blocks[path] = list(content)
        self.headers[path], self._pending_headers = self._pending_headers, {}
        return _Response({"path": path})


class _FakeModels:
    def __init__(self) -> None:
        self.created: list[tuple[str, str]] = []

    def create_model(self, *, workspace: str, body: Any, exist_ok: bool) -> _Response:
        self.created.append((body.name, body.fileset))
        return _Response({"name": body.name})


AUTOMODEL_REFS = SpecRefs(
    model=("model",),
    dataset=("dataset", "training"),
    dataset_validation=("dataset", "validation"),
)
UNSLOTH_REFS = SpecRefs(
    model=("model", "name"),
    dataset=("dataset", "path"),
    dataset_validation=("dataset", "validation_path"),
)
RL_REFS = SpecRefs(model=("model",), dataset=("dataset",), environment=("environment",))


def _Report(**kwargs: Any):
    from nmp.customization_common.cli.uploads import UploadReport

    return UploadReport(**kwargs)


def _run(
    *,
    files: FilesUploadClient,
    models: _FakeModels,
    model_source: str | None = None,
    dataset_source: str | None = None,
    environment_source: str | None = None,
    exist_ok: bool = False,
    hf_token_secret: str | None = None,
):
    return create_resources(
        model_source=model_source,
        dataset_source=dataset_source,
        environment_source=environment_source,
        files=files,
        models=models,
        workspace="default",
        exist_ok=exist_ok,
        hf_token_secret=hf_token_secret,
    )


class TestNaming:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("Qwen/Qwen3-1.7B", "qwen3-1-7b"),
            ("./data/train.jsonl", "train"),
            ("/tmp/My Data/", "my-data"),
        ],
    )
    def test_fileset_name_is_predictable(self, source: str, expected: str) -> None:
        assert fileset_name_for(source) == expected

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            # The platform requires a leading letter and at least two characters.
            ("./2024-train.jsonl", "dataset-2024-train"),
            ("./m", "dataset-m"),
            # Repeated hyphens are rejected, so runs are collapsed.
            ("/tmp/My  Data/", "my-data"),
        ],
    )
    def test_a_name_the_platform_would_reject_gets_the_prefix(self, source: str, expected: str) -> None:
        assert fileset_name_for(source, prefix="dataset") == expected

    def test_every_derived_name_satisfies_the_platform_pattern(self) -> None:
        import re

        from nemo_platform_plugin.entity_naming import NAME_PATTERN

        sources = ["Qwen/Qwen3-1.7B", "./2024-train.jsonl", "./m", "/tmp/My  Data/", "./train.jsonl", "./A_B.C"]
        for source in sources:
            assert re.match(NAME_PATTERN, fileset_name_for(source, prefix="dataset")), source

    def test_an_underivable_name_is_reported(self) -> None:
        with pytest.raises(UploadError, match="Cannot derive a valid fileset name"):
            fileset_name_for("///")

    def test_model_sizes_do_not_collide(self) -> None:
        """A version suffix is part of the name, not a file extension."""
        assert fileset_name_for("Qwen/Qwen3-1.7B") != fileset_name_for("Qwen/Qwen3-1.5B")

    def test_huggingface_id_is_recognised(self) -> None:
        assert is_huggingface_id("Qwen/Qwen3-1.7B")

    def test_existing_local_path_is_not_a_huggingface_id(self, tmp_path: Path) -> None:
        nested = tmp_path / "org" / "model"
        nested.mkdir(parents=True)
        assert not is_huggingface_id(str(nested))


class TestModelUpload:
    def test_huggingface_model_becomes_an_external_fileset(self) -> None:
        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, model_source="Qwen/Qwen3-1.7B")

        assert report.model_ref == "default/qwen3-1-7b"
        assert files.created[0]["storage"].repo_id == "Qwen/Qwen3-1.7B"
        assert not files.uploaded, "a HuggingFace repo is fetched by the platform, not uploaded"
        assert models.created == [("qwen3-1-7b", "default/qwen3-1-7b")]

    def test_gated_model_carries_the_token_secret(self) -> None:
        files, models = _FakeFiles(), _FakeModels()
        _run(files=files, models=models, model_source="meta-llama/Llama-3.1-8B", hf_token_secret="hf-token")

        assert files.created[0]["storage"].token_secret.root == "hf-token"

    def test_local_model_directory_is_uploaded(self, tmp_path: Path) -> None:
        source = tmp_path / "my-model"
        source.mkdir()
        (source / "config.json").write_text("{}")
        (source / "model.safetensors").write_bytes(b"w")

        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, model_source=str(source))

        assert report.model_ref == "default/my-model"
        assert sorted(path for _, path in files.uploaded) == ["config.json", "model.safetensors"]

    def test_model_entity_is_registered_against_the_fileset(self, tmp_path: Path) -> None:
        source = tmp_path / "m"
        source.mkdir()
        (source / "config.json").write_text("{}")
        files, models = _FakeFiles(), _FakeModels()
        _run(files=files, models=models, model_source=str(source))

        assert models.created == [("model-m", "default/model-m")]

    def test_a_bad_model_source_is_reported(self) -> None:
        files, models = _FakeFiles(), _FakeModels()
        with pytest.raises(UploadError, match="HuggingFace repo id of the form"):
            _run(files=files, models=models, model_source="./not/here")


class TestDatasetUpload:
    def test_local_file_keeps_its_name(self, tmp_path: Path) -> None:
        train = tmp_path / "mydata.jsonl"
        train.write_text("{}\n")

        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, dataset_source=str(train))

        assert report.dataset_ref == "default/mydata"
        assert files.uploaded == [("mydata", "mydata.jsonl")]

    def test_directory_keeps_its_layout(self, tmp_path: Path) -> None:
        """NeMo-RL reads training.jsonl and validation.jsonl from one fileset."""
        source = tmp_path / "prefs"
        source.mkdir()
        (source / "training.jsonl").write_text("{}\n")
        (source / "validation.jsonl").write_text("{}\n")

        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, dataset_source=str(source))

        assert report.dataset_ref == "default/prefs"
        assert files.uploaded == [("prefs", "training.jsonl"), ("prefs", "validation.jsonl")]

    def test_missing_local_dataset_is_reported(self) -> None:
        files, models = _FakeFiles(), _FakeModels()
        with pytest.raises(UploadError, match="does not exist"):
            _run(files=files, models=models, dataset_source="/nope/train.jsonl")

    def test_a_huggingface_id_is_not_accepted_as_a_dataset(self) -> None:
        """Converting an HF dataset to the backend's format is a separate step."""
        files, models = _FakeFiles(), _FakeModels()
        with pytest.raises(UploadError, match="convert a\\s+HuggingFace dataset"):
            _run(files=files, models=models, dataset_source="tau/commonsense_qa")


class TestStreamedUpload:
    """Files are streamed as ``nemo files upload`` streams them, never read whole."""

    def test_file_is_sent_in_blocks_with_its_length(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from filesets import FilesetFileSystem

        monkeypatch.setattr(FilesetFileSystem, "blocksize", 4)
        shard = tmp_path / "weights"
        shard.mkdir()
        (shard / "model.safetensors").write_bytes(b"0123456789")

        files = _FakeFiles()
        _run(files=files, models=_FakeModels(), model_source=str(shard))

        assert files.blocks["model.safetensors"] == [b"0123", b"4567", b"89"]
        assert files.headers["model.safetensors"] == {"Content-Length": "10"}

    def test_progress_goes_to_stderr_not_stdout(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """stdout carries the refs as JSON for scripts; the progress bars must not land there."""
        data = tmp_path / "train.jsonl"
        data.write_text("{}\n")

        _run(files=_FakeFiles(), models=_FakeModels(), dataset_source=str(data))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Uploading train" in captured.err

    def test_the_real_files_client_sends_the_streamed_body(self, tmp_path: Path) -> None:
        """The typed FilesClient satisfies the upload protocol, header included."""
        from nemo_platform_plugin.files.client import FilesClient

        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["length"] = request.headers.get("content-length")
            seen["body"] = request.read()
            return httpx.Response(
                200,
                json={
                    "file_ref": "default/train#train.jsonl",
                    "file_url": str(request.url),
                    "path": "train.jsonl",
                    "size": len(seen["body"]),
                },
            )

        data = tmp_path / "train.jsonl"
        data.write_bytes(b'{"a": 1}\n')
        client = FilesClient(
            base_url="http://nmp.test",
            workspace="default",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        _upload_source(client, name="train", workspace="default", source=data)

        assert seen == {
            "path": "/apis/files/v2/workspaces/default/filesets/train/-/train.jsonl",
            "length": "9",
            "body": b'{"a": 1}\n',
        }


class TestEitherOrBoth:
    def test_model_only_leaves_the_dataset_alone(self) -> None:
        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, model_source="Qwen/Qwen3-1.7B")
        assert report.dataset_ref is None

    def test_dataset_only_leaves_the_model_alone(self, tmp_path: Path) -> None:
        train = tmp_path / "train.jsonl"
        train.write_text("{}\n")
        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, dataset_source=str(train))
        assert report.model_ref is None
        assert models.created == []

    def test_both_creates_both(self, tmp_path: Path) -> None:
        train = tmp_path / "train.jsonl"
        train.write_text("{}\n")
        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, model_source="Qwen/Qwen3-1.7B", dataset_source=str(train))
        assert report.model_ref == "default/qwen3-1-7b"
        assert report.dataset_ref == "default/train"


class TestWriteRefs:
    """Each backend holds its references in a different place."""

    def test_automodel_paths(self) -> None:
        """One uploaded fileset fills training and validation; the backend splits them."""
        spec: dict[str, Any] = {"model": "x", "dataset": {"training": "y"}}
        report = _Report(model_ref="default/m", dataset_ref="default/d")
        write_refs(spec, AUTOMODEL_REFS, report)
        assert spec == {
            "model": "default/m",
            "dataset": {"training": "default/d", "validation": "default/d"},
        }

    def test_unsloth_paths(self) -> None:
        spec: dict[str, Any] = {"model": {"name": "x"}, "dataset": {"path": "y"}}
        report = _Report(model_ref="default/m", dataset_ref="default/d")
        write_refs(spec, UNSLOTH_REFS, report)
        assert spec == {
            "model": {"name": "default/m"},
            "dataset": {"path": "default/d", "validation_path": "default/d"},
        }

    def test_rl_has_no_separate_validation_ref(self) -> None:
        spec: dict[str, Any] = {"model": "x", "dataset": "y"}
        report = _Report(model_ref="default/m", dataset_ref="default/d")
        write_refs(spec, RL_REFS, report)
        assert spec == {"model": "default/m", "dataset": "default/d"}

    def test_an_absent_ref_is_not_written(self) -> None:
        spec: dict[str, Any] = {"model": "keep", "dataset": {"training": "keep"}}
        write_refs(spec, AUTOMODEL_REFS, _Report(dataset_ref="default/d"))
        assert spec["model"] == "keep"

    @pytest.mark.parametrize(
        ("refs", "spec", "expected"),
        [
            (UNSLOTH_REFS, {"model": None}, {"model": {"name": "default/m"}}),
            (
                AUTOMODEL_REFS,
                {"dataset": "stale"},
                {"dataset": {"training": "default/d", "validation": "default/d"}},
            ),
        ],
        ids=["unsloth-null-model", "automodel-string-dataset"],
    )
    def test_a_non_object_on_the_path_is_replaced(
        self, refs: SpecRefs, spec: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        """The pre-upload check fills these paths the same way, so writing must not crash after upload."""
        report = _Report(
            model_ref="default/m" if "model" in expected else None,
            dataset_ref="default/d" if "dataset" in expected else None,
        )
        write_refs(spec, refs, report)
        assert spec == expected


class TestExistOk:
    def test_existing_fileset_fails_without_exist_ok(self, tmp_path: Path) -> None:
        train = tmp_path / "train.jsonl"
        train.write_text("{}\n")
        files, models = _FakeFiles(existing={"train"}), _FakeModels()

        with pytest.raises(UploadError, match=r"Fileset default/train already exists"):
            _run(files=files, models=models, dataset_source=str(train))

    def test_existing_fileset_is_reused_with_exist_ok(self, tmp_path: Path) -> None:
        train = tmp_path / "train.jsonl"
        train.write_text("{}\n")
        files, models = _FakeFiles(existing={"train"}), _FakeModels()
        report = _run(files=files, models=models, dataset_source=str(train), exist_ok=True)

        assert report.dataset_ref == "default/train"
        assert report.reused == ["default/train"]
        assert report.created == []

    def test_reused_fileset_does_not_re_upload_files(self, tmp_path: Path) -> None:
        """Matches the Files service: --exist-ok reuses the fileset and leaves its files."""
        train = tmp_path / "train.jsonl"
        train.write_text("{}\n")
        files, models = _FakeFiles(existing={"train"}), _FakeModels()
        report = _run(files=files, models=models, dataset_source=str(train), exist_ok=True)

        assert files.uploaded == []
        assert report.reused_files_untouched

    def test_exist_ok_applies_to_both_when_both_are_given(self, tmp_path: Path) -> None:
        train = tmp_path / "train.jsonl"
        train.write_text("{}\n")
        files, models = _FakeFiles(existing={"train", "qwen3-1-7b"}), _FakeModels()
        report = _run(
            files=files,
            models=models,
            model_source="Qwen/Qwen3-1.7B",
            dataset_source=str(train),
            exist_ok=True,
        )

        assert sorted(report.reused) == ["default/qwen3-1-7b", "default/train"]
        assert report.created == []


class TestEnvironmentUpload:
    """GRPO needs an environment fileset alongside its dataset."""

    def test_environment_directory_becomes_an_environment_fileset(self, tmp_path: Path) -> None:
        env = tmp_path / "math-env"
        env.mkdir()
        (env / "config.yaml").write_text("x: 1\n")

        files, models = _FakeFiles(), _FakeModels()
        report = _run(files=files, models=models, environment_source=str(env))

        assert report.environment_ref == "default/math-env"
        assert files.created[0]["purpose"].value == "environment"
        assert files.uploaded == [("math-env", "config.yaml")]

    def test_environment_ref_is_written_for_rl_only(self) -> None:
        spec: dict[str, Any] = {"model": "m", "dataset": "d", "environment": "e"}
        write_refs(spec, RL_REFS, _Report(environment_ref="default/env"))
        assert spec["environment"] == "default/env"

        # automodel has no environment, so the ref is dropped rather than invented.
        other: dict[str, Any] = {"model": "m", "dataset": {"training": "d"}}
        write_refs(other, AUTOMODEL_REFS, _Report(environment_ref="default/env"))
        assert "environment" not in other


class TestSiblingOptionsSurvive:
    """Writing a reference must not disturb the other options in its block.

    Unsloth reads ``dataset.apply_chat_template`` beside ``dataset.path``; losing
    it makes SFTTrainer reject rows with "You must specify a formatting_func".
    """

    def test_dataset_options_are_kept(self) -> None:
        spec: dict[str, Any] = {"dataset": {"apply_chat_template": True, "text_field": "text"}}
        write_refs(spec, UNSLOTH_REFS, _Report(dataset_ref="default/d"))
        assert spec["dataset"] == {
            "apply_chat_template": True,
            "text_field": "text",
            "path": "default/d",
            "validation_path": "default/d",
        }

    def test_model_options_are_kept(self) -> None:
        spec: dict[str, Any] = {"model": {"max_seq_length": 2048, "load_in_4bit": True}}
        write_refs(spec, UNSLOTH_REFS, _Report(model_ref="default/m"))
        assert spec["model"] == {"max_seq_length": 2048, "load_in_4bit": True, "name": "default/m"}

    def test_a_missing_block_is_created(self) -> None:
        spec: dict[str, Any] = {}
        write_refs(spec, AUTOMODEL_REFS, _Report(dataset_ref="default/d"))
        assert spec == {"dataset": {"training": "default/d", "validation": "default/d"}}


class TestConflicts:
    """A reference in the job JSON and on the command line is a contradiction."""

    def test_model_in_both_places_is_a_conflict(self) -> None:
        spec: dict[str, Any] = {"model": "default/already-there"}
        conflicts = find_conflicts(spec, AUTOMODEL_REFS, model=True, dataset=False, environment=False)
        assert conflicts == ["model (default/already-there)"]

    def test_the_validation_ref_counts_as_a_dataset_conflict(self) -> None:
        """The reported bug: only training was replaced, leaving validation stale."""
        spec: dict[str, Any] = {"dataset": {"validation": "default/old"}}
        conflicts = find_conflicts(spec, AUTOMODEL_REFS, model=False, dataset=True, environment=False)
        assert conflicts == ["dataset.validation (default/old)"]

    def test_a_field_with_no_matching_flag_is_not_a_conflict(self) -> None:
        spec: dict[str, Any] = {"model": "default/keep"}
        assert find_conflicts(spec, AUTOMODEL_REFS, model=False, dataset=True, environment=False) == []

    def test_an_empty_spec_has_no_conflicts(self) -> None:
        assert find_conflicts({}, AUTOMODEL_REFS, model=True, dataset=True, environment=False) == []

    def test_an_empty_string_is_not_treated_as_set(self) -> None:
        spec: dict[str, Any] = {"model": ""}
        assert find_conflicts(spec, AUTOMODEL_REFS, model=True, dataset=False, environment=False) == []

    def test_every_conflicting_field_is_reported_at_once(self) -> None:
        spec: dict[str, Any] = {
            "model": "default/m",
            "dataset": {"training": "default/t", "validation": "default/v"},
        }
        conflicts = find_conflicts(spec, AUTOMODEL_REFS, model=True, dataset=True, environment=False)
        assert len(conflicts) == 3

    def test_rl_environment_conflict(self) -> None:
        spec: dict[str, Any] = {"environment": "default/env"}
        conflicts = find_conflicts(spec, RL_REFS, model=False, dataset=False, environment=True)
        assert conflicts == ["environment (default/env)"]
