# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml
from nmp.automodel.tasks.retrieval_mine.runner import RetrievalMineJobConfig, RetrievalMiningOptions, run_mine


def test_run_mine_launches_torchrun_then_unrolls(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage1_data_prep"
    output_dir.mkdir()
    (tmp_path / "model").mkdir()
    (output_dir / "train.json").write_text(
        json.dumps(
            {"corpus": {}, "data": [{"question_id": "q1", "question": "q", "corpus_id": "c", "pos_doc": ["p"]}]}
        ),
        encoding="utf-8",
    )
    ctx = Mock()
    ctx.workspace = "default"
    ctx.results.save.return_value = SimpleNamespace(model_dump=lambda: {"name": "artifacts"})

    def _fake_mine(**kwargs: object) -> None:
        config = yaml.safe_load(Path(str(kwargs["config_file"])).read_text(encoding="utf-8"))
        assert config["dist_env"] == {"backend": "nccl", "timeout_minutes": 30}
        assert config["mining"]["mining_batch_size"] == 8
        assert config["mining"]["corpus_chunk_size"] == 1024
        assert config["mining"]["model_name_or_path"] == str(tmp_path / "model")
        assert config["mining"]["tokenizer_name_or_path"] == str(tmp_path / "model")
        assert config["mining"]["trust_remote_code"] is True
        assert config["mining"]["train_qa_file_path"] == str(output_dir / "train.json")
        output_file = Path(config["mining"]["train_file_output_path"])
        output_file.write_text(
            json.dumps(
                {
                    "corpus": {},
                    "data": [
                        {
                            "question_id": "q1",
                            "question": "q",
                            "corpus_id": "c",
                            "pos_doc": ["p"],
                            "neg_doc": ["n"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    job = RetrievalMineJobConfig(mining_batch_size=8, mining=RetrievalMiningOptions(corpus_chunk_size=1024))
    with patch("nmp.automodel.tasks.retrieval_mine.runner.run_hard_negative_mining", side_effect=_fake_mine) as mine:
        result = run_mine(job, output_dir, ctx, model_trust_remote_code=True)

    mine.assert_called_once()
    assert result["exit_code"] == 0
    assert (output_dir / "training.jsonl").exists()
    assert (output_dir / "additional" / "train.json").exists()
    assert (output_dir / "additional" / "train_mined.automodel.json").exists()
    assert (output_dir / "additional" / "mining_config.yaml").exists()
    assert not (output_dir / "train.json").exists()
    ctx.results.save.assert_called_once()


def test_run_mine_selects_shallowest_train_file_deterministically(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage1_data_prep"
    shallow = output_dir / "a" / "train.json"
    deep = output_dir / "0" / "nested" / "train.json"
    shallow.parent.mkdir(parents=True)
    deep.parent.mkdir(parents=True)
    shallow.write_text("{}", encoding="utf-8")
    deep.write_text("{}", encoding="utf-8")
    (tmp_path / "model").mkdir()

    ctx = Mock()
    ctx.workspace = "default"
    ctx.results.save.return_value = SimpleNamespace(model_dump=lambda: {"name": "artifacts"})

    def _fake_mine(**kwargs: object) -> None:
        config = yaml.safe_load(Path(str(kwargs["config_file"])).read_text(encoding="utf-8"))
        assert config["mining"]["train_qa_file_path"] == str(shallow)
        Path(config["mining"]["train_file_output_path"]).write_text(
            json.dumps(
                {
                    "corpus": {},
                    "data": [
                        {
                            "question_id": "q1",
                            "question": "q",
                            "corpus_id": "c",
                            "pos_doc": ["p"],
                            "neg_doc": ["n"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    with patch("nmp.automodel.tasks.retrieval_mine.runner.run_hard_negative_mining", side_effect=_fake_mine):
        run_mine(RetrievalMineJobConfig(), output_dir, ctx, model_trust_remote_code=False)


def test_run_mine_rejects_empty_training_jsonl(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage1_data_prep"
    output_dir.mkdir()
    (tmp_path / "model").mkdir()
    (output_dir / "train.json").write_text(json.dumps({"corpus": {}, "data": []}), encoding="utf-8")
    ctx = Mock()
    ctx.workspace = "default"
    ctx.results.save.return_value = SimpleNamespace(model_dump=lambda: {"name": "artifacts"})

    def _fake_mine(**kwargs: object) -> None:
        config = yaml.safe_load(Path(str(kwargs["config_file"])).read_text(encoding="utf-8"))
        Path(config["mining"]["train_file_output_path"]).write_text(
            json.dumps({"corpus": {}, "data": []}),
            encoding="utf-8",
        )

    with (
        patch("nmp.automodel.tasks.retrieval_mine.runner.run_hard_negative_mining", side_effect=_fake_mine),
        pytest.raises(ValueError, match="No training rows"),
    ):
        run_mine(RetrievalMineJobConfig(), output_dir, ctx, model_trust_remote_code=False)
    ctx.results.save.assert_not_called()
