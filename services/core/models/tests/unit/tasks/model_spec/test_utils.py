# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Optional

import pytest
from nhx.core.models.tasks.model_spec.utils import infer_model_head_type, is_embedding_model_v2

TEST_MODEL_FIXTURES_DIR = Path(__file__).parent / "test_data" / "models"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_readme_frontmatter(
    path: Path, *, pipeline_tag: Optional[str] = None, tags: Optional[list[str]] = None
) -> None:
    frontmatter_lines = ["---"]
    if pipeline_tag is not None:
        frontmatter_lines.append(f"pipeline_tag: {pipeline_tag}")
    if tags is not None:
        frontmatter_lines.append("tags:")
        frontmatter_lines.extend([f"  - {tag}" for tag in tags])
    frontmatter_lines.append("---")
    frontmatter_lines.append("model card body")
    path.write_text("\n".join(frontmatter_lines), encoding="utf-8")


def _model_fixture_dir(repo_id: str) -> Path:
    return TEST_MODEL_FIXTURES_DIR / repo_id.replace("/", "__")


class TestIsEmbeddingModelV2:
    def test_returns_false_for_missing_directory(self) -> None:
        is_embedding, reason = is_embedding_model_v2("/tmp/non-existent-model-dir")
        assert is_embedding is False
        assert "Directory not found" in reason

    def test_returns_true_when_sentence_transformers_artifact_exists(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "modules.json").write_text("{}", encoding="utf-8")
        _write_json(model_dir / "config.json", {"architectures": ["Qwen3ForCausalLM"]})

        is_embedding, reason = is_embedding_model_v2(str(model_dir))

        assert is_embedding is True
        assert reason.startswith("embedding:")
        assert "artifact:modules.json" in reason

    def test_returns_true_for_embedding_pipeline_tag(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(model_dir / "config.json", {"architectures": ["LlamaForCausalLM"]})
        _write_readme_frontmatter(model_dir / "README.md", pipeline_tag="feature-extraction")

        is_embedding, reason = is_embedding_model_v2(str(model_dir))

        assert is_embedding is True
        assert "pipeline_tag:feature-extraction" in reason

    def test_returns_false_for_generative_pipeline_tag_and_architecture(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(model_dir / "config.json", {"architectures": ["LlamaForCausalLM"]})
        _write_readme_frontmatter(model_dir / "README.md", pipeline_tag="text-generation")

        is_embedding, reason = is_embedding_model_v2(str(model_dir))

        assert is_embedding is False
        assert reason.startswith("causal_lm:")
        assert "pipeline_tag:text-generation" in reason
        assert "architecture:LlamaForCausalLM" in reason

    def test_returns_false_for_generative_architecture_without_positive_signals(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(model_dir / "config.json", {"architectures": ["Qwen2ForConditionalGeneration"]})

        is_embedding, reason = is_embedding_model_v2(str(model_dir))

        assert is_embedding is False
        assert reason.startswith("causal_lm:")
        assert "Qwen2ForConditionalGeneration" in reason

    def test_returns_false_for_encoder_without_embedding_signals(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(model_dir / "config.json", {"architectures": ["BertModel"]})

        is_embedding, reason = is_embedding_model_v2(str(model_dir))

        assert is_embedding is False
        assert reason == "inconclusive:no_strong_head_signals"


class TestInferModelHeadType:
    def test_detects_cross_encoder_from_sequence_classification_head(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(
            model_dir / "config.json",
            {
                "architectures": ["LlamaBidirectionalForSequenceClassification"],
                "num_labels": 1,
            },
        )

        head_type, reason = infer_model_head_type(str(model_dir))

        assert head_type == "cross_encoder"
        assert "ForSequenceClassification" in reason

    def test_detects_automodel_bi_encoder_architecture(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(model_dir / "config.json", {"architectures": ["LlamaBidirectionalModel"]})

        head_type, reason = infer_model_head_type(str(model_dir))

        assert head_type == "embedding"
        assert "LlamaBidirectionalModel" in reason

    def test_cross_encoder_head_wins_over_embedding_card_metadata(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(
            model_dir / "config.json",
            {"architectures": ["BertForSequenceClassification"], "num_labels": 1},
        )
        _write_readme_frontmatter(model_dir / "README.md", pipeline_tag="feature-extraction")

        head_type, _ = infer_model_head_type(str(model_dir))

        assert head_type == "cross_encoder"

    def test_generic_sequence_classifier_is_not_assumed_to_be_a_reranker(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(
            model_dir / "config.json",
            {"architectures": ["BertForSequenceClassification"], "num_labels": 3},
        )
        _write_readme_frontmatter(model_dir / "README.md", pipeline_tag="text-classification")

        head_type, reason = infer_model_head_type(str(model_dir))

        assert head_type == "unknown"
        assert reason == "inconclusive:no_strong_head_signals"

    def test_sequence_classification_auto_map_is_not_enough_without_rerank_evidence(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        _write_json(
            model_dir / "config.json",
            {
                "architectures": ["BertModel"],
                "num_labels": 3,
                "auto_map": {"AutoModelForSequenceClassification": "modeling.CustomClassifier"},
            },
        )

        head_type, reason = infer_model_head_type(str(model_dir))

        assert head_type == "unknown"
        assert "auto_map:AutoModelForSequenceClassification" not in reason

    @pytest.mark.parametrize(
        "repo_id, expected",
        [
            ("nvidia/llama-nemotron-embed-1b-v2", True),
            ("BAAI/bge-base-en-v1.5", True),
            ("google/embeddinggemma-300m", True),
            ("Qwen/Qwen3-Embedding-8B", True),
            ("deepseek-ai/DeepSeek-V3.2", False),
            ("Qwen/Qwen3-8B", False),
        ],
        ids=[
            "llama_nemotron_embed_1b_v2",
            "bge_base_en_v1_5",
            "embeddinggemma_300m",
            "qwen3_embedding_8b",
            "deepseek_v3_2_negative_control",
            "qwen3_8b_negative_control",
        ],
    )
    def test_curated_huggingface_model_fixtures(self, repo_id: str, expected: bool) -> None:
        local_model_dir = _model_fixture_dir(repo_id)
        assert local_model_dir.is_dir(), f"Missing fixture directory for {repo_id}: {local_model_dir}"

        is_embedding, reason = is_embedding_model_v2(str(local_model_dir))

        assert is_embedding is expected, f"{repo_id} expected {expected}, got {is_embedding}; reason={reason}"
