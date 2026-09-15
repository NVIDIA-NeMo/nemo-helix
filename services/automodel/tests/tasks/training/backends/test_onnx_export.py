# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX export vs HuggingFace parity. Skipped unless torch/onnxruntime are installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

from nmp.automodel.tasks.training.backends.checkpoints import (  # noqa: E402
    ModelType,
    export_onnx,
)
from nmp.automodel.tasks.training.schemas import ExportConfig  # noqa: E402
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

pytestmark = [pytest.mark.gpu_integration, pytest.mark.slow]

HIDDEN_SIZE = 32
_VOCAB = "[PAD] [UNK] [CLS] [SEP] [MASK] hello world query passage question example sentence for tracing an :".split()


def _write_checkpoint(path, *, cross_encoder: bool):
    """Write a tiny BERT checkpoint (no network)."""
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

    path.mkdir(parents=True, exist_ok=True)
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB) + "\n")

    # transformers>=5 synthesizes config __init__ from class attributes, so from_dict keeps this typeable.
    config = BertConfig.from_dict(
        {
            "vocab_size": len(_VOCAB),
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "max_position_embeddings": 64,
        }
    )
    if cross_encoder:
        config.num_labels = 1
        model = BertForSequenceClassification(config)
    else:
        model = BertModel(config)

    torch.manual_seed(0)
    model.save_pretrained(path)
    BertTokenizer(vocab_file=str(vocab_file)).save_pretrained(path)
    return path


@pytest.fixture
def embedding_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path / "embed", cross_encoder=False)


@pytest.fixture
def cross_encoder_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path / "rerank", cross_encoder=True)


def _session(onnx_path):
    return onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _tokenize(checkpoint, texts):
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    assert tokenizer is not None
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)


def _hf_pooled_embeddings(checkpoint, batch, normalize: bool = True):
    """Pooled HuggingFace embeddings for the given batch."""
    model = AutoModel.from_pretrained(str(checkpoint), torch_dtype=torch.float32).eval()
    with torch.no_grad():
        hidden = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state

    mask = batch["attention_mask"]
    masked = hidden.masked_fill(~mask[..., None].bool(), 0.0)
    pooled = masked.sum(dim=1) / mask.sum(dim=1)[..., None]
    if normalize:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.numpy()


class TestEmbeddingExport:
    def test_onnx_output_matches_hf_output(self, embedding_checkpoint, tmp_path):
        out = tmp_path / "out-embed"
        onnx_path = export_onnx(
            model_path=embedding_checkpoint,
            output_path=out,
            tokenizer_path=str(embedding_checkpoint),
            model_type=ModelType.EMBEDDING,
            cfg=ExportConfig(),
        )

        # Different batch/sequence dimensions from the trace sample exercise dynamic axes.
        batch = _tokenize(embedding_checkpoint, ["hello", "world", "an example sentence for tracing"])
        expected = _hf_pooled_embeddings(embedding_checkpoint, batch)
        session = _session(onnx_path)
        actual = session.run(
            ["embeddings"],
            {
                "input_ids": batch["input_ids"].numpy(),
                "attention_mask": batch["attention_mask"].numpy(),
            },
        )[0]

        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-4)
        assert [inp.name for inp in session.get_inputs()] == ["input_ids", "attention_mask"]
        assert [result.name for result in session.get_outputs()] == ["embeddings"]
        assert (out / "tokenizer").is_dir()

    def test_dimensions_input_truncates_and_renormalizes(self, embedding_checkpoint, tmp_path):
        """`dimensions` truncates then L2-renormalizes."""
        onnx_path = export_onnx(
            model_path=embedding_checkpoint,
            output_path=tmp_path / "out-dims",
            tokenizer_path=str(embedding_checkpoint),
            model_type=ModelType.EMBEDDING,
            cfg=ExportConfig(dimensions=True),
        )

        session = _session(onnx_path)
        assert [inp.name for inp in session.get_inputs()] == ["input_ids", "attention_mask", "dimensions"]

        batch = _tokenize(embedding_checkpoint, ["hello world", "an example sentence for tracing"])
        keep = 8
        actual = session.run(
            ["embeddings"],
            {
                "input_ids": batch["input_ids"].numpy(),
                "attention_mask": batch["attention_mask"].numpy(),
                "dimensions": np.array([keep, keep], dtype=np.int64),
            },
        )[0]

        assert actual.shape == (2, HIDDEN_SIZE)
        assert np.count_nonzero(actual[:, keep:]) == 0
        np.testing.assert_allclose(np.linalg.norm(actual[:, :keep], axis=1), 1.0, atol=1e-4)

        full = _hf_pooled_embeddings(embedding_checkpoint, batch, normalize=False)[:, :keep]
        full = full / np.linalg.norm(full, axis=1, keepdims=True)
        np.testing.assert_allclose(actual[:, :keep], full, atol=1e-4)


class TestCrossEncoderExport:
    def test_onnx_logits_match_hf_logits(self, cross_encoder_checkpoint, tmp_path):
        out = tmp_path / "out-rerank"
        onnx_path = export_onnx(
            model_path=cross_encoder_checkpoint,
            output_path=out,
            tokenizer_path=str(cross_encoder_checkpoint),
            model_type=ModelType.CROSS_ENCODER,
            cfg=ExportConfig(),
        )

        pairs = [
            "question:hello \n \n passage:world",
            "question:an example \n \n passage:sentence for tracing",
        ]
        batch = _tokenize(cross_encoder_checkpoint, pairs)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(cross_encoder_checkpoint), torch_dtype=torch.float32
        ).eval()
        with torch.no_grad():
            expected = model(**batch).logits.numpy()

        feed = {name: tensor.numpy() for name, tensor in batch.items()}
        session = _session(onnx_path)
        graph_inputs = {inp.name for inp in session.get_inputs()}
        actual = session.run(["logits"], {k: v for k, v in feed.items() if k in graph_inputs})[0]

        assert actual.shape == expected.shape == (2, 1)
        np.testing.assert_allclose(actual, expected, atol=1e-4)
        assert [inp.name for inp in session.get_inputs()] == ["input_ids", "attention_mask", "token_type_ids"]
        assert [result.name for result in session.get_outputs()] == ["logits"]
        assert (out / "tokenizer").is_dir()
