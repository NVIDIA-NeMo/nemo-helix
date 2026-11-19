# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which Automodel implementation a checkpoint trains with.

``MODEL_ARCH_MAPPING`` matches the Automodel pin in
``docker/automodel/Dockerfile.nhx-automodel-base`` (v0.7.0 @
``48c88e16798246e8c7f1b9ea3974bfc488bd0849``),
``nemo_automodel/_transformers/registry.py``. Re-copy it when that pin moves.
The API compiler image does not install ``nemo_automodel``, so only the causal-reranker helpers,
which run in the training image, import it.
"""

import json
from collections import OrderedDict
from pathlib import Path

# Static mapping: architecture name → (module_path, class_name[, tags]).
# Analogous to HuggingFace transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.
# Models are loaded lazily on first access rather than imported at startup.
# Optional third element is a set of tags (e.g. {"retrieval"}) used by
# downstream code to classify model archs without importing them.
# "hf_parallelizer" opts the HF wrapper into the owning package's
# parallelization.PARALLELIZER; it requires compatibility with the HF layout.
MODEL_ARCH_MAPPING = OrderedDict(
    [
        (
            "BagelForUnifiedMultimodal",
            (
                "nemo_automodel.components.models.bagel.model",
                "BagelForUnifiedMultimodal",
            ),
        ),
        (
            "BagelForConditionalGeneration",
            (
                "nemo_automodel.components.models.bagel.model",
                "BagelForUnifiedMultimodal",
            ),
        ),
        (
            "BaichuanForCausalLM",
            ("nemo_automodel.components.models.baichuan.model", "BaichuanForCausalLM"),
        ),
        (
            "BailingMoeV2ForCausalLM",
            ("nemo_automodel.components.models.ling_v2.model", "BailingMoeV2ForCausalLM"),
        ),
        (
            "DeepseekV3ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v3.model", "DeepseekV3ForCausalLM"),
        ),
        (
            "DeepseekV32ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v32.model", "DeepseekV32ForCausalLM"),
        ),
        (
            "DeepseekV4ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v4.model", "DeepseekV4ForCausalLM", {"hf_parallelizer"}),
        ),
        (
            "DeepseekV41ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v41.model", "DeepseekV41ForCausalLM"),
        ),
        (
            "DiffusionGemmaForBlockDiffusion",
            ("nemo_automodel.components.models.diffusion_gemma.model", "DiffusionGemmaForBlockDiffusion"),
        ),
        (
            "Ernie4_5_MoeForCausalLM",
            ("nemo_automodel.components.models.ernie4_5.model", "Ernie4_5_MoeForCausalLM"),
        ),
        (
            "Glm4MoeForCausalLM",
            ("nemo_automodel.components.models.glm4_moe.model", "Glm4MoeForCausalLM"),
        ),
        (
            "Glm4MoeLiteForCausalLM",
            ("nemo_automodel.components.models.glm4_moe_lite.model", "Glm4MoeLiteForCausalLM"),
        ),
        (
            "GlmMoeDsaForCausalLM",
            ("nemo_automodel.components.models.glm_moe_dsa.model", "GlmMoeDsaForCausalLM"),
        ),
        (
            "Glm5NextForConditionalGeneration",
            (
                "nemo_automodel.components.models.glm5_next.model",
                "Glm5NextForConditionalGeneration",
            ),
        ),
        (
            "Gemma4ForConditionalGeneration",
            ("nemo_automodel.components.models.gemma4_moe.model", "Gemma4ForConditionalGeneration"),
        ),
        (
            "Gemma4UnifiedForConditionalGeneration",
            (
                "nemo_automodel.components.models.gemma4_unified.model",
                "Gemma4UnifiedForConditionalGeneration",
            ),
        ),
        (
            "Gemma4AssistantForCausalLM",
            ("nemo_automodel.components.models.gemma4_drafter.model", "Gemma4DrafterForCausalLM"),
        ),
        (
            "GptOssForCausalLM",
            ("nemo_automodel.components.models.gpt_oss.model", "GptOssForCausalLM"),
        ),
        (
            "KimiK25ForConditionalGeneration",
            ("nemo_automodel.components.models.kimi_k25_vl.model", "KimiK25VLForConditionalGeneration"),
        ),
        (
            "KimiK25VLForConditionalGeneration",
            ("nemo_automodel.components.models.kimi_k25_vl.model", "KimiK25VLForConditionalGeneration"),
        ),
        (
            "KimiK3ForConditionalGeneration",
            ("nemo_automodel.components.models.kimi_k3.multimodal", "KimiK3ForConditionalGeneration"),
        ),
        (
            "KimiK3ForCausalLM",
            ("nemo_automodel.components.models.kimi_k3.model", "KimiK3ForCausalLM"),
        ),
        (
            "KimiVLForConditionalGeneration",
            ("nemo_automodel.components.models.kimivl.model", "KimiVLForConditionalGeneration"),
        ),
        (
            "KimiLinear48BForCausalLM",
            ("nemo_automodel.components.models.kimi_linear.model", "KimiLinear48BForCausalLM"),
        ),
        (
            "LlamaBidirectionalForSequenceClassification",
            (
                "nemo_automodel.components.models.llama_bidirectional.model",
                "LlamaBidirectionalForSequenceClassification",
                {"retrieval"},
            ),
        ),
        (
            "LlamaBidirectionalModel",
            ("nemo_automodel.components.models.llama_bidirectional.model", "LlamaBidirectionalModel", {"retrieval"}),
        ),
        (
            "LlamaNemotronVLModel",
            ("nemo_automodel.components.models.llama_nemotron_vl.model", "LlamaNemotronVLModel", {"retrieval"}),
        ),
        (
            "LlamaForCausalLM",
            ("nemo_automodel.components.models.llama.model", "LlamaForCausalLM"),
        ),
        (
            "MiniMaxM2ForCausalLM",
            ("nemo_automodel.components.models.minimax_m2.model", "MiniMaxM2ForCausalLM"),
        ),
        (
            "MiniMaxM3SparseForConditionalGeneration",
            (
                "nemo_automodel.components.models.minimax_m3_vl.model",
                "MiniMaxM3SparseForConditionalGeneration",
            ),
        ),
        (
            "MiMoV2FlashForCausalLM",
            ("nemo_automodel.components.models.mimo_v2_flash.model", "MiMoV2FlashForCausalLM"),
        ),
        (
            "MiMoV2ForCausalLM",
            ("nemo_automodel.components.models.mimo_v2_flash.model", "MiMoV2ForCausalLM"),
        ),
        (
            "MiMoV25ForCausalLM",
            ("nemo_automodel.components.models.mimo_v25.model", "MiMoV2ForCausalLM"),
        ),
        (
            "LagunaForCausalLM",
            ("nemo_automodel.components.models.laguna.model", "LagunaForCausalLM"),
        ),
        (
            "Ministral3ForCausalLM",
            ("nemo_automodel.components.models.mistral3.model", "Ministral3ForCausalLM"),
        ),
        (
            "Ministral3BidirectionalModel",
            (
                "nemo_automodel.components.models.ministral_bidirectional.model",
                "Ministral3BidirectionalModel",
                {"retrieval"},
            ),
        ),
        (
            "Mistral3BidirectionalModel",
            (
                "nemo_automodel.components.models.ministral_bidirectional.model",
                "Mistral3BidirectionalModel",
                {"retrieval"},
            ),
        ),
        (
            "Mistral3VLBidirectionalForSequenceClassification",
            (
                "nemo_automodel.components.models.ministral_bidirectional.model",
                "Mistral3VLBidirectionalForSequenceClassification",
                {"retrieval"},
            ),
        ),
        (
            "Mistral4ForCausalLM",
            ("nemo_automodel.components.models.mistral4.model", "Mistral4ForCausalLM"),
        ),
        (
            "Mistral3ForConditionalGeneration",
            ("nemo_automodel.components.models.mistral4.model", "Mistral3ForConditionalGeneration"),
        ),
        (
            "Mistral3FP8VLMForConditionalGeneration",
            (
                "nemo_automodel.components.models.mistral3_vlm.model",
                "Mistral3FP8VLMForConditionalGeneration",
            ),
        ),
        (
            "Qwen3RerankerForCausalReranking",
            ("nemo_automodel.components.models.qwen3_reranker.model", "Qwen3RerankerForCausalReranking"),
        ),
        (
            "NemotronHForCausalLM",
            ("nemo_automodel.components.models.nemotron_v3.model", "NemotronHForCausalLM", {"hf_parallelizer"}),
        ),
        (
            "NemotronH_Nano_Omni_Reasoning_V3",
            (
                "nemo_automodel.components.models.nemotron_omni.model",
                "NemotronOmniForConditionalGeneration",
            ),
        ),
        (
            "NemotronH_Omni_Reasoning_V3",
            (
                "nemo_automodel.components.models.nemotron_omni.model",
                "NemotronOmniForConditionalGeneration",
            ),
        ),
        (
            "NemotronParseForConditionalGeneration",
            ("nemo_automodel.components.models.nemotron_parse.model", "NemotronParseForConditionalGeneration"),
        ),
        (
            "MuseGlimmerForConditionalGeneration",
            ("nemo_automodel.components.models.muse_glimmer.model", "MuseGlimmerForConditionalGeneration"),
        ),
        (
            "LLaVAOneVision1_5_ForConditionalGeneration",
            (
                "nemo_automodel.components.models.llava_onevision.model",
                "LLaVAOneVision1_5_ForConditionalGeneration",
            ),
        ),
        (
            "HYV3ForCausalLM",
            ("nemo_automodel.components.models.hy_v3.model", "HYV3ForCausalLM"),
        ),
        (
            "HyMT2ForCausalLM",
            ("nemo_automodel.components.models.hy_mt2.model", "HyMT2ForCausalLM"),
        ),
        (
            "HunyuanImage3ForCausalMM",
            ("nemo_automodel.components.models.hunyuan_image3.model", "HunyuanImage3ForCausalMM"),
        ),
        (
            "Qwen2ForCausalLM",
            ("nemo_automodel.components.models.qwen2.model", "Qwen2ForCausalLM"),
        ),
        (
            "Qwen3ForCausalLM",
            ("nemo_automodel.components.models.qwen3.model", "Qwen3ForCausalLM"),
        ),
        (
            "Qwen2_5OmniModel",
            (
                "nemo_automodel.components.models.qwen2_5_omni.model",
                "Qwen2_5OmniThinkerForConditionalGeneration",
            ),
        ),
        (
            "Qwen2_5OmniForConditionalGeneration",
            (
                "nemo_automodel.components.models.qwen2_5_omni.model",
                "Qwen2_5OmniThinkerForConditionalGeneration",
            ),
        ),
        (
            "Qwen2_5OmniThinkerForConditionalGeneration",
            (
                "nemo_automodel.components.models.qwen2_5_omni.model",
                "Qwen2_5OmniThinkerForConditionalGeneration",
            ),
        ),
        (
            "Qwen3MoeForCausalLM",
            ("nemo_automodel.components.models.qwen3_moe.model", "Qwen3MoeForCausalLM"),
        ),
        (
            "Qwen3NextForCausalLM",
            ("nemo_automodel.components.models.qwen3_next.model", "Qwen3NextForCausalLM"),
        ),
        (
            "Qwen3_5ForCausalLM",
            ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForCausalLM", {"hf_parallelizer"}),
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForConditionalGeneration", {"hf_parallelizer"}),
        ),
        (
            "Qwen3OmniMoeForConditionalGeneration",
            (
                "nemo_automodel.components.models.qwen3_omni_moe.model",
                "Qwen3OmniMoeThinkerForConditionalGeneration",
            ),
        ),
        (
            "Qwen3VLMoeForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_vl_moe.model", "Qwen3VLMoeForConditionalGeneration"),
        ),
        (
            "Qwen3VLForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_vl.model", "Qwen3VLForConditionalGeneration"),
        ),
        (
            "Qwen3_5MoeForCausalLM",
            ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForCausalLM"),
        ),
        (
            "Qwen3_5MoeForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForConditionalGeneration"),
        ),
        (
            "Qwen3_8_FlashNextForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_8_flash_next.model", "Qwen3_8_FlashNextForConditionalGeneration"),
        ),
        (
            # Immutable checkpoint dumps predate the Qwen3.8-Flash-Next rename.
            "Qwen4ExpForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_8_flash_next.model", "Qwen3_8_FlashNextForConditionalGeneration"),
        ),
        (
            "Step3p6ForConditionalGeneration",
            ("nemo_automodel.components.models.step3p7.model", "Step3p7ForConditionalGeneration"),
        ),
        (
            "Step3p5ForCausalLM",
            ("nemo_automodel.components.models.step3p5.model", "Step3p5ForCausalLM"),
        ),
        (
            "InklingForConditionalGeneration",
            ("nemo_automodel.components.models.inkling.model", "InklingForConditionalGeneration"),
        ),
        (
            "Step3p7ForConditionalGeneration",
            ("nemo_automodel.components.models.step3p7.model", "Step3p7ForConditionalGeneration"),
        ),
    ]
)


def uses_custom_automodel_impl(architecture: str | None) -> bool:
    """True when Automodel ships its own class for this architecture.

    Membership is the static name map. ``in ModelRegistry.model_arch_name_to_cls``
    imports the model class.
    """
    if not architecture or "/" in architecture:
        return False
    return architecture in MODEL_ARCH_MAPPING


def _automodel_cross_encoder_class(config: dict) -> type | None:
    """The class Automodel's retrieval dispatch builds for a cross-encoder on *config*, if it has one.

    Training image only: it imports ``nemo_automodel`` and ``transformers``.
    """
    from nemo_automodel._transformers.retrieval import _get_supported_backbone_class
    from transformers import PretrainedConfig

    model_type = str(config.get("model_type", ""))
    return _get_supported_backbone_class(model_type, "score", PretrainedConfig.from_dict(config))


def causal_reranker_class(model_path: str | Path) -> type | None:
    """The Automodel class that trains this checkpoint as a yes/no causal reranker, else ``None``.

    Such a reranker scores with ``logit("yes") - logit("no")`` from the language-model head instead
    of adding a score head. It rejects the classifier ``pooling`` and a model-level ``temperature``,
    and it saves a plain causal LM. Automodel decides which checkpoints train this way: its
    ``SUPPORTED_BACKBONES`` table maps the model type to a class, and that class's
    ``supports_config`` accepts or declines the checkpoint (``Qwen3RerankerForCausalReranking``
    declines an already-converted ``Qwen3ForSequenceClassification``). Asking that dispatch means a
    causal reranker Automodel adds later is found with no change here. A generative class is the
    sign, the same test Automodel uses to keep such a backbone's language-model export
    (``CrossEncoderModel._get_consolidated_hf_metadata_exporter``).
    """
    try:
        config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    backbone = _automodel_cross_encoder_class(config)
    return backbone if backbone is not None and backbone.can_generate() else None


def is_causal_reranker_checkpoint(model_path: str | Path) -> bool:
    """True when Automodel trains this checkpoint as a yes/no causal reranker (see ``causal_reranker_class``)."""
    return causal_reranker_class(model_path) is not None


# The training collator for each causal reranker class; see ``causal_reranker_collator``.
_CAUSAL_RERANKER_COLLATORS = {
    "Qwen3RerankerForCausalReranking": (
        "nemo_automodel.components.models.qwen3_reranker.collator.Qwen3ContextAwareRerankerCollator"
    ),
}


def causal_reranker_collator(reranker: type) -> str:
    """The collator that writes training pairs in *reranker*'s own prompt.

    A causal reranker reads its yes/no answer right after the prompt it was trained on: Qwen3-Reranker's
    system message, ``<Instruct>``/``<Query>``/``<Document>``, then an empty think block. A different
    prompt reads the score at the wrong place and no longer matches vLLM's Qwen3 reranker template.

    Automodel's code can't supply this choice. Its retrieval dispatch maps a checkpoint to a model class
    through a hardcoded table (``SUPPORTED_BACKBONES`` in ``nemo_automodel/_transformers/retrieval.py``),
    but the class carries no collator. Automodel's own Qwen3 recipe hardcodes the collator in its YAML
    (``examples/retrieval/cross_encoder/qwen3_4b_reranker_agentir.yaml``, ``collate_fn._target_``), and
    Helix's job API has no collator field, so Helix maps each class to its collator here.

    Raises:
        ValueError: For a causal reranker class with no known collator. Training it on another
            reranker's prompt would raise no error and score wrongly, so the job stops at setup.
    """
    try:
        return _CAUSAL_RERANKER_COLLATORS[reranker.__name__]
    except KeyError:
        raise ValueError(
            f"Automodel trains this checkpoint as a {reranker.__name__} yes/no reranker, and Helix has no "
            "training prompt for that class yet, so it can't train it with recipe cross_encoder."
        ) from None
