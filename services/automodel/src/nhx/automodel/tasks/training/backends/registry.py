# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether an architecture has an Automodel implementation.

``MODEL_ARCH_MAPPING`` matches the Automodel pin in
``docker/automodel/Dockerfile.nhx-automodel-base`` (r0.6.0 @
``b916107a5ce8e77a3f18682f8b0eaa46a4b54389``),
``nemo_automodel/_transformers/registry.py``. Re-copy it when that pin moves.
The API compiler image does not install ``nemo_automodel``.
"""

from collections import OrderedDict

# Static mapping: architecture name → (module_path, class_name[, tags]).
# Analogous to HuggingFace transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.
# Models are loaded lazily on first access rather than imported at startup.
# Optional third element is a set of tags (e.g. {"retrieval"}) used by
# downstream code to classify model archs without importing them.
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
            ("nemo_automodel.components.models.deepseek_v4.model", "DeepseekV4ForCausalLM"),
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
            "Gemma4ForConditionalGeneration",
            ("nemo_automodel.components.models.gemma4_moe.model", "Gemma4ForConditionalGeneration"),
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
            "NemotronHForCausalLM",
            ("nemo_automodel.components.models.nemotron_v3.model", "NemotronHForCausalLM"),
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
            ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForCausalLM"),
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForConditionalGeneration"),
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
