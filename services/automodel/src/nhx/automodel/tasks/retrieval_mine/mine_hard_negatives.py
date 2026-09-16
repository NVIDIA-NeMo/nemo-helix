# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hard-negative mining recipe: embed queries/passages and select difficult negatives."""

from __future__ import annotations

from nemo_automodel._transformers.auto_model import NeMoAutoModelBiEncoder
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.retrieval import mine_hard_negatives as automodel_mining


class MineHardNegativesRecipe(automodel_mining.MineHardNegativesRecipe):
    """Mine hard negatives, including encoders that require ``trust_remote_code``."""

    def setup(self):
        self.dist_env = automodel_mining.build_distributed(self.cfg.get("dist_env", {}))
        self.mining_cfg = self.cfg.get("mining", None)
        if self.mining_cfg is None:
            raise ValueError("Missing mining configuration")

        self._extract_mining_params()
        self._validate_mining_params()

        model_kwargs = {
            "use_liger_kernel": False,
            "use_sdpa_patching": True,
            "trust_remote_code": self._get_mining_param("trust_remote_code", False),
        }
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        automodel_mining.logger.info(f"Loading encoder model from {self.model_name_or_path}...")
        self.model = NeMoAutoModelBiEncoder.from_pretrained(self.model_name_or_path, **model_kwargs)
        self.model = self.model.to(self.dist_env.device)
        self.model.eval()
        self._configure_tokenizer()
        self._load_data()
        self._build_document_mappings()
        self._prepare_data()


def main() -> None:
    cfg = parse_args_and_load_config()
    recipe = MineHardNegativesRecipe(cfg)
    recipe.setup()
    recipe.run()


if __name__ == "__main__":
    main()
