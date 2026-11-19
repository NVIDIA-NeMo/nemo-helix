# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hard-negative mining recipe: embed queries/passages and select difficult negatives."""

from __future__ import annotations

import logging
import sys

from nhx.automodel.tasks.tqdm_logging import install_line_tqdm

install_line_tqdm()

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config  # noqa: E402
from nemo_automodel.recipes.retrieval import mine_hard_negatives as automodel_mining  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
)


def main() -> None:
    cfg = parse_args_and_load_config()
    # Automodel's recipe passes mining.trust_remote_code to both the encoder and its tokenizer.
    recipe = automodel_mining.MineHardNegativesRecipe(cfg)
    recipe.setup()
    recipe.run()
    automodel_mining.logger.info("Hard-negative mining recipe finished")


if __name__ == "__main__":
    main()
