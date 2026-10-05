# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging


def configure_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    if verbosity > 0:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logger = logging.getLogger("nemo_helix_ext.cli")
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("nemo_helix_ext.cli").setLevel(logging.WARNING)
