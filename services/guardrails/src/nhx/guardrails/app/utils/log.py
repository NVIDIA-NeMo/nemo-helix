# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging.config
import os


def configure_logging(logging_config: dict, log_dir: str = "logs"):
    """Configure logging for the application.

    If the log directory does not exist, it will be created.

    Args:
        logging_config (dict): The logging configuration dictionary.
        log_dir (str): The directory where log files will be stored.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.config.dictConfig(logging_config)
