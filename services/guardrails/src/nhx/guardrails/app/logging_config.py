# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# logging configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(levelname)s: %(message)s"},
        "verbose": {
            "format": "%(asctime)s %(levelname)s: %(name)s - %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%SZ",
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"message": "%(message)s", '
            '"name": "%(name)s", "path": "%(pathname)s", "line_no": "%(lineno)d"}',
            "datefmt": "%Y-%m-%dT%H:%M:%SZ",
        },
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": os.getenv("LOG_LEVEL", LOG_LEVEL),
            "formatter": "simple",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": os.getenv("LOG_LEVEL", LOG_LEVEL),
            "filename": f"{LOG_DIR}/app.log",
            "formatter": "verbose",
        },
        "jsonl_file": {
            "level": LOG_LEVEL,
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": f"{LOG_DIR}/app.log.jsonl",
            "formatter": "json",
        },
    },
    "loggers": {
        "": {
            "level": LOG_LEVEL,
            "handlers": ["console", "file", "jsonl_file"],
        }
    },
}
