# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the spike's transpile module importable when tests run from any cwd."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
