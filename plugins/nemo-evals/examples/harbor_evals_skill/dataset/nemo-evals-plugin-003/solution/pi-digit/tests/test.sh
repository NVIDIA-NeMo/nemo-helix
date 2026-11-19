#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
answer_file=${1:-/workspace/pi_digit.txt}
reward_file=${2:-/logs/verifier/reward.txt}
mkdir -p "$(dirname "$reward_file")"
python3 - "$answer_file" "$reward_file" <<'PY'
import os
import stat
import sys
from pathlib import Path

try:
    fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as answer:
        passed = stat.S_ISREG(os.fstat(answer.fileno()).st_mode) and answer.read(3) in (b"9", b"9\n")
except OSError:
    passed = False
Path(sys.argv[2]).write_text("1\n" if passed else "0\n", encoding="ascii")
PY
