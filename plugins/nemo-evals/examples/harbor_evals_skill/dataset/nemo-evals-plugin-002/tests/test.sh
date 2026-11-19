#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu
exec /usr/local/bin/python -I /tests/grade.py /var/lib/helix-task/evidence /logs/verifier/reward.json
