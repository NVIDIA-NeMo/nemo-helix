# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hard-negative mining for retrieval training data.

Given converted ``train.json``, this task mines hard negatives with a bi-encoder,
unrolls multi-positive records, and writes inline ``training.jsonl``.
"""
