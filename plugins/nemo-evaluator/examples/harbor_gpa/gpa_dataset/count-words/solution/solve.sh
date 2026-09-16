#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cd /app/notes && for f in $(ls | sort); do echo "$f: $(wc -w < "$f")"; done > /app/word_counts.txt
