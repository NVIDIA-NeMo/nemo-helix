# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The metric image every shard runs in. The scorer needs only the standard library. Every shard's build context is
# this directory, unchanged, so the whole dataset shares one build.
FROM public.ecr.aws/docker/library/python:3.13-slim

COPY cookbook.py scoring.py score.py /opt/metric/
WORKDIR /opt/metric
