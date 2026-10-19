# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to parse the output of bake --print and generate a list of nhx images
"""

import json
import sys

if __name__ == "__main__":
    count = 0
    items = []
    for value in json.load(sys.stdin)["target"].values():
        tags = value.get("tags", [])
        if len(tags) > 0:
            tag = tags[0]
            service = tag.split(":")[1].split("/")[-1]
            items.append((service, tag, count))
            count += 1
    for service, tag, count in items:
        print(service, tag, count + 1, len(items))
