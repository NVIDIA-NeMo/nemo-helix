# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib


def compute_token_headers_hash(token):
    """Compute a consistent hash of the token using SHA-256."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
