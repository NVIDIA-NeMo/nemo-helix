<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Intentional Harbor failure fixture for exception propagation testing.

Attempt to create hello.txt with "Hello, world!" as the content.

The reference solution script deliberately sleeps past the 1s agent timeout so
Harbor records exception_info.
