---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: Stop and clean up the NeMo Helix quickstart environment (removes volumes)
---

In addition to the instructions in the quickstart-down command,
we can also remove the sqlite database, encryption key, and uploaded files

```bash
  rm -rf ~/.local/share/nemo
```
