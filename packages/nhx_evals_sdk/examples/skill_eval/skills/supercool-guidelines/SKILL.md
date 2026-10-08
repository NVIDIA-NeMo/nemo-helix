---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: supercool-guidelines
description: The Supercool Coding Guidelines — mandatory naming conventions for every function you write. Use this whenever you are asked to write or edit a Python function.
---

# Supercool Coding Guidelines

Every function you write MUST follow these conventions:

1. The function name MUST start with the prefix `supercool_`
   (for example `supercool_compute`, never `compute`).
2. Parameter names MUST be French words, written in ASCII (no accents).
   In particular, a positional-index parameter MUST be named `enieme`
   (French for "nth") — never `n`, `index`, or `i`.

These conventions are mandatory and are not optional style suggestions.

## Example

```python
def supercool_racine(nombre):
    return nombre ** 0.5

def supercool_pi(enieme):
    ...
```
