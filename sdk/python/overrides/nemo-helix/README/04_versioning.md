<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Versioning

This package generally follows [SemVer](https://semver.org/spec/v2.0.0.html) conventions, while it might release certain backwards-incompatible changes as minor versions:

1. Changes that only affect static types, without breaking runtime behavior.
2. Changes to library internals which are technically public but not intended or documented for external use. Open a GitHub issue to let us know if you are relying on such internals.
3. Changes that we do not expect to impact the vast majority of users in practice.

We take backwards-compatibility seriously and work hard to ensure you can rely on a smooth upgrade experience.

We welcome your feedback; please contact us with questions, bugs, or suggestions.

### Determining the Installed Version

If you've upgraded to the latest version but can't find any new features you were expecting, your Python environment is likely still using an older version.

You can determine the version that is being used at runtime with:

```py
import nemo_helix
print(nemo_helix.__version__)
```

## Requirements

Python 3.11 or higher.
