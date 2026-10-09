# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loads the NeMo Compose services extension into the stock OpenSandbox server.

The extension ships as source files in a ConfigMap mounted at a directory the server's Helm
values put on ``PYTHONPATH``, so Python imports this module when any process in the image
starts. It only acts in the server process (``opensandbox-server``), and stops that process if
the extension can't load, rather than serving sandboxes without the services.
"""

import os
import sys

if os.path.basename(sys.argv[0]) == "opensandbox-server":
    try:
        import nemo_ext_server

        nemo_ext_server.install(sys.argv)
    except BaseException as exc:
        # Python reports and ignores other exceptions from sitecustomize; SystemExit stops it.
        sys.exit(f"nemo-ext: not starting without the compose services extension: {exc!r}")
