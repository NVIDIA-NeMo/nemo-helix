# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a sandbox's execd learns its access token, and how a client presents it.

execd, OpenSandbox's command server in every sandbox, refuses any request without the token when
``EXECD_ACCESS_TOKEN`` is set in the sandbox's environment. The driver sets a fresh one per sandbox and
sends it on that sandbox's own SDK connection. The Gym host runtime removes it from its environment at
startup, so nothing the host starts can read it.
"""

#: Environment variable execd reads its token from.
EXECD_ACCESS_TOKEN_ENV = "EXECD_ACCESS_TOKEN"

#: Request header execd checks the token against.
EXECD_ACCESS_TOKEN_HEADER = "X-EXECD-ACCESS-TOKEN"
