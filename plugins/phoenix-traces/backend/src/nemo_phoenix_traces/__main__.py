# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit operator launch, not a nemo.services entry point."""

import argparse
import os

import uvicorn
from nemo_phoenix_traces.app import Settings, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    args = parser.parse_args()
    # The public URL can point at a TLS reverse proxy. Never infer it from Host.
    if "PHOENIX_PUBLIC_URL" not in os.environ:
        os.environ["PHOENIX_PUBLIC_URL"] = f"http://127.0.0.1:{args.port}"
    uvicorn.run(create_app(Settings()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
