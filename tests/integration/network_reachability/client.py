# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One real HTTP request from the workload namespace, with proxies disabled."""

import json
import os
import urllib.error
import urllib.request

request = urllib.request.Request(
    os.environ["PROBE_URL"],
    data=b'{"probe":"network-reachability"}',
    headers={"Content-Type": "application/json", "Authorization": "Bearer probe-only-not-a-credential"},
)
try:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=3) as response:
        result = {"status": response.status, "received": json.load(response)}
except urllib.error.URLError as error:
    if isinstance(error, urllib.error.HTTPError):
        raise
    result = {"network_error": type(error.reason).__name__, "detail": str(error.reason)}
print(json.dumps(result), flush=True)
