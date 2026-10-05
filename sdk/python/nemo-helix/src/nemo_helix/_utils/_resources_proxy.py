# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from typing_extensions import override

from ._proxy import LazyProxy


class ResourcesProxy(LazyProxy[Any]):
    """A proxy for the `nemo_helix.resources` module.

    This is used so that we can lazily import `nemo_helix.resources` only when
    needed *and* so that users can just import `nemo_helix` and reference `nemo_helix.resources`
    """

    @override
    def __load__(self) -> Any:
        import importlib

        mod = importlib.import_module("nemo_helix.resources")
        return mod


resources = ResourcesProxy().__as_proxied__()
