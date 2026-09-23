# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for models service."""

from .utils import ModelConfigParseError as ModelConfigParseError
from .utils import ModelWeightsType as ModelWeightsType
from .utils import get_deployment_resource_name as get_deployment_resource_name
from .utils import get_deployment_secret_name as get_deployment_secret_name
from .utils import get_model_weights_type as get_model_weights_type
from .utils import get_nimcache_resource_name as get_nimcache_resource_name
from .utils import is_multi_llm_image as is_multi_llm_image
from .utils import normalize_model_entity_name as normalize_model_entity_name
from .utils import parse_model_name_revision as parse_model_name_revision
