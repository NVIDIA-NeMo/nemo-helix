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

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, Union, Optional
from typing_extensions import Annotated, TypeAlias

from ..._utils import PropertyInfo
from ..._models import BaseModel
from .regex_detection import RegexDetection
from .fiddler_guardrails import FiddlerGuardrails
from .g_li_ner_detection import GLiNERDetection
from .pangea_rail_config import PangeaRailConfig
from .polygraf_detection import PolygrafDetection
from .clavata_rail_config import ClavataRailConfig
from .injection_detection import InjectionDetection
from .patronus_rail_config import PatronusRailConfig
from .private_ai_detection import PrivateAIDetection
from .content_safety_config import ContentSafetyConfig
from .ai_defense_rail_config import AIDefenseRailConfig
from .auto_align_rail_config import AutoAlignRailConfig
from .trend_micro_rail_config import TrendMicroRailConfig
from .sensitive_data_detection import SensitiveDataDetection
from .fact_checking_rail_config import FactCheckingRailConfig
from .guardrails_ai_rail_config import GuardrailsAIRailConfig
from .jailbreak_detection_config import JailbreakDetectionConfig
from .local_hf_classifier_config import LocalHfClassifierConfig
from .remote_hf_classifier_config import RemoteHfClassifierConfig
from .crowd_strike_aidr_rail_config import CrowdStrikeAidrRailConfig
from .context_bloat_detection_config import ContextBloatDetectionConfig

__all__ = ["RailsConfigData", "HfClassifier"]

HfClassifier: TypeAlias = Annotated[
    Union[LocalHfClassifierConfig, RemoteHfClassifierConfig], PropertyInfo(discriminator="engine")
]


class RailsConfigData(BaseModel):
    """Configuration data for specific rails that are supported out-of-the-box."""

    ai_defense: Optional[AIDefenseRailConfig] = None
    """Configuration data for the Cisco AI Defense API"""

    autoalign: Optional[AutoAlignRailConfig] = None
    """Configuration data for the AutoAlign API"""

    clavata: Optional[ClavataRailConfig] = None
    """Configuration data for the Clavata API"""

    content_safety: Optional[ContentSafetyConfig] = None
    """Configuration data for content safety rails."""

    context_bloat_detection: Optional[ContextBloatDetectionConfig] = None
    """Configuration for context bloat / context manipulation detection."""

    crowdstrike_aidr: Optional[CrowdStrikeAidrRailConfig] = None
    """Configuration data for the CrowdStrike AIDR API"""

    fact_checking: Optional[FactCheckingRailConfig] = None
    """Configuration data for the fact-checking rail."""

    fiddler: Optional[FiddlerGuardrails] = None
    """Configuration for Fiddler Guardrails."""

    gliner: Optional[GLiNERDetection] = None
    """Configuration for GLiNER PII detection."""

    guardrails_ai: Optional[GuardrailsAIRailConfig] = None
    """Configuration data for Guardrails AI integration."""

    hf_classifier: Optional[Dict[str, HfClassifier]] = None
    """Named HF classifier configurations.

    Keys are classifier names referenced by flows.
    """

    injection_detection: Optional[InjectionDetection] = None
    """Configuration for injection detection."""

    jailbreak_detection: Optional[JailbreakDetectionConfig] = None
    """Configuration data for jailbreak detection."""

    pangea: Optional[PangeaRailConfig] = None
    """Configuration data for the Pangea AI Guard API"""

    patronus: Optional[PatronusRailConfig] = None
    """Configuration data for the Patronus Evaluate API"""

    polygraf: Optional[PolygrafDetection] = None
    """Configuration for Polygraf PII detection."""

    privateai: Optional[PrivateAIDetection] = None
    """Configuration for Private AI."""

    regex_detection: Optional[RegexDetection] = None
    """Configuration for regex pattern detection."""

    sensitive_data_detection: Optional[SensitiveDataDetection] = None
    """Configuration of what sensitive data should be detected."""

    trend_micro: Optional[TrendMicroRailConfig] = None
    """Configuration data for the Trend Micro AI Guard API"""
