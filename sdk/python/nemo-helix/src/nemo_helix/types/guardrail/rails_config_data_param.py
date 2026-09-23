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

from __future__ import annotations

from typing import Dict, Union
from typing_extensions import TypeAlias, TypedDict

from .regex_detection_param import RegexDetectionParam
from .fiddler_guardrails_param import FiddlerGuardrailsParam
from .g_li_ner_detection_param import GLiNERDetectionParam
from .pangea_rail_config_param import PangeaRailConfigParam
from .polygraf_detection_param import PolygrafDetectionParam
from .clavata_rail_config_param import ClavataRailConfigParam
from .injection_detection_param import InjectionDetectionParam
from .patronus_rail_config_param import PatronusRailConfigParam
from .private_ai_detection_param import PrivateAIDetectionParam
from .content_safety_config_param import ContentSafetyConfigParam
from .ai_defense_rail_config_param import AIDefenseRailConfigParam
from .auto_align_rail_config_param import AutoAlignRailConfigParam
from .trend_micro_rail_config_param import TrendMicroRailConfigParam
from .sensitive_data_detection_param import SensitiveDataDetectionParam
from .fact_checking_rail_config_param import FactCheckingRailConfigParam
from .guardrails_ai_rail_config_param import GuardrailsAIRailConfigParam
from .jailbreak_detection_config_param import JailbreakDetectionConfigParam
from .local_hf_classifier_config_param import LocalHfClassifierConfigParam
from .remote_hf_classifier_config_param import RemoteHfClassifierConfigParam
from .crowd_strike_aidr_rail_config_param import CrowdStrikeAidrRailConfigParam
from .context_bloat_detection_config_param import ContextBloatDetectionConfigParam

__all__ = ["RailsConfigDataParam", "HfClassifier"]

HfClassifier: TypeAlias = Union[LocalHfClassifierConfigParam, RemoteHfClassifierConfigParam]


class RailsConfigDataParam(TypedDict, total=False):
    """Configuration data for specific rails that are supported out-of-the-box."""

    ai_defense: AIDefenseRailConfigParam
    """Configuration data for the Cisco AI Defense API"""

    autoalign: AutoAlignRailConfigParam
    """Configuration data for the AutoAlign API"""

    clavata: ClavataRailConfigParam
    """Configuration data for the Clavata API"""

    content_safety: ContentSafetyConfigParam
    """Configuration data for content safety rails."""

    context_bloat_detection: ContextBloatDetectionConfigParam
    """Configuration for context bloat / context manipulation detection."""

    crowdstrike_aidr: CrowdStrikeAidrRailConfigParam
    """Configuration data for the CrowdStrike AIDR API"""

    fact_checking: FactCheckingRailConfigParam
    """Configuration data for the fact-checking rail."""

    fiddler: FiddlerGuardrailsParam
    """Configuration for Fiddler Guardrails."""

    gliner: GLiNERDetectionParam
    """Configuration for GLiNER PII detection."""

    guardrails_ai: GuardrailsAIRailConfigParam
    """Configuration data for Guardrails AI integration."""

    hf_classifier: Dict[str, HfClassifier]
    """Named HF classifier configurations.

    Keys are classifier names referenced by flows.
    """

    injection_detection: InjectionDetectionParam
    """Configuration for injection detection."""

    jailbreak_detection: JailbreakDetectionConfigParam
    """Configuration data for jailbreak detection."""

    pangea: PangeaRailConfigParam
    """Configuration data for the Pangea AI Guard API"""

    patronus: PatronusRailConfigParam
    """Configuration data for the Patronus Evaluate API"""

    polygraf: PolygrafDetectionParam
    """Configuration for Polygraf PII detection."""

    privateai: PrivateAIDetectionParam
    """Configuration for Private AI."""

    regex_detection: RegexDetectionParam
    """Configuration for regex pattern detection."""

    sensitive_data_detection: SensitiveDataDetectionParam
    """Configuration of what sensitive data should be detected."""

    trend_micro: TrendMicroRailConfigParam
    """Configuration data for the Trend Micro AI Guard API"""
