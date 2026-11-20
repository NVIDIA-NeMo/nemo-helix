# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The probes a simulation can run, described for people choosing them.

The CLI, its tab completion and Studio's New simulation dialog all read this catalog.
"""

from pydantic import BaseModel


class Probe(BaseModel):
    name: str
    label: str
    tests: str
    suits: str


PROBES = (
    Probe(
        name="general_open_ended",
        label="General questions",
        tests="Unhelpful hedging, mismatched tone, answers that get worse for some personalities",
        suits="Any agent that answers questions",
    ),
    Probe(
        name="general_educational",
        label="Teaching and explaining",
        tests="Explaining at the wrong level, condescension, no step-by-step help",
        suits="Tutors and explainers",
    ),
    Probe(
        name="financial_services",
        label="Banking and finance",
        tests="Wrong or unauthorized tool use, poor document search, made-up or wrong numbers",
        suits="Banking, payments and financial advice",
    ),
    Probe(
        name="safety_agentic",
        label="Tool use boundaries",
        tests="Taking actions the user didn't ask for, following instructions injected into tool results, changing things without confirming the consequences first",
        suits="Agents that change accounts, records or settings through tools",
    ),
    Probe(
        name="safety_chat_pressure",
        label="Pressure to break rules",
        tests="Giving in after several turns of reframing",
        suits="Agents that must refuse some requests",
    ),
    Probe(
        name="health_triage_disclosure",
        label="Health triage",
        tests="Missing urgent symptoms the user reveals only gradually",
        suits="Care navigation and triage",
    ),
    Probe(
        name="health_general_disclosure",
        label="General health",
        tests="Downplaying red-flag symptoms and missed medication",
        suits="General health assistants",
    ),
    Probe(
        name="health_therapy_disclosure",
        label="Mental health support",
        tests="Mishandling crisis content",
        suits="Mental health support",
    ),
    Probe(
        name="health_decision_support_disclosure",
        label="Clinician decision support",
        tests="Missing danger signs in a case a clinician presents",
        suits="Clinical decision support",
    ),
    Probe(
        name="sov_ai_facts",
        label="Local facts",
        tests="Wrong local facts and made-up institutions",
        suits="Agents answering country-specific questions",
    ),
    Probe(
        name="sov_ai_dynamic",
        label="Local topics without ground truth",
        tests="Made-up or self-contradicting answers on local topics",
        suits="Agents answering country-specific questions",
    ),
    Probe(
        name="sov_ai_multilingual_parity",
        label="Parity across locales",
        tests="Treating the same question differently across languages",
        suits="Multilingual agents, run across several locales",
    ),
    Probe(
        name="identity_disclosure",
        label="Model identity",
        tests="The AI naming the wrong company or model as its maker when asked who built it (about the AI itself, not the user's identity)",
        suits="Branded assistants that must say who made them",
    ),
)
