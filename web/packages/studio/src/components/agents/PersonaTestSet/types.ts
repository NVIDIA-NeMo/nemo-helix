// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NEMOTRON_PERSONAS_LOCALES } from '@studio/components/agents/PersonaTestSet/nemotronPersonas';

export type NemotronPersonasLocale = (typeof NEMOTRON_PERSONAS_LOCALES)[number];

/**
 * How each row's expected answer is written. `rubric` has the LLM describe what a correct reply
 * contains, for agents with open-ended output. `arithmetic` samples the operands and computes the
 * exact result, so ground truth for calculator-style agents never depends on an LLM.
 */
export type PersonaAnswerStyle = 'rubric' | 'arithmetic';

/** What the test set is generated for, read from the agent the user starts from. */
export interface PersonaTestSetAgent {
  name: string;
  description?: string;
  /** The agent's system instructions, so personas ask things the agent is meant to handle. */
  instructions?: string;
}

export interface PersonaTestSetOptions {
  agent: PersonaTestSetAgent;
  locale: NemotronPersonasLocale;
  rows: number;
  answerStyle: PersonaAnswerStyle;
}
