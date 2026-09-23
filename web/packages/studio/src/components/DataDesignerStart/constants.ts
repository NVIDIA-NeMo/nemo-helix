// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOption, StartOptionTag } from '@studio/components/DataDesignerStart/types';
import type { PromptSuggestion } from '@studio/components/PromptSuggestionTags/types';
import { Plus, Sparkles } from 'lucide-react';

/**
 * Example prompts offered as pills inside an empty prompt field. Each is a complete,
 * generation-ready description — the pill label is only the shorthand for it.
 */
export const PROMPT_SUGGESTIONS: PromptSuggestion[] = [
  {
    label: 'Phishing email triage',
    prompt:
      '200 customer support emails for training a phishing triage agent, each labelled as phishing or legitimate, with a short reason for the label and the sender domain. Sampled across categories (billing, returns, tech support) with subcategories per category (billing: overcharge, failed payment; returns: damaged item, wrong size)',
  },
  {
    label: 'Support ticket routing',
    prompt:
      '100 inbound customer support tickets for training a triage agent. Each row has the raw ticket text as the customer wrote it, the queue it should route to (billing, shipping, technical, account cancellation), an urgency level (P1 to P4) and a one-line summary. Include ambiguous tickets that plausibly span two queues, and a few where the customer threatens to churn',
  },
  {
    label: 'Refund policy Q&A',
    prompt:
      '50 evaluation examples for a customer-facing refund policy assistant. Each row has a passage from a returns and refunds policy, a question a real customer would ask, the answer grounded in that passage, and whether the policy actually covers the situation. Include questions the policy does not answer, marked as out of scope',
  },
];

/** Difficulty levels, kept together because they only mean anything relative to each other. */
const BEGINNER: StartOptionTag = { label: 'Beginner', color: 'gray', kind: 'solid' };
const ADVANCED: StartOptionTag = { label: 'Advanced', color: 'gray', kind: 'solid' };
export const TEMPLATES_TAG: StartOptionTag = {
  label: 'Intermediate',
  color: 'gray',
  kind: 'solid',
};

/** The non-template ways in; templates are picked directly, below the divider. */
export const START_OPTIONS: StartOption[] = [
  {
    id: 'ai',
    title: 'Describe with AI',
    description:
      'Tell us what you need in plain language. AI drafts the columns and prompts — then you refine everything visually.',
    icon: Sparkles,
    tag: BEGINNER,
    enabled: true,
  },
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Open an empty canvas and add columns block by block, your way.',
    icon: Plus,
    tag: ADVANCED,
    enabled: true,
  },
];

/** Section order. Tags outside this list fall into "Other". */
export const TEMPLATE_SECTIONS = ['Evaluation', 'Fine-tuning'] as const;
export const OTHER_SECTION = 'Other';

/** One accent per section. Tokens rather than the design's literals, for the light theme. */
export const SECTION_ACCENTS: Record<string, string> = {
  Evaluation: 'var(--text-color-accent-purple)',
  'Fine-tuning': 'var(--text-color-accent-yellow)',
  [OTHER_SECTION]: 'var(--text-color-accent-teal)',
};
