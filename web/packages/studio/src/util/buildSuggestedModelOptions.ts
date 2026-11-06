// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SelectItemOption } from '@nemo/common/src/components/form/ControlledSearchableSelect';

const EXCLUDED_TERMS = ['llama', 'safety', 'safeguard', 'embed', 'vl', 'reward', 'parse'];

const isSuggested = (name: string): boolean => {
  const lower = name.toLowerCase();
  if (EXCLUDED_TERMS.some((term) => lower.includes(term))) return false;
  return lower.includes('nvidia') && lower.includes('nemotron');
};

// Not chat LLMs — never valid as an agent's LLM, so excluded from the picker entirely.
const NON_LLM_TERMS = ['embed', 'rerank', 'reward', 'safeguard', 'safety', 'parse', 'vl', 'guard'];

const isLlmCandidate = (name: string): boolean => {
  const lower = name.toLowerCase();
  return !NON_LLM_TERMS.some((term) => lower.includes(term));
};

export interface ModelListEntry {
  name: string;
}

export const SUGGESTED_MODEL_GROUP_LABELS = {
  suggested: 'Suggested',
  all: 'All Models',
} as const;

/**
 * Build the option set for the Studio model picker with a Suggested / All split.
 * Only chat-LLM candidates are listed (embedding/rerank/safety/vl/... are excluded).
 * Suggested = NVIDIA Nemotron models. Each model appears once (suggested models are
 * not repeated under "All") so option values stay unique.
 */
export const buildSuggestedModelOptions = (models: ModelListEntry[]): SelectItemOption[] => {
  const seenValues = new Set<string>();
  const base = models
    .filter((m) => isLlmCandidate(m.name))
    .map((m) => ({ value: m.name, label: m.name }))
    .filter((o) => {
      if (seenValues.has(o.value)) return false;
      seenValues.add(o.value);
      return true;
    });
  const suggested = base
    .filter((o) => isSuggested(o.value))
    .map((o) => ({ ...o, group: 'suggested' as const }));
  const suggestedValues = new Set(suggested.map((o) => o.value));
  const all = base
    .filter((o) => !suggestedValues.has(o.value))
    .map((o) => ({ ...o, group: 'all' as const }));
  return [...suggested, ...all];
};

export const pickDefaultModelName = (models: ModelListEntry[]): string | undefined => {
  const names = models.map((m) => m.name);
  return names.find(isSuggested) ?? names.find(isLlmCandidate);
};
