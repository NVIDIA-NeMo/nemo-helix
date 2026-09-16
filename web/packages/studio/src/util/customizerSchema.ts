// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Strict, customizer-aligned schema detection for fine-tuning datasets.
 *
 * Mirrors services/customizer/src/nhx/customizer/tasks/training/datasets/schemas.py
 * (get_sft_dataset_discriminator and get_preference_dataset_discriminator).
 *
 * Common's detectFileStructure is too permissive for this purpose — it accepts
 * synonyms like "question"/"answer" and recurses to find a "messages"-like array.
 * Customizer requires literal top-level keys, so studio mirrors that exactly.
 */
/**
 * Training objective used to pick the expected dataset schema. Mirrors the
 * customizer backend's supported objectives (SFT, DPO, distillation). Owned
 * here so schema detection has no dependency on form-layer constants.
 */
export type TrainingType = 'sft' | 'dpo' | 'distillation' | 'grpo';

export type CustomizerSchemaVariant =
  /** SFT chat format: {"messages": [{"role": "...", ...}]} */
  | 'sft-chat'
  /** SFT prompt-completion: {"prompt": "...", "completion": "..."} */
  | 'sft-prompt-completion'
  /** DPO native: {"context": [...], "completions": [...]} */
  | 'dpo-preference'
  /** DPO HelpSteer3: {"overall_preference": ..., ...} */
  | 'dpo-helpsteer3'
  /** DPO Tulu3: {"chosen": [msgs], "rejected": [msgs]} */
  | 'dpo-tulu3'
  /** DPO BinaryPreference: {"prompt": "...", "chosen": "...", "rejected": "..."} */
  | 'dpo-binary-preference'
  /** GRPO NeMo Gym verifiers format (pi-to-gym-conversion output): {"task_idx": ..., "vf_env_id": "...", "responses_create_params": {...}, "agent_ref": {...}} */
  | 'grpo-gym';

/**
 * Single source of truth for the user-facing label of each detected schema.
 * Tests reference this map directly so a label change can never drift between
 * source and assertions, and the panel renders `CUSTOMIZER_SCHEMA_LABELS[variant]`
 * rather than carrying the string on every detection result.
 */
export const CUSTOMIZER_SCHEMA_LABELS: Record<CustomizerSchemaVariant, string> = {
  'sft-chat': 'Chat Completion',
  'sft-prompt-completion': 'Completion',
  'dpo-preference': 'Preference',
  'dpo-helpsteer3': 'HelpSteer3',
  'dpo-tulu3': 'Tulu3',
  'dpo-binary-preference': 'Binary Preference',
  'grpo-gym': 'GRPO Gym (responses_create_params + agent_ref)',
};

export interface CustomizerSchemaDetection {
  variant: CustomizerSchemaVariant;
  /** Friendly label for the panel checklist row, sourced from CUSTOMIZER_SCHEMA_LABELS. */
  label: string;
}

const detection = (variant: CustomizerSchemaVariant): CustomizerSchemaDetection => ({
  variant,
  label: CUSTOMIZER_SCHEMA_LABELS[variant],
});

const isMessageList = (value: unknown): boolean => {
  if (!Array.isArray(value) || value.length === 0) return false;
  const first = value[0];
  return typeof first === 'object' && first !== null && 'role' in (first as object);
};

const detectSft = (row: Record<string, unknown>): CustomizerSchemaDetection | null => {
  // Chat: literal top-level "messages" with role on first item.
  if (isMessageList(row.messages)) return detection('sft-chat');
  // Prompt-completion: literal "prompt" + "completion".
  if ('prompt' in row && 'completion' in row) return detection('sft-prompt-completion');
  return null;
};

const detectGrpo = (row: Record<string, unknown>): CustomizerSchemaDetection | null => {
  // NeMo Gym verifiers format — produced by pi-to-gym-conversion; validated by GRPO_SCHEMA backend
  // Agent-agnostic: NeMo-RL routes on agent_ref.name and applies sampling from
  // responses_create_params, and reads nothing else. Requiring vf_env_id here would
  // reject every dataset targeting a Gym agent other than verifiers_agent (see
  // GRPO_SCHEMA in services/rl/.../datasets/validation.py).
  if ('responses_create_params' in row && 'agent_ref' in row) return detection('grpo-gym');
  return null;
};

const detectDpo = (row: Record<string, unknown>): CustomizerSchemaDetection | null => {
  // Native PreferenceDataset: context + completions.
  if ('context' in row && 'completions' in row) return detection('dpo-preference');
  // HelpSteer3: overall_preference score.
  if ('overall_preference' in row) return detection('dpo-helpsteer3');
  // Tulu3: chosen/rejected as message lists. Must be checked before binary
  // (binary also has chosen/rejected, but as strings).
  if ('chosen' in row && 'rejected' in row && isMessageList(row.chosen)) {
    return detection('dpo-tulu3');
  }
  // BinaryPreference: prompt + chosen + rejected (typically as strings).
  if ('prompt' in row && 'chosen' in row && 'rejected' in row) {
    return detection('dpo-binary-preference');
  }
  return null;
};

/**
 * Strictly detect which customizer-recognized schema a sample row matches,
 * scoped to the currently selected training type. Returns null when none of
 * the type-specific shapes match — caller should render an error.
 *
 * Distillation isn't exposed in the new-model UI; if it ever is, it shares
 * SFT's schema rules in customizer.
 */
export const detectCustomizerSchema = (
  row: Record<string, unknown> | null,
  trainingType: TrainingType
): CustomizerSchemaDetection | null => {
  if (!row) return null;
  if (trainingType === 'dpo') return detectDpo(row);
  if (trainingType === 'grpo') return detectGrpo(row);
  return detectSft(row);
};

/**
 * Human-readable description of the keys customizer expects for a given
 * training type, used for the "Schema does not match" copy.
 */
export const expectedSchemaCopy = (trainingType: TrainingType): string => {
  if (trainingType === 'dpo') {
    return 'Must contain chosen and rejected columns (or context + completions, or overall_preference).';
  }
  if (trainingType === 'grpo') {
    return 'Must contain responses_create_params and agent_ref (NeMo Gym format). Agent-specific fields such as vf_env_id are optional. Use pi-to-gym-conversion to prepare a verifiers dataset.';
  }
  return 'Must contain messages (chat) or prompt and completion.';
};

// Encoding validation lives in the customizer validation hook, not here — it
// uses TextDecoder('utf-8', { fatal: true }) on raw bytes and runs as its own
// TanStack query (see useCustomizationDatasetValidation/encodingQuery.ts).

const SCHEMA_PREVIEW_INDENT = '  ';
/** Cap recursion to keep degenerate inputs (deep/cyclic JSON) from looping. */
const SCHEMA_PREVIEW_MAX_DEPTH = 8;

/**
 * Render a JSON value as a TypeScript-shaped string, recursing into nested
 * objects and arrays so the schema preview shows the full structure rather
 * than collapsing everything to "object".
 *
 * Heterogeneous arrays use the FIRST element's shape as representative —
 * matches how customizer's discriminator validates per-row.
 */
const renderInferredType = (value: unknown, depth: number): string => {
  if (depth > SCHEMA_PREVIEW_MAX_DEPTH) return 'unknown';
  if (value === null) return 'null';
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    const inner = renderInferredType(value[0], depth + 1);
    if (inner.includes('\n')) {
      const open = SCHEMA_PREVIEW_INDENT.repeat(depth + 1);
      const close = SCHEMA_PREVIEW_INDENT.repeat(depth);
      return `[\n${open}${inner},\n${close}]`;
    }
    return `[${inner}]`;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return '{}';
    const open = SCHEMA_PREVIEW_INDENT.repeat(depth + 1);
    const close = SCHEMA_PREVIEW_INDENT.repeat(depth);
    const lines = entries
      .map(([key, v]) => `${open}${key}: ${renderInferredType(v, depth + 1)},`)
      .join('\n');
    return `{\n${lines}\n${close}}`;
  }
  return typeof value;
};

/**
 * Build a TypeScript-like schema preview string from a parsed JSON row.
 * Returns an empty string when the row is missing — callers that render the
 * preview should treat empty as "nothing to show" and skip the block.
 */
export const inferRowSchema = (row: Record<string, unknown> | null): string => {
  if (!row) return '';
  return renderInferredType(row, 0);
};

const isNonEmptyString = (v: unknown): v is string => typeof v === 'string' && v.length > 0;

const isNonEmptyArray = (v: unknown): v is unknown[] => Array.isArray(v) && v.length > 0;

const validateChatMessages = (messages: unknown): string | null => {
  if (!isNonEmptyArray(messages)) return 'messages must be a non-empty array';
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (typeof msg !== 'object' || msg === null) {
      return `messages[${i}] is not an object`;
    }
    const m = msg as Record<string, unknown>;
    if (!isNonEmptyString(m.role)) {
      return `messages[${i}].role is missing or empty`;
    }
    // Backend SFTChatMessage validator (services/customizer/.../schemas.py:292)
    // rejects messages with BOTH content and thinking — they're mutually
    // exclusive. Use key presence (not just non-empty) to match the backend's
    // `is not None` semantics.
    const contentPresent = m.content !== undefined && m.content !== null;
    const thinkingPresent = m.thinking !== undefined && m.thinking !== null;
    if (contentPresent && thinkingPresent) {
      return `messages[${i}] cannot have both content and thinking (mutually exclusive)`;
    }
    const hasContent = isNonEmptyString(m.content);
    const hasThinking = isNonEmptyString(m.thinking);
    const hasToolCalls = isNonEmptyArray(m.tool_calls);
    if (!hasContent && !hasThinking && !hasToolCalls) {
      return `messages[${i}] must have non-empty content, thinking, or tool_calls`;
    }
  }
  return null;
};

/**
 * Per-variant required-field check on a single parsed row. Returns null when
 * the row is complete enough for customizer to use, or a short human-readable
 * message naming the first offending field. Mirrors the spirit of the Pydantic
 * validators in services/customizer/.../schemas.py without re-implementing
 * every edge case (we cover the common "missing key" / "empty value" cases the
 * pre-flight panel needs to flag).
 */
export const validateRowCompleteness = (
  row: Record<string, unknown>,
  variant: CustomizerSchemaVariant
): string | null => {
  switch (variant) {
    case 'sft-chat':
      return validateChatMessages(row.messages);

    case 'sft-prompt-completion':
      if (!isNonEmptyString(row.prompt)) return 'prompt is missing or empty';
      if (!isNonEmptyString(row.completion)) return 'completion is missing or empty';
      return null;

    case 'dpo-binary-preference': {
      // prompt can be a string OR a list of chat messages.
      if (typeof row.prompt === 'string') {
        if (!isNonEmptyString(row.prompt)) return 'prompt is empty';
      } else if (Array.isArray(row.prompt)) {
        const msgError = validateChatMessages(row.prompt);
        if (msgError) return `prompt: ${msgError}`;
      } else {
        return 'prompt must be a non-empty string or list of messages';
      }
      if (!isNonEmptyString(row.chosen)) return 'chosen is missing or empty';
      if (!isNonEmptyString(row.rejected)) return 'rejected is missing or empty';
      return null;
    }

    case 'dpo-tulu3': {
      const chosenError = validateChatMessages(row.chosen);
      if (chosenError) return `chosen: ${chosenError}`;
      const rejectedError = validateChatMessages(row.rejected);
      if (rejectedError) return `rejected: ${rejectedError}`;
      return null;
    }

    case 'dpo-preference': {
      const contextError = validateChatMessages(row.context);
      if (contextError) return `context: ${contextError}`;
      if (!isNonEmptyArray(row.completions)) return 'completions must be a non-empty array';
      return null;
    }

    case 'grpo-gym': {
      // Only the two fields NeMo-RL reads unconditionally are required. Agent-specific
      // keys (vf_env_id for verifiers_agent, expected_answer for the math resources
      // servers) pass through untouched — see GymDatasetRow on the backend.
      if (
        !row.responses_create_params ||
        typeof row.responses_create_params !== 'object' ||
        Array.isArray(row.responses_create_params)
      ) {
        return 'responses_create_params must be an object';
      }
      if (!row.agent_ref || typeof row.agent_ref !== 'object' || Array.isArray(row.agent_ref)) {
        return 'agent_ref must be an object';
      }
      const agentRef = row.agent_ref as Record<string, unknown>;
      if (!isNonEmptyString(agentRef.name)) return 'agent_ref.name is missing or empty';
      // Optional, but wrong-typed values still fail at rollout, so check when present.
      if (row.task_idx !== undefined && typeof row.task_idx !== 'number') {
        return 'task_idx must be a number when present';
      }
      if (row.vf_env_id !== undefined && !isNonEmptyString(row.vf_env_id)) {
        return 'vf_env_id must be a non-empty string when present';
      }
      return null;
    }

    case 'dpo-helpsteer3': {
      if (typeof row.context === 'string') {
        if (!isNonEmptyString(row.context)) return 'context is empty';
      } else if (Array.isArray(row.context)) {
        const msgError = validateChatMessages(row.context);
        if (msgError) return `context: ${msgError}`;
      } else {
        return 'context must be a non-empty string or list of messages';
      }
      if (!isNonEmptyString(row.response1)) return 'response1 is missing or empty';
      if (!isNonEmptyString(row.response2)) return 'response2 is missing or empty';
      if (typeof row.overall_preference !== 'number') {
        return 'overall_preference must be a number';
      }
      return null;
    }
  }
};
