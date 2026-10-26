// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VariableButton } from '@nemo/common/src/components/buttons/VariableButton';
import {
  type VariableDef,
  VariableTextArea,
  type VariableTextAreaHandle,
} from '@nemo/common/src/components/form/VariableTextArea';
import { Pencil } from '@nemo/icons/icons';
import { Banner, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { templateFields } from '@studio/components/evaluation/AgentEvaluationWizard/agentEvalConfig';
import { hasMalformedBraces } from '@studio/components/evaluation/ModelEvaluationWizard/judgePromptVariables';
import {
  CANONICAL_FIELD_LABELS,
  type EvaluationFormValues,
  toFieldMapping,
} from '@studio/components/evaluation/ModelEvaluationWizard/types';
import { toDatasetBindings } from '@studio/components/evaluation/ModelEvaluationWizard/useDatasetBindings';
import { FC, useMemo, useRef, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

interface PromptTemplateFormValues {
  promptTemplate: string | null;
  fieldMapping: EvaluationFormValues['fieldMapping'];
}

interface PromptTemplateSectionProps {
  /** The dataset's columns, offered as ``item.<column>``. */
  columns: string[];
  /** Fit-check findings for the edited prompt. */
  issues: string[];
}

/**
 * The user message sent to the agent for every row.
 *
 * Follows the Input mapping until the user edits it: ``promptTemplate`` holds ``null`` while it
 * still does, the same convention as the judge prompt. Editing is how a prompt combines several
 * fields, which a single Input mapping cannot.
 */
export const PromptTemplateSection: FC<PromptTemplateSectionProps> = ({ columns, issues }) => {
  const { control, setValue } = useFormContext<PromptTemplateFormValues>();
  const [edited, fieldMapping] = useWatch({ control, name: ['promptTemplate', 'fieldMapping'] });
  const [open, setOpen] = useState(false);
  const editorRef = useRef<VariableTextAreaHandle>(null);

  const bindings = toDatasetBindings(fieldMapping);
  const generated = bindings.input;
  const mapping = useMemo(() => toFieldMapping(fieldMapping) ?? {}, [fieldMapping]);

  const available = useMemo<VariableDef[]>(
    () => [
      ...(['input', 'reference', 'context'] as const)
        .filter((field) => mapping[field])
        .map((field) => ({
          name: field,
          description: `The ${CANONICAL_FIELD_LABELS[field]} field (mapped to \`${mapping[field]}\`)`,
        })),
      ...columns.map((column) => ({ name: `item.${column}`, description: 'Dataset field' })),
    ],
    [columns, mapping]
  );

  const isKnown = (token: string) =>
    templateFields(`{{${token}}}`).every((name) => columns.includes(name) || name in mapping);

  return (
    <Stack gap="density-xs" className="border border-base rounded-lg p-4 bg-surface-raised">
      <Flex gap="density-sm" align="center" justify="between">
        <Text kind="body/bold/md" className="shrink-0">
          Agent Prompt
        </Text>
        <Flex gap="density-xs" align="center" className="shrink-0">
          {open ? (
            <VariableButton
              variables={available}
              onSelect={(variable) => editorRef.current?.insertVariable(variable.name)}
            />
          ) : null}
          <Button
            type="button"
            kind="tertiary"
            size="small"
            aria-label={open ? 'Hide agent prompt' : 'Edit agent prompt'}
            aria-expanded={open}
            onClick={() => setOpen(!open)}
          >
            <Pencil className="size-3.5" aria-hidden />
          </Button>
        </Flex>
      </Flex>

      <Text kind="body/regular/sm" className="text-secondary">
        Sent to the agent as the user message for every row. It is the Input field unless you edit
        it, for example to combine several fields.
      </Text>

      {edited && hasMalformedBraces(edited) ? (
        <Banner kind="inline" status="error">
          The prompt has nested or unclosed <code>{'{{ }}'}</code> braces, so the evaluation would
          fail to parse it.
        </Banner>
      ) : null}

      {issues.map((issue) => (
        <Banner key={issue} kind="inline" status="error">
          {issue}
        </Banner>
      ))}

      {open ? (
        <VariableTextArea
          ref={editorRef}
          attributes={{ TextAreaElement: { 'aria-label': 'Agent prompt' } }}
          className="font-mono text-xs"
          variables={available}
          isKnown={isKnown}
          value={edited ?? generated}
          onChange={(next) => {
            setValue('promptTemplate', next === generated ? null : next, { shouldValidate: true });
          }}
        />
      ) : null}
    </Stack>
  );
};
