// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VariableButton } from '@nemo/common/src/components/buttons/VariableButton';
import {
  type VariableDef,
  VariableTextArea,
  type VariableTextAreaHandle,
} from '@nemo/common/src/components/form/VariableTextArea';
import { Banner, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import {
  itemRef,
  templateFields,
} from '@studio/components/evaluation/AgentEvaluationWizard/agentEvalConfig';
import { hasMalformedBraces } from '@studio/components/evaluation/EvalWizard/judgePromptVariables';
import { ComponentProps, FC, useMemo, useRef } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

interface PromptTemplateFormValues {
  promptTemplate: string | null;
  body: { judgePrompt: string | null };
}

interface PromptTemplateSectionProps {
  name: 'promptTemplate' | 'body.judgePrompt';
  title: string;
  titleKind: ComponentProps<typeof Text>['kind'];
  description: string;
  /** The prompt while the user has not edited it. */
  generated: string;
  columns: string[];
  /** Fit-check findings for the prompt. */
  issues: string[];
  /** Offer the agent's response, which only the judge can read. */
  withResponse?: boolean;
}

/**
 * A prompt rendered against each dataset row.
 *
 * The field holds ``null`` while the prompt is still ``generated``, so picking another dataset
 * regenerates it. Once edited it is left alone.
 */
export const PromptTemplateSection: FC<PromptTemplateSectionProps> = ({
  name,
  title,
  titleKind,
  description,
  generated,
  columns,
  issues,
  withResponse = false,
}) => {
  const { control, setValue, getFieldState, formState } =
    useFormContext<PromptTemplateFormValues>();
  const edited = useWatch({ control, name });
  const editorRef = useRef<VariableTextAreaHandle>(null);

  const value = edited ?? generated;
  const malformed = hasMalformedBraces(value);
  const error = getFieldState(name, formState).error?.message;

  const variables = useMemo<VariableDef[]>(
    () => [
      ...columns.map((column) => ({ name: itemRef(column), description: 'Dataset field' })),
      ...(withResponse
        ? [{ name: 'sample.output_text', description: "The agent's response" }]
        : []),
    ],
    [columns, withResponse]
  );

  const isKnown = (token: string) =>
    templateFields(`{{${token}}}`).every((field) => columns.includes(field));

  return (
    <Stack gap="density-xs">
      <Flex gap="density-sm" align="center" justify="between">
        <Text kind={titleKind} className="shrink-0">
          {title}
        </Text>
        <VariableButton
          variables={variables}
          onSelect={(variable) => editorRef.current?.insertVariable(variable.name)}
        />
      </Flex>

      <Text kind="body/regular/sm" className="text-secondary">
        {description}
      </Text>

      {malformed ? (
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

      {error && !malformed ? (
        <Banner kind="inline" status="error">
          {error}
        </Banner>
      ) : null}

      <VariableTextArea
        ref={editorRef}
        attributes={{ TextAreaElement: { 'aria-label': title } }}
        className="font-mono text-xs border border-base rounded-lg p-4 bg-surface-raised"
        variables={variables}
        isKnown={isKnown}
        value={value}
        onChange={(next) => {
          setValue(name, next === generated ? null : next, { shouldValidate: true });
        }}
      />
    </Stack>
  );
};
