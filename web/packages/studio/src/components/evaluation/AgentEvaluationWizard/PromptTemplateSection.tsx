// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VariableButton } from '@nemo/common/src/components/buttons/VariableButton';
import {
  type VariableDef,
  VariableTextArea,
  type VariableTextAreaHandle,
} from '@nemo/common/src/components/form/VariableTextArea';
import { Pencil } from '@nemo/icons/icons';
import {
  Banner,
  Button,
  Checkbox,
  Flex,
  FormField,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  AGENT_PROMPT_NAMES,
  type FitIssues,
  itemRef,
  templateFields,
} from '@studio/components/evaluation/AgentEvaluationWizard/agentEvalConfig';
import { hasMalformedBraces } from '@studio/components/evaluation/EvalWizard/judgePromptVariables';
import { type ComponentProps, type FC, useMemo, useRef, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

interface PromptTemplateFormValues {
  promptTemplate: string | null;
  promptColumns: string[];
  judgeColumns: string[];
  body: { judgePrompt: string | null };
}

interface PromptTemplateSectionProps {
  name: 'promptTemplate' | 'body.judgePrompt';
  /** The columns the generated prompt sends, in the order they were checked. */
  columnsName: 'promptColumns' | 'judgeColumns';
  title: string;
  titleKind: ComponentProps<typeof Text>['kind'];
  description: string;
  /** The prompt generated from the checked columns, used until the user edits it. */
  generated: string;
  columns: string[];
  /** Fit-check findings for the prompt. */
  issues: FitIssues;
  /** Offer the agent's response, which only the judge can read. */
  withResponse?: boolean;
}

/**
 * A prompt rendered against each dataset row, shown as a checkbox per column.
 *
 * Checking a column only records it in ``columnsName``; the prompt generated from those columns
 * is shown in the editor behind the pencil. The prompt field holds ``null`` while it is still
 * generated. Once edited it is left alone, and the checkboxes only show which columns it reads
 * until it is reset.
 */
export const PromptTemplateSection: FC<PromptTemplateSectionProps> = ({
  name,
  columnsName,
  title,
  titleKind,
  description,
  generated,
  columns,
  issues,
  withResponse = false,
}) => {
  const { control, setValue, getFieldState, formState, trigger } =
    useFormContext<PromptTemplateFormValues>();
  const edited = useWatch({ control, name });
  const selected = useWatch({ control, name: columnsName });
  const [open, setOpen] = useState(false);
  const editorRef = useRef<VariableTextAreaHandle>(null);

  const value = edited ?? generated;
  const malformed = hasMalformedBraces(value);
  const error = getFieldState(name, formState).error?.message;
  const fieldError = error && !malformed ? error : undefined;
  const available = withResponse ? undefined : AGENT_PROMPT_NAMES;
  const isCustom = edited !== null;
  const checked = isCustom ? templateFields(edited, available) : selected;

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
    templateFields(`{{${token}}}`, available).every((field) => columns.includes(field));

  const toggle = (column: string, isChecked: boolean) => {
    setValue(
      columnsName,
      isChecked ? [...selected, column] : selected.filter((item) => item !== column)
    );
    if (error) {
      void trigger(name);
    }
  };

  return (
    <Stack gap="density-xs">
      <Flex gap="density-sm" align="center" justify="between">
        <Text kind={titleKind} className="shrink-0">
          {title}
        </Text>
        <Flex gap="density-xs" align="center" className="shrink-0">
          {open ? (
            <VariableButton
              variables={variables}
              onSelect={(variable) => editorRef.current?.insertVariable(variable.name)}
            />
          ) : null}
          <Button
            type="button"
            kind="tertiary"
            size="small"
            aria-label={open ? `Hide ${title}` : `Edit ${title}`}
            aria-expanded={open}
            onClick={() => setOpen(!open)}
          >
            <Pencil className="size-3.5" aria-hidden />
          </Button>
        </Flex>
      </Flex>

      <Text kind="body/regular/md" className="text-secondary">
        {description}
      </Text>

      {malformed ? (
        <Banner kind="inline" status="error">
          The prompt has nested or unclosed <code>{'{{ }}'}</code> braces, so the evaluation would
          fail to parse it.
        </Banner>
      ) : null}

      {issues.errors.map((issue) => (
        <Banner key={issue} kind="inline" status="error">
          {issue}
        </Banner>
      ))}
      {issues.warnings.map((issue) => (
        <Banner key={issue} kind="inline" status="warning">
          {issue}
        </Banner>
      ))}

      {open ? (
        <Stack gap="density-xs">
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
          {fieldError ? <FormField status="error" slotError={fieldError} /> : null}
        </Stack>
      ) : (
        <Stack gap="density-sm">
          <Stack gap="density-xs">
            <Stack
              role="group"
              aria-label={`${title} fields`}
              gap="density-xs"
              className="max-h-32 overflow-y-auto border border-base rounded-lg p-density-md"
            >
              {withResponse ? (
                <Checkbox slotLabel="Agent response (sample.output_text)" checked disabled />
              ) : null}
              {columns.map((column) => (
                <Checkbox
                  key={column}
                  slotLabel={column}
                  checked={checked.includes(column)}
                  disabled={isCustom}
                  onCheckedChange={(isChecked) => toggle(column, isChecked === true)}
                />
              ))}
            </Stack>
            {fieldError ? <FormField status="error" slotError={fieldError} /> : null}
          </Stack>
          {isCustom ? (
            <Flex gap="density-sm" align="center" justify="between">
              <Text kind="body/regular/md" className="text-secondary">
                Edited prompt. Change it in the editor, or reset it to pick fields here.
              </Text>
              <Button
                type="button"
                kind="tertiary"
                size="small"
                onClick={() => setValue(name, null, { shouldValidate: true })}
              >
                Reset
              </Button>
            </Flex>
          ) : null}
        </Stack>
      )}
    </Stack>
  );
};
