// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VariableButton } from '@nemo/common/src/components/buttons/VariableButton';
import {
  VariableTextArea,
  type VariableTextAreaHandle,
} from '@nemo/common/src/components/form/VariableTextArea';
import { Banner, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import {
  composeJudgeUserPrompt,
  type EvaluationFormValues,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import { useJudgePromptVariables } from '@studio/routes/evaluation/EvaluationNewRoute/useJudgePromptVariables';
import { Pencil } from 'lucide-react';
import { FC, useRef, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

/**
 * The prompt sent to the judge for every row.
 *
 * Derived from the dataset bindings until the user edits it: `body.judgePrompt`
 * holds `null` while it still tracks them, so changing the dataset or a mapping
 * regenerates it with no work. Once edited it is left alone, stale bindings and
 * all -- the edit is the user's to maintain.
 */
export const JudgePromptSection: FC = () => {
  const { control, setValue } = useFormContext<EvaluationFormValues>();
  const edited = useWatch({ control, name: 'body.judgePrompt' });
  const bindings = useDatasetBindings();
  const { available, isKnown, invalid, nonPortable, malformed } = useJudgePromptVariables();
  const [open, setOpen] = useState(false);
  const editorRef = useRef<VariableTextAreaHandle>(null);

  const generated = composeJudgeUserPrompt(bindings);

  return (
    <Stack gap="density-xs" className="border border-base rounded-lg p-4 bg-surface-raised">
      <Flex gap="density-sm" align="center" justify="between">
        <Text kind="body/bold/md" className="shrink-0">
          Judge Prompt
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
            aria-label={open ? 'Hide judge prompt' : 'Edit judge prompt'}
            aria-expanded={open}
            onClick={() => setOpen(!open)}
          >
            <Pencil className="size-3.5" aria-hidden />
          </Button>
        </Flex>
      </Flex>

      {malformed ? (
        <Banner kind="inline" status="error">
          The prompt has nested or unclosed <code>{'{{ }}'}</code> braces, so the evaluation would
          fail to parse it.
        </Banner>
      ) : null}

      {invalid.length > 0 ? (
        <Banner kind="inline" status="error">
          {invalid.map((name) => `{{${name}}}`).join(', ')} {invalid.length === 1 ? 'is' : 'are'}{' '}
          not available for this dataset, so the evaluation would fail on every row.
        </Banner>
      ) : null}

      {nonPortable.length > 0 ? (
        <Banner kind="inline" status="warning">
          {nonPortable.map(({ token }) => `{{${token}}}`).join(', ')} names a dataset field
          directly, so this configuration would only run on this dataset. Use{' '}
          {nonPortable.map(({ canonical }) => `{{${canonical}}}`).join(', ')} to keep it reusable.
        </Banner>
      ) : null}

      {open ? (
        <>
          <Text kind="body/regular/sm" className="text-secondary">
            Double-brace expressions are resolved against each dataset row. Editing this stops it
            following your field mappings.
          </Text>
          <VariableTextArea
            ref={editorRef}
            attributes={{ TextAreaElement: { 'aria-label': 'Judge prompt' } }}
            className="font-mono text-xs"
            variables={available}
            isKnown={isKnown}
            value={edited ?? generated}
            onChange={(next) => {
              // Null restores tracking, so an edit undone back to the generated text
              // resumes following the bindings rather than freezing a copy of them.
              setValue('body.judgePrompt', next === generated ? null : next, {
                shouldValidate: true,
              });
            }}
          />
        </>
      ) : null}
    </Stack>
  );
};
