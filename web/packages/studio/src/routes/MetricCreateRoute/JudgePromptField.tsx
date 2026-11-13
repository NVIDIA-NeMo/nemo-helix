// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VariableButton } from '@nemo/common/src/components/buttons/VariableButton';
import {
  type VariableDef,
  VariableTextArea,
  type VariableTextAreaHandle,
} from '@nemo/common/src/components/form/VariableTextArea';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { composeJudgeUserPrompt } from '@studio/components/evaluation/EvalWizard/types';
import { CANONICAL_BINDINGS, type MetricFormValues } from '@studio/routes/MetricCreateRoute/types';
import { type FC, useRef } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const VARIABLES: VariableDef[] = [
  { name: 'input', description: 'The prompt sent to the model' },
  { name: 'reference', description: 'The expected answer, when the dataset has one' },
  { name: 'context', description: 'Retrieved or supporting context, when the dataset has one' },
  { name: 'sample.output_text', description: "The model's response, which is what gets graded" },
];

export const DEFAULT_JUDGE_PROMPT = composeJudgeUserPrompt(CANONICAL_BINDINGS);

/** The judge's user message. `null` in the form means "the default", so an edit
 *  undone back to it is stored as untouched. */
export const JudgePromptField: FC = () => {
  const { control, setValue } = useFormContext<MetricFormValues>();
  const prompt = useWatch({ control, name: 'body.judgePrompt' });
  const editorRef = useRef<VariableTextAreaHandle>(null);

  return (
    <Stack gap="density-xs">
      <Flex align="center" justify="between">
        <Text kind="label/bold/md">Judge Prompt</Text>
        <VariableButton
          variables={VARIABLES}
          onSelect={(variable) => editorRef.current?.insertVariable(variable.name)}
        />
      </Flex>
      <Text kind="body/regular/sm" className="text-secondary">
        Sent to the judge for every row. Variables resolve against each row when an evaluation runs.
      </Text>
      <VariableTextArea
        ref={editorRef}
        attributes={{ TextAreaElement: { 'aria-label': 'Judge prompt' } }}
        className="font-mono text-xs"
        variables={VARIABLES}
        minRows={4}
        value={prompt ?? DEFAULT_JUDGE_PROMPT}
        onChange={(next) =>
          setValue('body.judgePrompt', next === DEFAULT_JUDGE_PROMPT ? null : next, {
            shouldDirty: true,
          })
        }
      />
    </Stack>
  );
};
