// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text, TextArea } from '@nvidia/foundations-react-core';
import {
  composeJudgeUserPrompt,
  type EvaluationFormValues,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import { Pencil } from 'lucide-react';
import { FC, useState } from 'react';
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
  const [open, setOpen] = useState(false);

  const generated = composeJudgeUserPrompt(bindings);

  return (
    <Stack gap="density-xs" className="border border-base rounded-lg p-4 bg-surface-raised">
      <Flex gap="density-sm" align="center" justify="between">
        <Text kind="body/bold/md" className="shrink-0">
          Judge Prompt
        </Text>
        <Button
          type="button"
          kind="tertiary"
          size="small"
          aria-label={open ? 'Hide judge prompt' : 'Edit judge prompt'}
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className="shrink-0"
        >
          <Pencil className="size-3.5" aria-hidden />
        </Button>
      </Flex>

      {open ? (
        <>
          <Text kind="body/regular/sm" className="text-secondary">
            Double-brace expressions are resolved against each dataset row. Editing this stops it
            following your field mappings.
          </Text>
          <TextArea
            aria-label="Judge prompt"
            resizeable="auto"
            className="font-mono text-xs"
            value={edited ?? generated}
            onChange={(event) => {
              // Null restores tracking, so an edit undone back to the generated text
              // resumes following the bindings rather than freezing a copy of them.
              const next = event.target.value;
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
