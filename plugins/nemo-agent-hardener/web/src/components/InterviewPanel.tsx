// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InterviewAnswer, InterviewPrompt } from '@agent-hardener/components/hitlTypes';
import { RadioCard } from '@nemo/common';
import {
  Button,
  Card,
  Flex,
  FormField,
  RadioGroupRoot,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { FC, FormEvent, useState } from 'react';

interface InterviewPanelProps {
  prompt: InterviewPrompt;
  loading?: boolean;
  onSubmit: (answers: InterviewAnswer[]) => void;
}

const defaultAnswer = (options?: InterviewPrompt['questions'][number]['options']): string =>
  options?.find((o) => o.recommended)?.description ?? options?.[0]?.description ?? '';

// Sentinel radio value for the "Other" escape hatch, so a question with options still accepts free text.
const OTHER = '__other__';

// The synth interview, rendered inline (manifest generate flow, or a run's fallback tab): one card per
// question, options as radio cards with an "Other" free-text escape hatch. Returns the operator's answers
// so the job can resume benign-suite synthesis.
export const InterviewPanel: FC<InterviewPanelProps> = ({ prompt, loading, onSubmit }) => {
  const [answers, setAnswers] = useState<Record<string, string>>(() =>
    Object.fromEntries(prompt.questions.map((q) => [q.gap, defaultAnswer(q.options)]))
  );
  const [otherText, setOtherText] = useState<Record<string, string>>({});

  const setAnswer = (gap: string, value: string) =>
    setAnswers((prev) => ({ ...prev, [gap]: value }));
  const setOther = (gap: string, value: string) =>
    setOtherText((prev) => ({ ...prev, [gap]: value }));

  // "Other" answers submit the typed text, not the sentinel.
  const resolve = (gap: string): string =>
    answers[gap] === OTHER ? (otherText[gap] ?? '') : (answers[gap] ?? '');

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSubmit(
      prompt.questions.map((q) => ({ gap: q.gap, question: q.question, answer: resolve(q.gap) }))
    );
  };

  return (
    <form onSubmit={handleSubmit} className="flex h-full flex-col">
      <Stack gap="density-xs" className="mb-4 shrink-0">
        <Text kind="body/semibold/lg">Answer the synth interview</Text>
        <Text kind="body/regular/md" className="text-subtle">
          Your answers shape the benign test suite the war-game replays against the agent.{' '}
          {prompt.questions.length} question{prompt.questions.length === 1 ? '' : 's'}.
        </Text>
      </Stack>

      <Stack gap="density-lg" className="min-h-0 flex-1 overflow-auto pr-density-xs">
        {prompt.questions.map((q, index) => (
          <Card key={q.gap} className="p-4">
            <Stack gap="density-md">
              <Text kind="body/semibold/md">
                <span className="text-subtle">{index + 1}. </span>
                {q.question}
              </Text>
              {q.options && q.options.length > 0 ? (
                <Stack gap="density-md">
                  <RadioGroupRoot
                    name={q.gap}
                    value={answers[q.gap]}
                    onValueChange={(value) => setAnswer(q.gap, value)}
                    className="w-full"
                  >
                    <Stack gap="3">
                      {q.options.map((o) => (
                        <RadioCard
                          key={o.description}
                          value={o.description}
                          label={o.label || o.description}
                          description={
                            o.recommended ? `${o.description} (recommended)` : o.description
                          }
                        />
                      ))}
                      <RadioCard value={OTHER} label="Other" description="Write your own answer" />
                    </Stack>
                  </RadioGroupRoot>
                  {answers[q.gap] === OTHER ? (
                    <FormField name={`${q.gap}_other`} slotLabel="Your Answer">
                      <TextInput
                        value={otherText[q.gap] ?? ''}
                        onChange={(e) => setOther(q.gap, e.target.value)}
                      />
                    </FormField>
                  ) : null}
                </Stack>
              ) : (
                <FormField name={q.gap}>
                  <TextInput
                    value={answers[q.gap] ?? ''}
                    onChange={(e) => setAnswer(q.gap, e.target.value)}
                  />
                </FormField>
              )}
            </Stack>
          </Card>
        ))}
      </Stack>

      <Flex className="mt-4 shrink-0 justify-end">
        <Button color="brand" type="submit" disabled={loading}>
          {loading ? 'Submitting…' : 'Submit answers'}
        </Button>
      </Flex>
    </form>
  );
};
