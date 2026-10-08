// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LiveTestPanel } from '@studio/routes/evaluation/EvaluationNewRoute/LiveTestPanel';
import {
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FormProvider, type UseFormReturn, useForm } from 'react-hook-form';

const run = vi.fn();
const cancel = vi.fn();

vi.mock('@studio/routes/evaluation/EvaluationNewRoute/useLiveTest', () => ({
  useLiveTest: () => ({ state: { status: 'idle' }, run, cancel }),
}));

vi.mock('@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview', () => ({
  useDatasetPreview: () => ({ row: { question: 'q' }, rowCount: 1, isPartial: false }),
}));

vi.mock('@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings', () => ({
  useDatasetBindings: () => ({
    messagesColumn: null,
    input: '{{item.question}}',
    reference: null,
    context: null,
    inputPath: 'question',
    referencePath: null,
  }),
}));

let form: UseFormReturn<EvaluationFormValues>;

const Harness = () => {
  form = useForm<EvaluationFormValues>({
    defaultValues: { ...EVALUATION_FORM_DEFAULTS, model: 'ws/a', dataset: 'ws/files#data.jsonl' },
  });
  return (
    <FormProvider {...form}>
      <LiveTestPanel />
    </FormProvider>
  );
};

describe('LiveTestPanel', () => {
  beforeEach(() => {
    run.mockReset();
    cancel.mockReset();
  });

  it('does not cancel its own run when Test is clicked', async () => {
    render(<Harness />);
    cancel.mockClear();

    await userEvent.click(screen.getByRole('button', { name: 'Test' }));

    await waitFor(() => expect(run).toHaveBeenCalledTimes(1));
    expect(cancel).not.toHaveBeenCalled();
  });

  it.each([
    ['model', () => form.setValue('model', 'ws/b')],
    ['fieldMapping', () => form.setValue('fieldMapping.reference', 'answer')],
    ['metrics', () => form.setValue('body.metrics.exact-match', true)],
    ['judge model', () => form.setValue('body.judgeModel', 'ws/judge')],
  ])('cancels the live test when the %s changes', (_, edit) => {
    render(<Harness />);
    cancel.mockClear();

    act(edit);

    expect(cancel).toHaveBeenCalled();
  });

  it('leaves the live test alone when a field outside the request changes', () => {
    render(<Harness />);
    cancel.mockClear();

    act(() => form.setValue('name', 'renamed'));

    expect(cancel).not.toHaveBeenCalled();
  });
});
