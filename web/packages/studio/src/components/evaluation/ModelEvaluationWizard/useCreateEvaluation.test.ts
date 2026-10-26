// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { createToastMock } from '@nemo/common/src/providers/toast/useToastMock';
import { useEvalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import { createEvalConfigFileset } from '@studio/api/evaluation/eval-config-fileset';
import {
  type DatasetBindings,
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
} from '@studio/components/evaluation/ModelEvaluationWizard/types';
import { useCreateEvaluation } from '@studio/components/evaluation/ModelEvaluationWizard/useCreateEvaluation';
import { act, renderHook } from '@testing-library/react';
import { AxiosError, AxiosHeaders, type InternalAxiosRequestConfig } from 'axios';

vi.mock('@nemo/common/src/providers/toast/useToast');

vi.mock('@nemo/sdk/generated/evals/evals-plugin-jobs-routes', () => ({
  useEvalsCreateEvaluateJob: vi.fn(),
}));

vi.mock('@studio/api/evaluation/eval-config-fileset', () => ({
  createEvalConfigFileset: vi.fn(),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'ws',
}));

vi.mock('react-router', () => ({
  useNavigate: () => vi.fn(),
}));

const values: EvaluationFormValues = {
  ...EVALUATION_FORM_DEFAULTS,
  name: 'my-eval',
  model: 'ws/target-model',
  fieldMapping: { ...EVALUATION_FORM_DEFAULTS.fieldMapping, input: 'question' },
};

const bindings: DatasetBindings = {
  messagesColumn: null,
  input: '{{item.question}}',
  reference: null,
  context: null,
  inputPath: 'question',
  referencePath: null,
};

/** An AxiosError shaped like a real submit rejection, with the given body. */
const submitError = (status: number, data: unknown): AxiosError => {
  const config: InternalAxiosRequestConfig = { headers: new AxiosHeaders() };
  return new AxiosError(
    `Request failed with status code ${status}`,
    'ERR_BAD_REQUEST',
    config,
    undefined,
    { status, statusText: 'Unprocessable Entity', data, headers: {}, config }
  );
};

let toast = createToastMock();
const createJob = vi.fn();

beforeEach(() => {
  // The hook logs the same message it toasts; the logger writes to console.error.
  vi.spyOn(console, 'error').mockImplementation(() => {});
  toast = createToastMock();
  vi.mocked(useToast).mockReturnValue(toast);
  createJob.mockReset();
  vi.mocked(createEvalConfigFileset).mockResolvedValue(undefined as never);
  vi.mocked(useEvalsCreateEvaluateJob).mockReturnValue({
    mutateAsync: createJob,
    isPending: false,
  } as never);
});

const submit = async () => {
  const view = renderHook(() => useCreateEvaluation());
  await act(() => view.result.current.createEvaluation(values, bindings));
};

describe('useCreateEvaluation error reporting', () => {
  it('shows the server message in full, however long it is', async () => {
    const detail = 'X'.repeat(300);
    createJob.mockRejectedValue(submitError(422, { detail }));

    await submit();

    expect(toast.error).toHaveBeenCalledWith(detail);
  });

  it('shows a FastAPI validation detail rather than the axios message', async () => {
    createJob.mockRejectedValue(
      submitError(422, {
        detail: [
          { msg: 'Field is required', type: 'value_error', loc: ['body', 'spec', 'target'] },
        ],
      })
    );

    await submit();

    expect(toast.error).toHaveBeenCalledWith('spec.target: Field is required');
  });

  it('shows a {"message": ...} body', async () => {
    createJob.mockRejectedValue(submitError(400, { message: 'Dataset has no usable columns' }));

    await submit();

    expect(toast.error).toHaveBeenCalledWith('Dataset has no usable columns');
  });

  it('shows a plain-text body', async () => {
    createJob.mockRejectedValue(submitError(500, 'upstream evaluator is unavailable'));

    await submit();

    expect(toast.error).toHaveBeenCalledWith('upstream evaluator is unavailable');
  });
});
