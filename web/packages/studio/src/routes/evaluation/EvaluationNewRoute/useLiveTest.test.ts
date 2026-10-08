// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evalsRunEvaluateLiveEvaluation } from '@nemo/sdk/generated/evals/evals-plugin-live-evaluation-route';
import type { LiveScoreResponse } from '@nemo/sdk/generated/evals/schema';
import {
  composeGenerationPrompt,
  type DatasetBindings,
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useLiveTest } from '@studio/routes/evaluation/EvaluationNewRoute/useLiveTest';
import { getModelInferenceGatewayUrl } from '@studio/util/models';
import { act, renderHook } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/evals/evals-plugin-live-evaluation-route', () => ({
  evalsRunEvaluateLiveEvaluation: vi.fn(),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => 'ws',
}));

const runLive = vi.mocked(evalsRunEvaluateLiveEvaluation);

const values: EvaluationFormValues = {
  ...EVALUATION_FORM_DEFAULTS,
  model: 'ws/target-model',
  fieldMapping: { ...EVALUATION_FORM_DEFAULTS.fieldMapping, input: 'question' },
  body: {
    ...EVALUATION_FORM_DEFAULTS.body,
    metrics: { ...EVALUATION_FORM_DEFAULTS.body.metrics, 'llm-judge': true },
    judgeModel: 'ws/judge-model',
  },
};

const bindings: DatasetBindings = {
  messagesColumn: null,
  input: '{{item.question}}',
  reference: null,
  context: null,
  inputPath: 'question',
  referencePath: null,
};

const row = { question: 'What is 2 + 2?' };

const runTest = async () => {
  const view = renderHook(() => useLiveTest());
  await act(() => view.result.current.run(values, bindings, row));
  return view;
};

describe('useLiveTest', () => {
  beforeEach(() => {
    runLive.mockReset();
  });

  it('sends one request built like a real run, with the abort signal', async () => {
    runLive.mockResolvedValue({ output: '4', metrics: [] });
    await runTest();

    expect(runLive).toHaveBeenCalledTimes(1);
    const [workspace, request, signal] = runLive.mock.calls[0];
    expect(workspace).toBe('ws');
    expect(request.dataset).toEqual([row]);
    expect(request.target).toEqual({
      url: getModelInferenceGatewayUrl('ws', 'ws/target-model'),
      name: 'target-model',
    });
    expect(request.prompt_template).toEqual(composeGenerationPrompt(bindings));
    expect(request.field_mapping).toEqual({ input: 'question' });
    expect(request.params).toBeUndefined();
    expect(signal).toBeInstanceOf(AbortSignal);
  });

  it('reports the server output, scores and per-metric errors', async () => {
    const response: LiveScoreResponse = {
      output: '4',
      metrics: [
        {
          metric: 'exact-match',
          scores: [{ name: 'exact-match', mean: 1, nan_count: 0, value: 1 }],
        },
        { metric: 'llm-judge', scores: [], error: 'judge returned 502' },
      ],
    };
    runLive.mockResolvedValue(response);
    const { result } = await runTest();

    expect(result.current.state).toEqual({
      status: 'done',
      result: {
        output: '4',
        scores: [{ name: 'exact-match', value: 1, label: undefined }],
        errors: [{ metric: 'llm-judge', message: 'judge returned 502' }],
      },
    });
  });

  it('surfaces the server detail on a failed request', async () => {
    runLive.mockRejectedValue({
      response: { status: 502, data: { detail: 'target generation failed: 404 Not Found' } },
    });
    const { result } = await runTest();

    expect(result.current.state).toEqual({
      status: 'error',
      message: 'The live test failed (502). target generation failed: 404 Not Found',
    });
  });

  it('joins a validation error list into the message', async () => {
    runLive.mockRejectedValue({
      response: { status: 422, data: { detail: [{ msg: 'too short' }, { msg: 'bad field' }] } },
    });
    const { result } = await runTest();

    expect(result.current.state).toEqual({
      status: 'error',
      message: 'The live test failed (422). too short; bad field',
    });
  });

  it('fails fast without a request when the input is empty', async () => {
    const view = renderHook(() => useLiveTest());
    await act(() => view.result.current.run(values, bindings, { question: '' }));

    expect(runLive).not.toHaveBeenCalled();
    expect(view.result.current.state.status).toBe('error');
  });

  it('aborts the request and ignores its result once cancelled', async () => {
    let resolve: (response: LiveScoreResponse) => void = () => {};
    runLive.mockReturnValue(new Promise((settle) => (resolve = settle)));
    const { result } = renderHook(() => useLiveTest());

    let pending: Promise<void> = Promise.resolve();
    act(() => {
      pending = result.current.run(values, bindings, row);
    });
    act(() => result.current.cancel());
    await act(async () => {
      resolve({ output: '4', metrics: [] });
      await pending;
    });

    expect(runLive.mock.calls[0][2]?.aborted).toBe(true);
    expect(result.current.state).toEqual({ status: 'idle' });
  });
});
