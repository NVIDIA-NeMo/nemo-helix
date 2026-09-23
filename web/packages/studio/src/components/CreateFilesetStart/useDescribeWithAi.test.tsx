// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import {
  ERROR_NO_TOOL_CALL,
  ERROR_PARSE_RESPONSE,
  useDescribeWithAi,
} from '@studio/components/CreateFilesetStart/useDescribeWithAi';
import { act, renderHook, waitFor } from '@testing-library/react';

const mutateAsync = vi.fn();

vi.mock('@nemo/common/src/hooks/useChatCompletion', () => ({
  useChatCompletion: () => ({ mutateAsync, isPending: false }),
}));

/** Stands in for the panel's model search results, which the hook uses to resolve model configs. */
const MODEL_GROUPS: ModelWorkspaceGroup[] = [
  {
    workspace: 'default',
    models: [
      {
        id: 'default/llama-3.3',
        workspace: 'default',
        name: 'llama-3.3',
        model_providers: ['default/nim'],
      },
    ],
  } as unknown as ModelWorkspaceGroup,
];

const JOB_REQUEST = {
  name: 'qa-pairs',
  spec: {
    num_records: 25,
    config: {
      columns: [
        {
          name: 'topic',
          column_type: 'sampler',
          sampler_type: 'category',
          params: { values: ['science', 'history'] },
        },
      ],
    },
  },
};

const toolCallResponse = (args: unknown) => ({
  choices: [
    {
      message: {
        tool_calls: [{ function: { arguments: JSON.stringify(args) } }],
      },
    },
  ],
});

const setUp = ({ fill = true }: { fill?: boolean } = {}) => {
  const onValidConfig = vi.fn();
  const { result } = renderHook(() => useDescribeWithAi('default', onValidConfig, MODEL_GROUPS));
  if (fill) {
    act(() => {
      result.current.form.setValue('model', 'default/llama-3.3');
      result.current.form.setValue('provider', 'default/nim');
      result.current.form.setValue('prompt', '50 trivia questions by topic');
    });
  }
  return { result, onValidConfig };
};

describe('useDescribeWithAi', () => {
  beforeEach(() => {
    mutateAsync.mockReset();
  });

  it('does not call the model until both fields are filled in', async () => {
    const { result } = setUp({ fill: false });

    // Submitting empty: the form's own rules (asserted in the panel) stop the request.
    await act(() => result.current.generate());
    expect(mutateAsync).not.toHaveBeenCalled();

    act(() => {
      result.current.form.setValue('model', 'default/llama-3.3');
      result.current.form.setValue('prompt', '50 trivia questions by topic');
    });
    await act(() => result.current.generate());
    expect(mutateAsync).toHaveBeenCalledTimes(1);
  });

  it('reports a loadable draft and hands the job request to the caller', async () => {
    mutateAsync.mockResolvedValue(toolCallResponse({ job_request: JOB_REQUEST }));
    const { result, onValidConfig } = setUp();

    await act(() => result.current.generate());

    await waitFor(() => expect(result.current.validation?.status).toBe('valid'));
    expect(onValidConfig).toHaveBeenCalledWith(expect.objectContaining({ name: 'qa-pairs' }));
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({ workspace: 'default', model: 'llama-3.3', tool_choice: 'required' })
    );
  });

  it('rejects a reply with no tool call', async () => {
    mutateAsync.mockResolvedValue({ choices: [{ message: { content: 'sure!' } }] });
    const { result, onValidConfig } = setUp();

    await act(() => result.current.generate());

    expect(result.current.validation).toEqual({
      status: 'invalid',
      errors: [ERROR_NO_TOOL_CALL],
      warnings: [],
    });
    expect(onValidConfig).toHaveBeenCalledWith(null);
  });

  it('rejects tool arguments that are not a job request', async () => {
    mutateAsync.mockResolvedValue(toolCallResponse({ not: 'a job request' }));
    const { result, onValidConfig } = setUp();

    await act(() => result.current.generate());

    expect(result.current.validation).toEqual({
      status: 'invalid',
      errors: [ERROR_PARSE_RESPONSE],
      warnings: [],
    });
    expect(onValidConfig).toHaveBeenCalledWith(null);
  });

  it('keeps the raw model output for inspection, valid draft or not', async () => {
    mutateAsync.mockResolvedValue(toolCallResponse({ not: 'a job request' }));
    const { result } = setUp();

    await act(() => result.current.generate());

    // Pretty-printed, and kept even though the draft was rejected.
    expect(result.current.rawOutput).toBe('{\n  "not": "a job request"\n}');
  });

  describe('requestFix', () => {
    it('replays the draft and its issues, then settles on the reworked config', async () => {
      // First run is rejected: an image column the builder can't load leaves it with no columns.
      mutateAsync.mockResolvedValueOnce(
        toolCallResponse({
          job_request: {
            name: 'qa-pairs',
            spec: { num_records: 25, config: { columns: [{ name: 'art', column_type: 'image' }] } },
          },
        })
      );
      const { result, onValidConfig } = setUp();
      await act(() => result.current.generate());
      await waitFor(() => expect(result.current.validation?.status).toBe('invalid'));

      mutateAsync.mockResolvedValueOnce(toolCallResponse({ job_request: JOB_REQUEST }));
      await act(() => result.current.requestFix());

      const [{ messages }] = mutateAsync.mock.calls[1] as [{ messages: { content: string }[] }];
      // Original request, the config that failed, and what was wrong with it.
      expect(messages[1].content).toBe('50 trivia questions by topic');
      expect(messages[2].content).toContain('"art"');
      expect(messages[3].content).toContain('Errors');

      await waitFor(() => expect(result.current.validation?.status).toBe('valid'));
      expect(onValidConfig).toHaveBeenLastCalledWith(expect.objectContaining({ name: 'qa-pairs' }));
    });

    it('does nothing before a draft exists', async () => {
      const { result } = setUp();

      await act(() => result.current.requestFix());

      expect(mutateAsync).not.toHaveBeenCalled();
    });

    it('reports which request is in flight', async () => {
      mutateAsync.mockResolvedValue(toolCallResponse({ job_request: JOB_REQUEST }));
      const { result } = setUp();

      expect(result.current.pendingAction).toBeNull();
      await act(() => result.current.generate());
      expect(result.current.pendingAction).toBeNull();
    });
  });

  it('drops the previous draft as soon as a regeneration starts', async () => {
    mutateAsync.mockResolvedValueOnce(toolCallResponse({ job_request: JOB_REQUEST }));
    const { result, onValidConfig } = setUp();
    await act(() => result.current.generate());
    await waitFor(() => expect(result.current.validation?.status).toBe('valid'));

    // Second run held open: the caller must not still hold the first request while it is pending.
    let settleSecondRun: (response: unknown) => void = () => {};
    mutateAsync.mockReturnValueOnce(
      new Promise((resolve) => {
        settleSecondRun = resolve;
      })
    );
    let generation: Promise<void>;
    act(() => {
      generation = result.current.generate();
    });

    await waitFor(() => expect(result.current.pendingAction).toBe('generate'));
    expect(onValidConfig).toHaveBeenLastCalledWith(null);

    await act(async () => {
      settleSecondRun(toolCallResponse({ job_request: JOB_REQUEST }));
      await generation;
    });
    expect(onValidConfig).toHaveBeenLastCalledWith(expect.objectContaining({ name: 'qa-pairs' }));
  });

  it('clears a previously valid draft when a regeneration fails outright', async () => {
    mutateAsync.mockResolvedValueOnce(toolCallResponse({ job_request: JOB_REQUEST }));
    const { result, onValidConfig } = setUp();
    await act(() => result.current.generate());
    await waitFor(() => expect(result.current.validation?.status).toBe('valid'));

    mutateAsync.mockRejectedValueOnce(new Error('model is offline'));
    await act(() => result.current.generate());

    expect(result.current.requestError).toBe('model is offline');
    expect(result.current.validation).toBeNull();
    expect(onValidConfig).toHaveBeenLastCalledWith(null);
  });
});
