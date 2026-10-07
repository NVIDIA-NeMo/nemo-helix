// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DRAFT_TOOL_NAME } from '@studio/components/CreateCustomizationStart/aiInstructions';
import { automodelDraft, INPUTS } from '@studio/components/CreateCustomizationStart/testFixtures';
import {
  ERROR_INPUTS_NOT_READY,
  ERROR_NO_TOOL_CALL,
  MAX_RETRIES,
  useDescribeWithAi,
} from '@studio/components/CreateCustomizationStart/useDescribeWithAi';
import { act, renderHook, waitFor } from '@testing-library/react';

const mutateAsync = vi.fn();

vi.mock('@nemo/common/src/hooks/useChatCompletion', () => ({
  useChatCompletion: () => ({ mutateAsync, isPending: false }),
}));

const toolCallResponse = (args: unknown) => ({
  choices: [
    {
      message: {
        tool_calls: [
          {
            type: 'function',
            function: { name: DRAFT_TOOL_NAME, arguments: JSON.stringify(args) },
          },
        ],
      },
    },
  ],
});

const setUp = ({ fill = true }: { fill?: boolean } = {}) => {
  const onDraft = vi.fn();
  const { result } = renderHook(() => useDescribeWithAi('default', onDraft));
  if (fill) {
    act(() => {
      result.current.form.setValue('model', 'default/drafter');
      result.current.form.setValue('baseModel', 'default/llama-8b');
      result.current.form.setValue('dataset', 'default/tickets');
      result.current.form.setValue('prompt', 'route support tickets');
    });
  }
  return { result, onDraft };
};

describe('useDescribeWithAi', () => {
  beforeEach(() => {
    mutateAsync.mockReset();
  });

  it('does not call the model until both fields are filled in', async () => {
    const { result } = setUp({ fill: false });
    await act(() => result.current.generate(INPUTS));
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it('says so instead of doing nothing when the picks have not been read', async () => {
    const { result } = setUp();
    await act(() => result.current.generate(null));
    expect(mutateAsync).not.toHaveBeenCalled();
    expect(result.current.requestError).toBe(ERROR_INPUTS_NOT_READY);
  });

  it('forces a low-temperature tool call with the picked inputs in the request', async () => {
    mutateAsync.mockResolvedValue(toolCallResponse(automodelDraft()));
    const { result } = setUp();
    await act(() => result.current.generate(INPUTS));

    const request = mutateAsync.mock.calls[0][0];
    expect(request).toMatchObject({
      workspace: 'default',
      model: 'drafter',
      tool_choice: 'required',
      temperature: 0.2,
    });
    expect(request.tools[0].function.name).toBe(DRAFT_TOOL_NAME);
    expect(request.messages[1].content).toContain('## Dataset\ndefault/tickets');
  });

  it('hands valid drafts over as form values', async () => {
    mutateAsync.mockResolvedValue(toolCallResponse(automodelDraft()));
    const { result, onDraft } = setUp();
    await act(() => result.current.generate(INPUTS));

    expect(result.current.validation?.status).toBe('valid');
    expect(onDraft).toHaveBeenLastCalledWith(
      expect.objectContaining({ backend: 'automodel', outputName: 'llama-8b-ticket-router' })
    );
  });

  it('reports a reply without a tool call as an invalid draft', async () => {
    mutateAsync.mockResolvedValue({ choices: [{ message: { content: 'Which model?' } }] });
    const { result, onDraft } = setUp();
    await act(() => result.current.generate(INPUTS));

    expect(result.current.validation).toEqual({
      status: 'invalid',
      errors: [ERROR_NO_TOOL_CALL],
    });
    expect(onDraft).toHaveBeenLastCalledWith(null);
  });

  it('points at the context window when the instructions do not fit', async () => {
    mutateAsync.mockRejectedValue(new Error("This model's maximum context length is 16384 tokens"));
    const { result } = setUp();
    await act(() => result.current.generate(INPUTS));

    expect(result.current.requestError).toMatch(/larger context/);
  });

  it('keeps a request failure apart from a bad draft', async () => {
    mutateAsync.mockRejectedValue(new Error('401 Unauthorized'));
    const { result } = setUp();
    await act(() => result.current.generate(INPUTS));

    expect(result.current.requestError).toBe('401 Unauthorized');
    expect(result.current.validation).toBeNull();
  });

  it('clears the draft when the picks it was built on change', async () => {
    mutateAsync.mockResolvedValue(toolCallResponse(automodelDraft()));
    const { result, onDraft } = setUp();
    await act(() => result.current.generate(INPUTS));
    expect(result.current.validation?.status).toBe('valid');

    act(() => result.current.clearDraft());
    expect(result.current.validation).toBeNull();
    expect(onDraft).toHaveBeenLastCalledWith(null);
  });

  it('hands validation errors back to the model and accepts the fixed job', async () => {
    const broken = automodelDraft({ training: { training_type: 'sft', lora_rank: 8 } });
    mutateAsync
      .mockResolvedValueOnce(toolCallResponse(broken))
      .mockResolvedValueOnce(toolCallResponse(automodelDraft()));
    const { result, onDraft } = setUp();
    await act(() => result.current.generate(INPUTS));

    expect(mutateAsync).toHaveBeenCalledTimes(2);
    const retryMessages = mutateAsync.mock.calls[1][0].messages;
    expect(retryMessages.at(-1).content).toContain('training.lora_rank: not a field');
    expect(result.current.validation?.status).toBe('valid');
    expect(onDraft).toHaveBeenLastCalledWith(expect.objectContaining({ backend: 'automodel' }));
  });

  it(`stops after ${MAX_RETRIES} retries and shows what is still wrong`, async () => {
    const broken = automodelDraft({ training: { training_type: 'sft', lora_rank: 8 } });
    mutateAsync.mockResolvedValue(toolCallResponse(broken));
    const { result, onDraft } = setUp();
    await act(() => result.current.generate(INPUTS));

    expect(mutateAsync).toHaveBeenCalledTimes(MAX_RETRIES + 1);
    expect(result.current.validation).toEqual({
      status: 'invalid',
      errors: ['training.lora_rank: not a field of the automodel job schema'],
    });
    expect(onDraft).toHaveBeenLastCalledWith(null);
  });

  it('drops a reply that lands after the panel unmounts', async () => {
    let resolve: (value: unknown) => void = () => {};
    mutateAsync.mockReturnValue(new Promise((r) => (resolve = r)));
    const onDraft = vi.fn();
    const { result, unmount } = renderHook(() => useDescribeWithAi('default', onDraft));
    act(() => {
      result.current.form.setValue('model', 'default/drafter');
      result.current.form.setValue('baseModel', 'default/llama-8b');
      result.current.form.setValue('dataset', 'default/tickets');
      result.current.form.setValue('prompt', 'route support tickets');
    });

    let pending: Promise<void> = Promise.resolve();
    act(() => {
      pending = result.current.generate(INPUTS);
    });
    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    const { signal } = mutateAsync.mock.calls[0][0];

    unmount();
    expect(signal.aborted).toBe(true);
    resolve(toolCallResponse(automodelDraft()));
    await pending;

    expect(onDraft).not.toHaveBeenCalledWith(expect.objectContaining({ backend: 'automodel' }));
  });
});
