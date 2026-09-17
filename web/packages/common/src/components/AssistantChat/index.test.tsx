// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AssistantRuntimeProvider,
  type ThreadMessageLike,
  type ToolCallMessagePartComponent,
  useExternalStoreRuntime,
} from '@assistant-ui/react';
import { AssistantChatThread } from '@nemo/common/src/components/AssistantChat/AssistantChatThread';
import {
  ThemeProvider as KaizenThemeProvider,
  TooltipProvider,
} from '@nvidia/foundations-react-core';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ChatCompletion, ChatCompletionChunk } from 'openai/resources/index.mjs';
import type { Stream } from 'openai/streaming.mjs';
import { type ReactElement, useState } from 'react';

import { AssistantChat } from './index';

const mocks = vi.hoisted(() => ({
  createChatCompletion: vi.fn(),
}));

vi.mock('../../hooks/useChatCompletion', () => ({
  useChatCompletion: () => ({
    mutateAsync: mocks.createChatCompletion,
  }),
}));

const createCompletion = (content: string): ChatCompletion => ({
  id: 'completion-id',
  object: 'chat.completion',
  created: 1740419165,
  model: 'test-model',
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content,
        refusal: null,
      },
      finish_reason: 'stop',
      logprobs: null,
    },
  ],
});

const createCompletionChunk = (content: string): ChatCompletionChunk => ({
  id: 'completion-id',
  object: 'chat.completion.chunk',
  created: 1740419165,
  model: 'test-model',
  choices: [
    {
      index: 0,
      delta: content ? { content } : {},
      finish_reason: null,
      logprobs: null,
    },
  ],
});

const createImageCompletionChunk = (imageUrl: string): ChatCompletionChunk =>
  ({
    id: 'completion-id',
    object: 'chat.completion.chunk',
    created: 1740419165,
    model: 'test-image-model',
    choices: [
      {
        index: 0,
        delta: { images: [{ image_url: { url: imageUrl } }] },
        finish_reason: null,
        logprobs: null,
      },
    ],
  }) as unknown as ChatCompletionChunk;

const createHangingStream = (content: string): Stream<ChatCompletionChunk> => {
  const controller = new AbortController();
  const waitForAbort = () =>
    new Promise<void>((resolve) => {
      if (controller.signal.aborted) {
        resolve();
        return;
      }

      controller.signal.addEventListener('abort', () => resolve(), { once: true });
    });

  return {
    controller,
    async *[Symbol.asyncIterator]() {
      yield createCompletionChunk(content);
      await waitForAbort();
    },
  } as unknown as Stream<ChatCompletionChunk>;
};

const completion = createCompletion('Hello from inference gateway.');
const interactionTimeoutMs = 10_000;
const defaultStaticMessages: readonly ThreadMessageLike[] = [
  {
    role: 'user',
    content: [{ type: 'text', text: 'Existing prompt' }],
  },
  {
    role: 'assistant',
    content: [{ type: 'text', text: 'Existing response' }],
    status: { type: 'complete', reason: 'stop' },
  },
];

interface CompletionRequestWithSignal {
  signal?: AbortSignal;
}

const renderAssistantChat = (element: ReactElement) =>
  render(
    <KaizenThemeProvider className="h-full" density="standard" theme="light">
      <TooltipProvider>{element}</TooltipProvider>
    </KaizenThemeProvider>
  );

const StaticAssistantChatThread = ({
  hideAssistantMessageActions,
  isRunning = false,
  messages: initialMessages = defaultStaticMessages,
  toolCallPartComponent,
}: {
  hideAssistantMessageActions?: boolean;
  isRunning?: boolean;
  messages?: readonly ThreadMessageLike[];
  toolCallPartComponent?: ToolCallMessagePartComponent;
}) => {
  const [messages, setMessages] = useState<readonly ThreadMessageLike[]>(initialMessages);
  const runtime = useExternalStoreRuntime<ThreadMessageLike>({
    messages,
    setMessages,
    isRunning,
    onNew: async () => undefined,
    onEdit: async () => undefined,
    onReload: async () => undefined,
    onCancel: async () => undefined,
    convertMessage: (message) => message,
    unstable_capabilities: {
      copy: true,
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <AssistantChatThread
        hideAssistantMessageActions={hideAssistantMessageActions}
        placeholder="Task prompt"
        onReset={() => undefined}
        toolCallPartComponent={toolCallPartComponent}
      />
    </AssistantRuntimeProvider>
  );
};

describe('AssistantChat', () => {
  beforeEach(() => {
    mocks.createChatCompletion.mockReset();
    mocks.createChatCompletion.mockResolvedValue(completion);
  });

  it(
    'sends text prompts through useChatCompletion',
    async () => {
      renderAssistantChat(<AssistantChat model="test-model" workspace="default" />);

      await userEvent.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Hello model');
      await userEvent.click(screen.getByRole('button', { name: /Submit/i }));

      expect(await screen.findByText('Hello from inference gateway.')).toBeInTheDocument();
      expect(screen.getByText('Hello model')).toBeInTheDocument();

      await waitFor(() =>
        expect(mocks.createChatCompletion).toHaveBeenCalledWith(
          expect.objectContaining({
            model: 'test-model',
            workspace: 'default',
            stream: true,
            messages: [{ role: 'user', content: 'Hello model' }],
          })
        )
      );
    },
    interactionTimeoutMs
  );

  it(
    'omits reasoning_effort unless reasoning is turned off',
    async () => {
      renderAssistantChat(<AssistantChat model="test-model" workspace="default" />);

      await userEvent.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Hello model');
      await userEvent.click(screen.getByRole('button', { name: /Submit/i }));

      await waitFor(() =>
        expect(mocks.createChatCompletion).toHaveBeenCalledWith(
          expect.objectContaining({
            reasoning_effort: undefined,
            chat_template_kwargs: undefined,
          })
        )
      );
    },
    interactionTimeoutMs
  );

  it(
    'sends both reasoning-off conventions when reasoning is disabled',
    async () => {
      renderAssistantChat(
        <AssistantChat model="test-model" workspace="default" reasoningEnabled={false} />
      );

      await userEvent.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Hello model');
      await userEvent.click(screen.getByRole('button', { name: /Submit/i }));

      await waitFor(() =>
        expect(mocks.createChatCompletion).toHaveBeenCalledWith(
          expect.objectContaining({
            reasoning_effort: 'none',
            chat_template_kwargs: { enable_thinking: false },
          })
        )
      );
    },
    interactionTimeoutMs
  );

  it('allows a caller to render trusted Markdown links in messages', async () => {
    mocks.createChatCompletion.mockResolvedValueOnce(
      createCompletion('[Trace source](#zoomer-node=summary-1)')
    );
    renderAssistantChat(
      <AssistantChat
        model="test-model"
        workspace="default"
        messageContentProps={{
          markdownLinkComponent: ({ href, children }) => <a href={href}>{children}</a>,
        }}
      />
    );

    await userEvent.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Show evidence');
    await userEvent.click(screen.getByRole('button', { name: /Submit/i }));

    expect(await screen.findByRole('link', { name: 'Trace source' })).toHaveAttribute(
      'href',
      '#zoomer-node=summary-1'
    );
  });

  it('renders base64 images returned by an image model stream', async () => {
    const imageUrl = 'data:image/png;base64,iVBORw0KGgo=';
    const stream = {
      controller: new AbortController(),
      async *[Symbol.asyncIterator]() {
        yield createImageCompletionChunk(imageUrl);
      },
    } as unknown as Stream<ChatCompletionChunk>;
    mocks.createChatCompletion.mockResolvedValueOnce(stream);

    renderAssistantChat(<AssistantChat model="test-image-model" workspace="default" />);

    await userEvent.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Make an image');
    await userEvent.click(screen.getByRole('button', { name: /Submit/i }));

    expect(await screen.findByRole('img', { name: /Attached image/i })).toHaveAttribute(
      'src',
      imageUrl
    );
  });

  it(
    'clears the current thread',
    async () => {
      renderAssistantChat(
        <AssistantChat
          model="test-model"
          initialMessages={[
            {
              role: 'user',
              content: [{ type: 'text', text: 'Existing message' }],
            },
          ]}
        />
      );

      expect(screen.getByText('Existing message')).toBeInTheDocument();

      await userEvent.click(screen.getByRole('button', { name: /Reset/i }));

      expect(screen.queryByText('Existing message')).not.toBeInTheDocument();
      expect(screen.getByText('Ready')).toBeInTheDocument();
    },
    interactionTimeoutMs
  );

  it('shows copy and regenerate message actions by default', () => {
    renderAssistantChat(
      <AssistantChat
        model="test-model"
        initialMessages={[
          {
            role: 'user',
            content: [{ type: 'text', text: 'Existing prompt' }],
          },
          {
            role: 'assistant',
            content: [{ type: 'text', text: 'Existing response' }],
            status: { type: 'complete', reason: 'stop' },
          },
        ]}
      />
    );

    const assistantMessage = screen
      .getAllByTestId('assistant-chat-message')
      .find((message) => message.getAttribute('data-testspeaker') === 'assistant');
    if (!assistantMessage) throw new Error('Expected assistant message');

    expect(screen.getByText('Existing prompt')).toBeInTheDocument();
    expect(screen.getByText('Existing response')).toBeInTheDocument();
    expect(
      within(assistantMessage).getByRole('button', { name: /Copy message/i })
    ).toBeInTheDocument();
    expect(
      within(assistantMessage).getByRole('button', { name: /Regenerate response/i })
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Edit message/i })).toBeInTheDocument();
  });

  it('can hide assistant copy and regenerate message actions', () => {
    renderAssistantChat(<StaticAssistantChatThread hideAssistantMessageActions />);

    const assistantMessage = screen
      .getAllByTestId('assistant-chat-message')
      .find((message) => message.getAttribute('data-testspeaker') === 'assistant');
    if (!assistantMessage) throw new Error('Expected assistant message');

    expect(screen.getByText('Existing prompt')).toBeInTheDocument();
    expect(screen.getByText('Existing response')).toBeInTheDocument();
    expect(
      within(assistantMessage).queryByRole('button', { name: /Copy message/i })
    ).not.toBeInTheDocument();
    expect(
      within(assistantMessage).queryByRole('button', { name: /Regenerate response/i })
    ).not.toBeInTheDocument();
    expect(
      within(assistantMessage).queryByTestId('assistant-chat-message-actions')
    ).not.toBeInTheDocument();
    expect(
      within(assistantMessage).queryByTestId('assistant-chat-running-indicator')
    ).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Edit message/i })).toBeInTheDocument();
  });

  it('renders tool-only assistant messages without the assistant response surface', () => {
    renderAssistantChat(
      <StaticAssistantChatThread
        hideAssistantMessageActions
        messages={[
          {
            role: 'assistant',
            content: [
              {
                type: 'tool-call',
                toolCallId: 'toolu_read',
                toolName: 'Read',
                args: { file_path: 'README.md' },
              },
            ],
            status: { type: 'complete', reason: 'stop' },
          },
        ]}
        toolCallPartComponent={({ toolName }) => (
          <div data-testid="assistant-chat-tool-pill">{toolName}</div>
        )}
      />
    );

    const assistantMessage = screen.getByTestId('assistant-chat-message');

    expect(within(assistantMessage).getByTestId('assistant-chat-tool-pill')).toHaveTextContent(
      'Read'
    );
    expect(
      within(assistantMessage).queryByTestId('assistant-chat-message-surface')
    ).not.toBeInTheDocument();
  });

  it('renders streamed text and tool-call parts as separate message blocks', () => {
    renderAssistantChat(
      <StaticAssistantChatThread
        hideAssistantMessageActions
        messages={[
          {
            role: 'assistant',
            content: [
              { type: 'text', text: 'I will inspect the repository.' },
              {
                type: 'tool-call',
                toolCallId: 'toolu_read',
                toolName: 'Read',
                args: { file_path: 'README.md' },
              },
              { type: 'text', text: 'I found the relevant file.' },
            ],
            status: { type: 'running' },
          },
        ]}
        toolCallPartComponent={({ toolName }) => (
          <div data-testid="assistant-chat-tool-pill">{toolName}</div>
        )}
      />
    );

    const assistantMessage = screen.getByTestId('assistant-chat-message');
    const messageSurfaces = within(assistantMessage).getAllByTestId(
      'assistant-chat-message-surface'
    );
    const toolCall = within(assistantMessage).getByTestId('assistant-chat-tool-pill');

    expect(messageSurfaces).toHaveLength(2);
    expect(messageSurfaces[0]).toHaveTextContent('I will inspect the repository.');
    expect(messageSurfaces[1]).toHaveTextContent('I found the relevant file.');
    expect(toolCall).toHaveTextContent('Read');
    for (const messageSurface of messageSurfaces) {
      expect(
        within(messageSurface).queryByTestId('assistant-chat-tool-pill')
      ).not.toBeInTheDocument();
    }
  });

  it('shows the running skeleton without an empty assistant response surface', () => {
    renderAssistantChat(
      <StaticAssistantChatThread
        hideAssistantMessageActions
        isRunning
        messages={[
          {
            role: 'assistant',
            content: [{ type: 'text', text: '' }],
            status: { type: 'running' },
          },
        ]}
      />
    );

    const assistantMessage = screen.getByTestId('assistant-chat-message');

    expect(
      within(assistantMessage).queryByTestId('assistant-chat-message-surface')
    ).not.toBeInTheDocument();
    expect(
      within(assistantMessage).getByTestId('assistant-chat-running-indicator')
    ).toBeInTheDocument();
    expect(within(assistantMessage).getByTestId('assistant-chat-skeleton')).toBeInTheDocument();
  });

  it('hides an empty text surface while a tool is running', () => {
    renderAssistantChat(
      <StaticAssistantChatThread
        hideAssistantMessageActions
        isRunning
        messages={[
          {
            role: 'assistant',
            content: [
              { type: 'text', text: '  \n' },
              {
                type: 'tool-call',
                toolCallId: 'toolu_read',
                toolName: 'Read',
                args: { file_path: 'README.md' },
              },
            ],
            status: { type: 'running' },
          },
        ]}
        toolCallPartComponent={({ toolName }) => (
          <div data-testid="assistant-chat-tool-pill">{toolName}</div>
        )}
      />
    );

    const assistantMessage = screen.getByTestId('assistant-chat-message');

    expect(within(assistantMessage).getByTestId('assistant-chat-tool-pill')).toHaveTextContent(
      'Read'
    );
    expect(
      within(assistantMessage).queryByTestId('assistant-chat-message-surface')
    ).not.toBeInTheDocument();
    expect(within(assistantMessage).getByTestId('assistant-chat-skeleton')).toBeInTheDocument();
  });

  it('offers an enabled add-image affordance for a vision model when enabled', () => {
    renderAssistantChat(
      <AssistantChat model="test-vision-model" workspace="default" enableImageAttachments />
    );

    const addImageButton = screen.getByRole('button', { name: /Add image/i });
    expect(addImageButton).toBeInTheDocument();
    expect(addImageButton).toBeEnabled();
  });

  it('hides the add-image affordance when image attachments are disabled', () => {
    renderAssistantChat(
      <AssistantChat model="test-vision-model" workspace="default" enableImageAttachments={false} />
    );

    expect(screen.queryByRole('button', { name: /Add image/i })).not.toBeInTheDocument();
  });

  it('hides the add-image affordance for a non-vision model even when enabled', () => {
    renderAssistantChat(
      <AssistantChat model="test-model" workspace="default" enableImageAttachments />
    );

    expect(screen.queryByRole('button', { name: /Add image/i })).not.toBeInTheDocument();
  });

  it('disables the add-image affordance when the chat is disabled', () => {
    renderAssistantChat(
      <AssistantChat
        model="test-vision-model"
        workspace="default"
        enableImageAttachments
        disabled
      />
    );

    expect(screen.getByRole('button', { name: /Add image/i })).toBeDisabled();
  });

  it('renders image content parts in a user message', () => {
    const imageUrl = 'data:image/png;base64,AAAA';
    renderAssistantChat(
      <AssistantChat
        model="test-model"
        initialMessages={[
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Look at this' },
              { type: 'image', image: imageUrl },
            ],
          },
        ]}
      />
    );

    expect(screen.getByText('Look at this')).toBeInTheDocument();
    const image = screen.getByRole('img', { name: /Attached image/i });
    expect(image).toHaveAttribute('src', imageUrl);
  });

  it('seeds the edit composer with the original image attachment', async () => {
    const imageUrl = 'data:image/png;base64,AAAA';
    renderAssistantChat(
      <AssistantChat
        model="test-vision-model"
        workspace="default"
        enableImageAttachments
        initialMessages={[
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Look at this' },
              { type: 'image', image: imageUrl },
            ],
          },
          {
            role: 'assistant',
            content: [{ type: 'text', text: 'Nice picture' }],
            status: { type: 'complete', reason: 'stop' },
          },
        ]}
      />
    );

    await userEvent.click(screen.getByRole('button', { name: /Edit message/i }));

    const editComposer = screen.getByTestId('assistant-chat-edit-composer');
    expect(within(editComposer).getByRole('textbox', { name: /Edit message/i })).toHaveValue(
      'Look at this'
    );
    expect(within(editComposer).getByRole('img')).toHaveAttribute('src', imageUrl);
    expect(within(editComposer).getByRole('button', { name: /Remove/i })).toBeInTheDocument();
  });

  it('renders composer override content in place of the prompt input', () => {
    renderAssistantChat(
      <AssistantChat
        model="test-model"
        workspace="default"
        composerOverride={<div>Approval required</div>}
      />
    );

    expect(screen.getByText('Approval required')).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: /Task prompt/i })).not.toBeInTheDocument();
  });

  it(
    'edits a user message and re-runs inference with the edited prompt',
    async () => {
      const user = userEvent.setup({ delay: null });
      mocks.createChatCompletion
        .mockResolvedValueOnce(createCompletion('Original response.'))
        .mockResolvedValueOnce(createCompletion('Edited response.'));

      renderAssistantChat(<AssistantChat model="test-model" workspace="default" />);

      await user.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Original prompt');
      await user.click(screen.getByRole('button', { name: /Submit/i }));

      expect(await screen.findByText('Original response.')).toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: /Edit message/i }));
      const editInput = screen.getByRole('textbox', { name: /Edit message/i });
      expect(editInput).toHaveValue('Original prompt');
      expect(editInput.tagName).toBe('TEXTAREA');

      await user.clear(editInput);
      await user.type(editInput, 'Edited prompt');
      await user.click(screen.getByRole('button', { name: /Save edit/i }));

      expect(await screen.findByText('Edited response.')).toBeInTheDocument();
      expect(screen.getByText('Edited prompt')).toBeInTheDocument();
      expect(screen.queryByText('Original prompt')).not.toBeInTheDocument();
      expect(screen.queryByText('Original response.')).not.toBeInTheDocument();

      await waitFor(() => expect(mocks.createChatCompletion).toHaveBeenCalledTimes(2));
      expect(mocks.createChatCompletion).toHaveBeenLastCalledWith(
        expect.objectContaining({
          model: 'test-model',
          workspace: 'default',
          stream: true,
          messages: [{ role: 'user', content: 'Edited prompt' }],
        })
      );
    },
    interactionTimeoutMs
  );

  it(
    'stops a hanging stream when stop is clicked',
    async () => {
      const stream = createHangingStream('0 this is an example response');
      const abortSpy = vi.spyOn(stream.controller, 'abort');
      mocks.createChatCompletion.mockResolvedValueOnce(stream);

      renderAssistantChat(<AssistantChat model="test-model" workspace="default" />);

      await userEvent.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Hang forever');
      await userEvent.click(screen.getByRole('button', { name: /Submit/i }));

      expect(await screen.findByText('0 this is an example response')).toBeInTheDocument();
      const stopButton = screen.getByRole('button', { name: /Stop/i });
      expect(stopButton).toBeEnabled();

      await userEvent.click(stopButton);

      await waitFor(() => expect(abortSpy).toHaveBeenCalledTimes(1));
      await waitFor(() =>
        expect(screen.queryByRole('button', { name: /Stop/i })).not.toBeInTheDocument()
      );
      expect(screen.getByRole('button', { name: /Submit/i })).toBeInTheDocument();
      expect(screen.getByText('0 this is an example response')).toBeInTheDocument();
    },
    interactionTimeoutMs
  );

  it(
    'can hide the running indicator while a stream is pending',
    async () => {
      const stream = createHangingStream('Waiting for user input.');
      mocks.createChatCompletion.mockResolvedValueOnce(stream);

      renderAssistantChat(
        <AssistantChat model="test-model" workspace="default" showRunningIndicator={false} />
      );

      await userEvent.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Need input');
      await userEvent.click(screen.getByRole('button', { name: /Submit/i }));

      expect(await screen.findByText('Waiting for user input.')).toBeInTheDocument();
      expect(screen.queryByTestId('assistant-chat-skeleton')).not.toBeInTheDocument();

      await userEvent.click(screen.getByRole('button', { name: /Stop/i }));
    },
    interactionTimeoutMs
  );

  it(
    'aborts a pending completion request when stop is clicked',
    async () => {
      const user = userEvent.setup({ delay: null });
      let requestSignal: AbortSignal | undefined;
      const abortError = new Error('aborted');
      abortError.name = 'AbortError';

      mocks.createChatCompletion.mockImplementationOnce((request: CompletionRequestWithSignal) => {
        requestSignal = request.signal;

        return new Promise<ChatCompletion>((_resolve, reject) => {
          request.signal?.addEventListener('abort', () => reject(abortError), { once: true });
        });
      });

      renderAssistantChat(<AssistantChat model="test-model" workspace="default" />);

      await user.type(screen.getByRole('textbox', { name: /Task prompt/i }), 'Hang before stream');
      await user.click(screen.getByRole('button', { name: /Submit/i }));

      await waitFor(() => expect(requestSignal).toBeDefined());
      expect(requestSignal?.aborted).toBe(false);

      // Wait for isRunning to be reflected in the DOM before clicking Stop.
      await user.click(await screen.findByRole('button', { name: /Stop/i }));

      await waitFor(() => expect(requestSignal?.aborted).toBe(true));
      await waitFor(() =>
        expect(screen.queryByRole('button', { name: /Stop/i })).not.toBeInTheDocument()
      );
      expect(screen.queryByText('aborted')).not.toBeInTheDocument();
    },
    interactionTimeoutMs
  );
});
