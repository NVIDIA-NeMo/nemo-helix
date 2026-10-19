<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AssistantChat

`AssistantChat` is an assistant-ui based chat surface for Studio consumers. It owns an assistant-ui `ExternalStoreRuntime` and calls the existing `useChatCompletion` hook, so it can use the same inference gateway routing as the current chat components without depending on the legacy `ChatProvider` state.

## Reference

### Basic Usage

```tsx
import { AssistantChat } from '@nemo/common/src/components/AssistantChat';

export const ChatPanel = () => (
  <AssistantChat
    model="nvidia/nemotron-3.5-lightning-30b-a3b"
    workspace="default"
    assistantName="Inference Gateway"
  />
);
```

Use `baseURL` instead of `workspace` when the caller already has an OpenAI-compatible chat completions endpoint:

```tsx
<AssistantChat
  model="nvidia/nemotron-3.5-lightning-30b-a3b"
  baseURL="https://example.test/v1"
  promptData={{
    system_prompt: 'Answer concisely.',
    inference_params: { temperature: 0.2, max_tokens: 512 },
  }}
/>
```

### Props

| Prop              | Type                           | Required | Description                                                                       |
| ----------------- | ------------------------------ | -------- | --------------------------------------------------------------------------------- |
| `model`           | `string`                       | Yes      | Model name routed through inference gateway or sent to the explicit `baseURL`.    |
| `workspace`       | `string`                       | No       | Workspace used by `useChatCompletion` to build the default inference gateway URL. |
| `baseURL`         | `string`                       | No       | Explicit OpenAI-compatible chat completions base URL.                             |
| `promptData`      | `PromptData`                   | No       | Supplies `system_prompt`, `temperature`, and `max_tokens` defaults.               |
| `tools`           | `ChatCompletionTool[]`         | No       | OpenAI-compatible tool definitions forwarded to the completion request.           |
| `assistantName`   | `string`                       | No       | Display name used in the composer placeholder.                                    |
| `placeholder`     | `string`                       | No       | Overrides the composer placeholder text.                                          |
| `disabled`        | `boolean`                      | No       | Disables assistant-ui input and runtime actions.                                  |
| `className`       | `string`                       | No       | Additional class names for the chat container.                                    |
| `initialMessages` | `readonly ThreadMessageLike[]` | No       | Initial assistant-ui messages for the thread.                                     |
| `onError`         | `(error: Error) => void`       | No       | Called when a completion request fails for a non-cancel reason.                   |

### Runtime Integration

`AssistantChat` creates an assistant-ui `ExternalStoreRuntime` through `useAssistantChatRuntime`. Consumers pass props to `AssistantChat`; they do not need to instantiate the runtime directly. The runtime converts assistant-ui messages to OpenAI messages, calls `useChatCompletion`, streams deltas into the latest assistant message, and exposes edit, reload, cancel, and reset handlers to `AssistantChatThread`.

Related documentation:

- [assistant-ui documentation](https://www.assistant-ui.com/docs)
- [AssistantChat stories](./AssistantChat.stories.tsx)

## Common Patterns

- **Editing**: saving a user-message edit trims downstream messages and runs inference again from the edited prompt.
- **Regeneration**: reload removes the latest assistant response and re-runs inference with the preceding messages.
- **Stop**: cancel aborts the active request or stream and marks any running assistant message as cancelled.
- **Reset**: reset aborts active work and clears the local thread.
- **Testing**: Storybook uses MSW handlers for streaming, indexed example responses, and a hanging stream that validates Stop behavior.

## Supported

- Text user prompts with Enter-to-send and Shift+Enter newlines.
- Image attachments (pick or paste) forwarded as OpenAI multimodal content, enabled only when `enableImageAttachments` is true and the selected model supports image attachments.
- Streaming assistant responses through `useChatCompletion` with `model`, `workspace`, or explicit `baseURL`.
- Optional `PromptData` support for system prompt, temperature, and max tokens.
- Optional OpenAI-compatible tool definitions on the completion request.
- User-message editing that re-runs inference from the edited prompt and drops stale downstream messages.
- Stop/cancel for an in-flight stream, regenerate for assistant responses, and reset for clearing the current thread.
- Basic assistant-ui runtime hooks for future copy, branch, and thread-list features.
- Storybook MSW mocks for normal streaming, indexed example responses, and a hanging stream used to validate Stop behavior.

## File Layout

- `index.tsx` — `AssistantChat` entry point; instantiates the runtime and renders the thread.
- `useAssistantChatRuntime.ts` — assistant-ui `ExternalStoreRuntime`, message conversion, and edit/reload/cancel/reset handlers.
- `AssistantChatThread.tsx` — thread viewport layout that wires message renderers and the composer together.
- `AssistantMessage.tsx` / `UserMessage.tsx` — assistant/system and user message renderers (`AssistantMessage` also renders system messages).
- `UserEditComposer.tsx` — the inline composer shown when editing a user message.
- `AssistantChatMessageContent.tsx` — shared message-parts renderer (text, image, tool-call fallback, error banner) used by both message renderers.
- `messageActions.tsx` — shared message-action styles and the copy-message button.
- `AssistantComposer.tsx` — the prompt composer, send/stop/reset controls, and image-attachment chips.
- `types.ts` — shared prop and config types. `messageUtils.ts` / `completionUtils.ts` — message and completion helpers.

## Not Yet Supported

- Non-image file attachments (documents, audio).
- Persisted multi-thread history or server-backed thread management.
- Feedback/intake submission parity with the existing `Chat` component.
- Custom tool call result UIs.
