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

| Prop              | Type                                                     | Required | Description                                                                                 |
| ----------------- | -------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------- |
| `model`           | `string`                                                 | Yes      | Model name routed through inference gateway or sent to the explicit `baseURL`.              |
| `workspace`       | `string`                                                 | No       | Workspace used by `useChatCompletion` to build the default inference gateway URL.           |
| `baseURL`         | `string`                                                 | No       | Explicit OpenAI-compatible chat completions base URL.                                       |
| `promptData`      | `PromptData`                                             | No       | Supplies `system_prompt`, `temperature`, and `max_tokens` defaults.                         |
| `tools`           | `ChatCompletionTool[]`                                   | No       | OpenAI-compatible tool definitions forwarded to the completion request.                     |
| `assistantName`   | `string`                                                 | No       | Display name used in the composer placeholder.                                              |
| `placeholder`     | `string`                                                 | No       | Overrides the composer placeholder text.                                                    |
| `disabled`        | `boolean`                                                | No       | Disables assistant-ui input and runtime actions.                                            |
| `className`       | `string`                                                 | No       | Additional class names for the chat container.                                              |
| `initialMessages` | `readonly ThreadMessageLike[]`                           | No       | Initial assistant-ui messages for the thread.                                               |
| `onError`         | `(error: Error) => void`                                 | No       | Called when a completion request fails for a non-cancel reason.                             |
| `describeError`   | `(message: string) => ChatErrorDescription \| undefined` | No       | Explains a failed reply from its raw error message (see [Failed replies](#failed-replies)). |

### Runtime Integration

`AssistantChat` creates an assistant-ui `ExternalStoreRuntime` through `useAssistantChatRuntime`. Consumers pass props to `AssistantChat`; they do not need to instantiate the runtime directly. The runtime converts assistant-ui messages to OpenAI messages, calls `useChatCompletion`, streams deltas into the latest assistant message, and exposes edit, reload, cancel, and reset handlers to `AssistantChatThread`.

### Thinking and tool calls

Streamed reasoning (`reasoning_content` or `reasoning`) and `tool_calls` become reasoning and tool-call parts on the reply. Each run of adjacent reasoning and tool-call parts renders as one collapsed work group (`chainOfThought/`), built on `MessagePrimitive.GroupedParts`:

- the group label reads "Thinking…" while the run is still streaming, then the tool-call count and how long the run took, for example "Ran 3 tool calls · 12s" or "Thought · 4s";
- expanding it shows one row per step: a **Thought** with the reasoning text, a **Called {tool}** row with the arguments and result (or error), or an **Edited {file}** row with the written content for any tool whose arguments carry a file path and content;
- the runtime times each run from its first reasoning or tool delta until answer text arrives and stores the durations in `message.metadata.custom.workDurationsMs`. Seeded messages without durations show the count only.

Pass `toolCallPartComponent` to render tool calls your own way inside the group; the NeMo Assistant does this to unpack its synthetic thinking and grouped-tool parts. Job progress tool calls stay outside the group so their progress card remains visible. See the [Thinking and tools story](./AssistantChatThinkingAndTools.stories.tsx) for every state.

### Failed replies

When a request fails for a non-cancel reason the assistant message is marked `incomplete`/`error` and renders as one inline error banner (successful replies render directly on the canvas without a bubble):

- the banner message is the reply's text, which holds the human-readable explanation;
- a "Show details" row under the banner expands to the **raw** error, with the status code and provider detail, in a code snippet that can be copied.

The banner has no actions of its own: the reply's hover action row stays available, and its "Regenerate response" action reruns the request.

The explanation is the title and hint returned by the `describeError` prop, or the generic line "There was an error generating a response." when the prop is omitted or returns `undefined`. Only the agent chat passes one (`describeChatError`, wired in `ChatPlaygroundContent`), because its wording is specific to agents; every other surface shows the generic line above the raw error.

`describeChatError.ts` holds the agent rules. The platform returns 502 for any agent failure and the model's HTTP status survives only inside the message text, so the rules match phrases the platform writes (`Fabric runtime startup failed`, `Could not connect to agent`, a bodyless `502`) and the envelope the Fabric runtime wraps an openai SDK failure in (`RuntimeError: internal error: APIStatusError: Error code: NNN - ...`). Every rule is anchored at the start of the message, and platform phrases are checked before the upstream status, so a status or phrase that only appears inside quoted detail is ignored and a failure that merely quotes a status code is not blamed on the model provider. Statuses without a rule (400, 409, 422...) fall back to the generic line. Replace the text matching with a switch on a structured error code once the platform returns one.

Related documentation:

- [assistant-ui documentation](https://www.assistant-ui.com/docs)
- [AssistantChat stories](./AssistantChat.stories.tsx)
- [Error states story](./AssistantChatErrorStates.stories.tsx): pick a failure in the Controls panel (or paste custom error text) to see the reply the chat renders for it. Use it to review and reword the messages in `describeChatError.ts`.

## Common Patterns

- **Editing**: saving a user-message edit trims downstream messages and runs inference again from the edited prompt.
- **Regeneration**: reload removes the latest assistant response and re-runs inference with the preceding messages.
- **Stop**: cancel aborts the active request or stream and marks any running assistant message as cancelled.
- **Reset**: reset aborts active work and clears the local thread.
- **Testing**: Storybook uses MSW handlers for streaming, indexed example responses, and a hanging stream that validates Stop behavior.
- **Failures**: a failed reply shows the explanation in a red-barred body, the raw error in the banner, and copies the raw error.

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
- Explained failures for surfaces that pass `describeError`, with the raw error kept in the banner and copy action.

## File Layout

- `index.tsx` — `AssistantChat` entry point; instantiates the runtime and renders the thread.
- `useAssistantChatRuntime.ts` — assistant-ui `ExternalStoreRuntime`, message conversion, and edit/reload/cancel/reset handlers.
- `AssistantChatThread.tsx` — thread viewport layout that wires message renderers and the composer together.
- `AssistantMessage.tsx` / `UserMessage.tsx` — assistant/system and user message renderers (`AssistantMessage` also renders system messages).
- `UserEditComposer.tsx` — the inline composer shown when editing a user message.
- `AssistantChatMessageContent.tsx` — shared message-parts renderer (text, image, tool-call fallback, error banner with the raw error) used by both message renderers.
- `messageActions.tsx` — shared message-action styles, the copy-message button, and the copy-error button shown on failed replies.
- `describeChatError.ts` — rules that explain agent chat failures (`describeChatError`), `humanizeChatError`, and the generic error line. `messageUtils.ts` `getFailedReply` builds the failed reply from them.
- `useMessageError.ts` — hook returning the raw error of a failed message, shared by the red bar, banner, and copy action.
- `AssistantChatErrorStates.stories.tsx` — error-state story driven by Storybook Controls.
- `AssistantComposer.tsx` — the prompt composer, send/stop/reset controls, and image-attachment chips.
- `types.ts` — shared prop and config types. `messageUtils.ts` / `completionUtils.ts` — message and completion helpers.

## Not Yet Supported

- Non-image file attachments (documents, audio).
- Persisted multi-thread history or server-backed thread management.
- Feedback/intake submission parity with the existing `Chat` component.
- Custom tool call result UIs.
