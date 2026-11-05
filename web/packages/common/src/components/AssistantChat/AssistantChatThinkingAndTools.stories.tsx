// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { AssistantChat } from '@nemo/common/src/components/AssistantChat';
import { ThoughtStep } from '@nemo/common/src/components/AssistantChat/chainOfThought/ThoughtStep';
import { ToolCallStep } from '@nemo/common/src/components/AssistantChat/chainOfThought/ToolCallStep';
import { WorkGroupView } from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkGroup';
import { WORK_DURATIONS_METADATA_KEY } from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import { Stack, Text, TooltipProvider } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { http, HttpResponse } from 'msw';

// Review how a reply's thinking and tool calls render: one collapsed group per run of work,
// labelled "Thinking…" while it streams and "Ran N tool calls · 12s" once the answer starts.
// Expand the group to see each step: a Thought, a tool call with its arguments and result, or a
// file edit with the written content.

const STORY_INFERENCE_BASE_PATH = '/storybook-inference/v1';
const STORY_COMPLETIONS_API = `${STORY_INFERENCE_BASE_PATH}/chat/completions`;
const STORY_MODEL = 'nvidia/nemotron-3-super';

const getStoryInferenceBaseURL = (): string =>
  typeof window === 'undefined'
    ? `http://localhost${STORY_INFERENCE_BASE_PATH}`
    : `${window.location.origin}${STORY_INFERENCE_BASE_PATH}`;

const REASONING =
  'The user wants to know if they need an umbrella in Paris tomorrow. I should look up the forecast, then check the hourly rain chance for the afternoon when they will be outside.';

const WEATHER_ARGS = { city: 'Paris', date: '2026-11-05' };
const WEATHER_RESULT = {
  city: 'Paris',
  forecast: 'Light rain',
  high_c: 12,
  low_c: 7,
  precipitation_chance: 0.7,
};
const MEMORY_EDIT = {
  file_path: 'notes/MEMORY.md',
  content: "Don't forget to pick up **milk** from the store",
};

const withDurations = (durationsMs: number[]): ThreadMessageLike['metadata'] => ({
  custom: { [WORK_DURATIONS_METADATA_KEY]: durationsMs },
});

const staticMessages: readonly ThreadMessageLike[] = [
  { role: 'user', content: [{ type: 'text', text: 'Is 9.11 bigger than 9.9?' }] },
  {
    role: 'assistant',
    content: [
      {
        type: 'reasoning',
        text: 'Compare the decimals digit by digit. 9.9 is 9.90, and 90 hundredths is more than 11 hundredths, so 9.9 is larger.',
      },
      { type: 'text', text: 'No. **9.9** is bigger: 9.90 is greater than 9.11.' },
    ],
    status: { type: 'complete', reason: 'stop' },
    metadata: withDurations([3200]),
  },
  { role: 'user', content: [{ type: 'text', text: 'Do I need an umbrella in Paris tomorrow?' }] },
  {
    role: 'assistant',
    content: [
      { type: 'reasoning', text: REASONING },
      {
        type: 'tool-call',
        toolCallId: 'call_weather',
        toolName: 'get_weather',
        args: WEATHER_ARGS,
        argsText: JSON.stringify(WEATHER_ARGS),
        result: WEATHER_RESULT,
      },
      {
        type: 'tool-call',
        toolCallId: 'call_hourly',
        toolName: 'get_hourly_precipitation',
        args: { city: 'Paris', from: '12:00', to: '18:00' },
        argsText: '{"city":"Paris","from":"12:00","to":"18:00"}',
        result: '[{"hour":"12:00","chance":0.6},{"hour":"15:00","chance":0.8}]',
      },
      {
        type: 'text',
        text: 'Yes. Expect light rain with a **70%** chance overall, peaking around 3pm. Take an umbrella.',
      },
    ],
    status: { type: 'complete', reason: 'stop' },
    metadata: withDurations([12_400]),
  },
  { role: 'user', content: [{ type: 'text', text: 'Remind me to buy milk.' }] },
  {
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'call_memory',
        toolName: 'write_file',
        args: MEMORY_EDIT,
        argsText: JSON.stringify(MEMORY_EDIT),
        result: 'ok',
      },
      { type: 'text', text: "Saved. I'll remind you to pick up milk." },
    ],
    status: { type: 'complete', reason: 'stop' },
    metadata: withDurations([1800]),
  },
  { role: 'user', content: [{ type: 'text', text: "What's on my calendar today?" }] },
  {
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'call_calendar',
        toolName: 'list_calendar_events',
        args: { date: 'today' },
        argsText: '{"date":"today"}',
        result: 'PermissionDenied: calendar access was not granted for this agent.',
        isError: true,
      },
      {
        type: 'text',
        text: "I couldn't read your calendar: this agent doesn't have calendar access yet.",
      },
    ],
    status: { type: 'complete', reason: 'stop' },
    metadata: withDurations([900]),
  },
];

type StreamDelta = Record<string, unknown>;

const STREAM_STEPS: readonly StreamDelta[] = [
  ...REASONING.split(/(?<= )/).map((word) => ({ reasoning_content: word })),
  {
    tool_calls: [
      {
        index: 0,
        id: 'call_weather',
        type: 'function',
        function: { name: 'get_weather', arguments: '' },
      },
    ],
  },
  { tool_calls: [{ index: 0, function: { arguments: '{"city":"Paris",' } }] },
  { tool_calls: [{ index: 0, function: { arguments: '"date":"2026-11-05"}' } }] },
  { reasoning_content: 'Rain is likely, so I will recommend an umbrella.' },
  ...'Yes. Expect light rain tomorrow, so take an umbrella.'
    .split(/(?<= )/)
    .map((word) => ({ content: word })),
];

const makeChunk = (delta: StreamDelta, finishReason: string | null = null) => ({
  id: 'story-thinking-completion',
  object: 'chat.completion.chunk',
  created: 1740419165,
  model: STORY_MODEL,
  choices: [{ index: 0, delta, finish_reason: finishReason, logprobs: null }],
});

const makeThinkingStreamHandler = (delayMs = 90) =>
  http.post(STORY_COMPLETIONS_API, () => {
    const encoder = new TextEncoder();
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    const stream = new ReadableStream({
      start(controller) {
        let index = 0;
        const send = (value: unknown) =>
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(value)}\n\n`));
        const next = () => {
          const step = STREAM_STEPS[index];
          index += 1;
          if (!step) {
            send(makeChunk({}, 'stop'));
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
            controller.close();
            return;
          }
          send(makeChunk(step));
          // Pause longer around the tool call so the live label is easy to watch.
          timeoutId = setTimeout(next, 'tool_calls' in step ? delayMs * 8 : delayMs);
        };
        next();
      },
      cancel() {
        if (timeoutId) clearTimeout(timeoutId);
      },
    });
    return new HttpResponse(stream, { headers: { 'Content-Type': 'text/event-stream' } });
  });

const meta = {
  component: AssistantChat,
  title: 'UI/AssistantChat/Thinking and tools',
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="h-[720px] max-w-[760px]">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  args: {
    model: STORY_MODEL,
    baseURL: getStoryInferenceBaseURL(),
    assistantName: 'Agent',
    initialMessages: staticMessages,
  },
  parameters: {
    msw: { handlers: [makeThinkingStreamHandler()] },
  },
} satisfies Meta<typeof AssistantChat>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Thinking only, thinking with tool calls, a file edit, and a tool that returned an error. */
export const StaticThread: Story = {};

/** Send any message to watch reasoning and a tool call stream in, then the answer. */
export const LiveStream: Story = {
  args: {
    initialMessages: [],
    emptyState: {
      slotHeading: 'Send a message',
      slotSubheading: 'The mocked model streams its reasoning and a tool call before answering.',
    },
  },
};

// One call per entry in TOOL_STEP_DEFINITIONS, with names in the styles different agents use.
const KNOWN_TOOL_EXAMPLES: readonly { readonly toolName: string; readonly args: object }[] = [
  { toolName: 'Bash', args: { command: 'pnpm --filter @nemo/common test' } },
  { toolName: 'read_file', args: { file_path: 'web/packages/common/README.md' } },
  { toolName: 'Write', args: { file_path: 'notes/MEMORY.md', content: '- Pack an umbrella' } },
  { toolName: 'Edit', args: { file_path: 'src/app.ts', old_string: 'foo', new_string: 'bar' } },
  { toolName: 'Glob', args: { pattern: '**/*.stories.tsx' } },
  { toolName: 'Grep', args: { pattern: 'ToolCallStep' } },
  { toolName: 'list_directory', args: { path: 'web/packages' } },
  { toolName: 'web_search', args: { query: 'Paris weather tomorrow' } },
  { toolName: 'WebFetch', args: { url: 'https://example.com/forecast' } },
  { toolName: 'TodoWrite', args: { todos: [{}, {}, {}] } },
  { toolName: 'TaskCreate', args: { description: 'Summarize the eval results' } },
  { toolName: 'TaskUpdate', args: { task: 'Summarize the eval results', status: 'done' } },
  { toolName: 'AskUserQuestion', args: { question: 'Which dataset should I use?' } },
  { toolName: 'ToolSearch', args: { query: 'calendar' } },
  { toolName: 'mcp__studio__Skill', args: { skill: 'nemo-status' } },
];

const GalleryLabel = ({ children }: { readonly children: string }) => (
  <Text kind="label/bold/sm" className="text-secondary">
    {children}
  </Text>
);

/** Each group label and step type on its own, expanded where it has detail to show. */
export const ComponentGallery: Story = {
  render: () => (
    <Stack gap="density-2xl" className="max-w-[520px]">
      <Stack gap="density-sm">
        <GalleryLabel>Group labels (collapsed)</GalleryLabel>
        <WorkGroupView hasToolCalls={false} label="Thinking…">
          {null}
        </WorkGroupView>
        <WorkGroupView hasToolCalls={false} label="Thought · 4s">
          {null}
        </WorkGroupView>
        <WorkGroupView hasToolCalls label="Ran 3 tool calls · 12s">
          {null}
        </WorkGroupView>
      </Stack>
      <Stack gap="density-sm">
        <GalleryLabel>Group expanded</GalleryLabel>
        <WorkGroupView defaultOpen hasToolCalls label="Ran 3 tool calls · 12s">
          <ThoughtStep defaultOpen text={REASONING} />
          <ToolCallStep toolName="get_weather" args={WEATHER_ARGS} result={WEATHER_RESULT} />
          <ToolCallStep toolName="search_web" args={{ query: 'Paris weather' }} running />
          <ToolCallStep defaultOpen toolName="write_file" args={MEMORY_EDIT} result="ok" />
        </WorkGroupView>
      </Stack>
      <Stack gap="density-sm">
        <GalleryLabel>Tool call with result</GalleryLabel>
        <ToolCallStep
          defaultOpen
          toolName="get_weather"
          args={WEATHER_ARGS}
          result={WEATHER_RESULT}
        />
      </Stack>
      <Stack gap="density-sm">
        <GalleryLabel>Tool call that failed</GalleryLabel>
        <ToolCallStep
          defaultOpen
          toolName="list_calendar_events"
          args={{ date: 'today' }}
          result="PermissionDenied: calendar access was not granted for this agent."
          isError
        />
      </Stack>
      <Stack gap="density-sm">
        <GalleryLabel>Tool call still running</GalleryLabel>
        <ToolCallStep defaultOpen toolName="search_web" args={{ query: 'Paris weather' }} running />
      </Stack>
      <Stack gap="density-sm">
        <GalleryLabel>Known tools</GalleryLabel>
        {KNOWN_TOOL_EXAMPLES.map(({ toolName, args }) => (
          <ToolCallStep key={toolName} toolName={toolName} args={args} result="ok" />
        ))}
        <ToolCallStep toolName="Bash" args={{ command: 'make test' }} running />
      </Stack>
    </Stack>
  ),
};
