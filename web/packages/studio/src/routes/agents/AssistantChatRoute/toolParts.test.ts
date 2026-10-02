// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadAssistantMessagePart } from '@assistant-ui/react';
import { ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME } from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';
import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  createAssistantThinkingPart,
  getAssistantCompletedMessageParts,
  STUDIO_MESSAGE_SUMMARY_END,
  STUDIO_MESSAGE_SUMMARY_START,
} from '@studio/routes/agents/AssistantChatRoute/toolParts';

describe('Assistant tool parts', () => {
  it('collapses reasoning on a completed message that made no tool call', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      createAssistantThinkingPart('The user wants the capital of France.', 'thinking-1'),
      { type: 'text', text: 'Paris.' },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
        args: { text: 'The user wants the capital of France.' },
      },
      { type: 'text', text: 'Paris.' },
    ]);
  });

  it('folds reasoning into the collapsed thinking block alongside pre-tool text', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      createAssistantThinkingPart('First I need the repo root.', 'thinking-1'),
      { type: 'text', text: 'Checking the repo.' },
      {
        type: 'tool-call',
        toolCallId: 'toolu_bash',
        toolName: 'Bash',
        args: { command: 'pwd' },
        argsText: '{"command":"pwd"}',
      },
      { type: 'text', text: 'Done.' },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
        args: { text: 'First I need the repo root.\n\nChecking the repo.' },
      },
      { type: 'tool-call', toolName: 'Bash' },
      { type: 'text', text: 'Done.' },
    ]);
  });

  it('keeps reasoning and narration in the order they streamed', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      createAssistantThinkingPart('First I need the repo root.', 'thinking-1'),
      { type: 'text', text: 'Checking the repo.' },
      {
        type: 'tool-call',
        toolCallId: 'toolu_bash',
        toolName: 'Bash',
        args: { command: 'pwd' },
        argsText: '{"command":"pwd"}',
      },
      createAssistantThinkingPart('Now I know where to look.', 'thinking-2'),
      { type: 'text', text: 'Found it in stream.ts.' },
      {
        type: 'tool-call',
        toolCallId: 'toolu_read',
        toolName: 'Read',
        args: { file_path: 'stream.ts' },
        argsText: '{"file_path":"stream.ts"}',
      },
      { type: 'text', text: 'Done.' },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
        args: {
          text: [
            'First I need the repo root.',
            'Checking the repo.',
            'Now I know where to look.',
            'Found it in stream.ts.',
          ].join('\n\n'),
        },
      },
      { type: 'tool-call', toolName: 'Bash' },
      { type: 'tool-call', toolName: 'Read' },
      { type: 'text', text: 'Done.' },
    ]);
  });

  it('ignores a whitespace-only reasoning part instead of collapsing the answer', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      createAssistantThinkingPart('   ', 'thinking-1'),
      { type: 'text', text: 'One.\n\nTwo.\n\nThree.' },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toEqual([
      { type: 'text', text: 'One.\n\nTwo.\n\nThree.' },
    ]);
  });

  it('keeps a reasoning-free message without a tool call unchanged', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [{ type: 'text', text: 'Paris.' }];

    expect(getAssistantCompletedMessageParts(parts)).toEqual(parts);
  });

  it('collapses details before a Studio summary block and shows only the summary text', () => {
    const bashPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_bash',
      toolName: 'Bash',
      args: { command: 'pwd' },
      argsText: '{"command":"pwd"}',
    };
    const parts: readonly ThreadAssistantMessagePart[] = [
      { type: 'text', text: 'I inspected the repo first.' },
      bashPart,
      {
        type: 'text',
        text: [
          'I found the prompt builder and updated it.',
          '',
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: unknown',
          'summary: Updated Studio so completed assistant messages collapse to a short summary.',
          'details_label: worked for unknown',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    const completedParts = getAssistantCompletedMessageParts(parts, { elapsedMs: 123_000 });

    expect(completedParts).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: {
          label: 'worked for 2m 3s',
          parts: [
            { type: 'text', text: 'I inspected the repo first.' },
            { type: 'tool-call', toolName: 'Bash', args: { command: 'pwd' } },
            { type: 'text', text: 'I found the prompt builder and updated it.' },
          ],
        },
      },
      {
        type: 'text',
        text: 'Updated Studio so completed assistant messages collapse to a short summary.',
      },
    ]);
  });

  it('preserves non-text parts interleaved across a streamed Studio summary block', () => {
    const interleavedToolPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_interleaved',
      toolName: 'Bash',
      args: { command: 'pwd' },
      argsText: '{"command":"pwd"}',
    };
    const parts: readonly ThreadAssistantMessagePart[] = [
      { type: 'text', text: 'Detailed work that should be collapsed.' },
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: 8s',
          'summary: Finished the streamed work.',
        ].join('\n'),
      },
      interleavedToolPart,
      {
        type: 'text',
        text: ['details_label: worked for 8s', STUDIO_MESSAGE_SUMMARY_END].join('\n'),
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: {
          parts: [
            { type: 'text', text: 'Detailed work that should be collapsed.' },
            { type: 'tool-call', toolCallId: 'toolu_interleaved', toolName: 'Bash' },
          ],
        },
      },
      { type: 'text', text: 'Finished the streamed work.' },
    ]);
  });

  it('hides a raw Studio summary marker when the streamed block is truncated', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      { type: 'text', text: 'I finished the available repository analysis.' },
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: 8s',
          'summary: This partial field must not expose its sentinel.',
        ].join('\n'),
      },
    ];

    const completedParts = getAssistantCompletedMessageParts(parts, { elapsedMs: 8_000 });

    expect(completedParts).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: {
          label: 'worked for 8s',
          parts: [{ type: 'text', text: 'I finished the available repository analysis.' }],
        },
      },
      { type: 'text', text: 'I finished the available repository analysis.' },
    ]);
    expect(JSON.stringify(completedParts)).not.toContain(STUDIO_MESSAGE_SUMMARY_START);
  });

  it('removes the Studio summary markers when there are no details to collapse', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: 4s',
          'summary: Ready for the next step.',
          'details_label: worked for 4s',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toEqual([
      { type: 'text', text: 'Ready for the next step.' },
    ]);
  });

  it('uses a neutral details label when the work time is unknown', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      { type: 'text', text: 'Detailed work that should be collapsed.' },
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: unknown',
          'summary: Ready for the next step.',
          'details_label: worked for unknown',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: { label: 'Work details' },
      },
      { type: 'text', text: 'Ready for the next step.' },
    ]);
  });

  it('keeps an unanswered trailing question visible when the model omits it from the summary', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      {
        type: 'text',
        text: [
          'I found three deployed agents.',
          '',
          'Which agent do you want to optimize?',
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: 20s',
          'summary: I investigated the available optimization targets.',
          'details_label: worked for 20s',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: {
          parts: [{ type: 'text', text: 'I found three deployed agents.' }],
        },
      },
      {
        type: 'text',
        text: [
          'I investigated the available optimization targets.',
          '',
          'Which agent do you want to optimize?',
        ].join('\n'),
      },
    ]);
  });

  it('accepts an inline Studio summary block from the model', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      { type: 'text', text: 'Detailed work that should be collapsed.' },
      {
        type: 'text',
        text: `${STUDIO_MESSAGE_SUMMARY_START} worked_for: ~3 minutes summary: Analyzed calculator-agent and generated 3 optimization suggestions. Snapshot and suggestions persisted. details_label: worked for ~3 minutes ${STUDIO_MESSAGE_SUMMARY_END}`,
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: {
          label: 'worked for ~3 minutes',
          parts: [{ type: 'text', text: 'Detailed work that should be collapsed.' }],
        },
      },
      {
        type: 'text',
        text: 'Analyzed calculator-agent and generated 3 optimization suggestions. Snapshot and suggestions persisted.',
      },
    ]);
  });

  it('preserves markdown formatting in a Studio summary block', () => {
    const markdownSummary = [
      '## Completed',
      '',
      '1. Preserved **emphasis**',
      '2. Preserved `inline code`',
      '',
      '```ts',
      'const formatted = true;',
      '```',
    ].join('\n');
    const parts: readonly ThreadAssistantMessagePart[] = [
      { type: 'text', text: 'Detailed work that should be collapsed.' },
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: 12s',
          'summary:',
          markdownSummary,
          'details_label: worked for 12s',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
      },
      { type: 'text', text: markdownSummary },
    ]);
  });

  it('keeps tables and prominent custom components visible with the final summary', () => {
    const bashPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_bash',
      toolName: 'Bash',
      args: { command: 'inspect results' },
      argsText: '{"command":"inspect results"}',
    };
    const jobPart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_job',
      toolName: ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
      args: { job_name: 'spam-detector-eval' },
      argsText: '{"job_name":"spam-detector-eval"}',
    };
    const fileChangePart: ThreadAssistantMessagePart = {
      type: 'tool-call',
      toolCallId: 'toolu_write',
      toolName: 'Write',
      args: { file_path: 'agents/spam-detector.yml', content: 'workflow: spam-detector' },
      argsText: '{"file_path":"agents/spam-detector.yml","content":"workflow: spam-detector"}',
    };
    const markdownTable = [
      '| Metric | Result |',
      '| --- | ---: |',
      '| Precision | 0.98 |',
      '| Recall | 0.96 |',
    ].join('\n');
    const parts: readonly ThreadAssistantMessagePart[] = [
      {
        type: 'text',
        text: `I inspected the evaluation output.\n\n${markdownTable}\n\nThe detailed logs are available.`,
      },
      bashPart,
      jobPart,
      fileChangePart,
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'title: Evaluate Spam Detector',
          'worked_for: 20s',
          'summary: The spam detector evaluation completed successfully.',
          'details_label: worked for 20s',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        args: {
          parts: [
            {
              type: 'text',
              text: 'I inspected the evaluation output.\n\nThe detailed logs are available.',
            },
            { type: 'tool-call', toolName: 'Bash' },
          ],
        },
      },
      { type: 'text', text: markdownTable },
      { type: 'tool-call', toolName: ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME },
      { type: 'tool-call', toolName: 'Write' },
      { type: 'text', text: 'The spam detector evaluation completed successfully.' },
    ]);
  });

  it('includes links from collapsed details at the bottom of the summary', () => {
    const parts: readonly ThreadAssistantMessagePart[] = [
      {
        type: 'text',
        text: [
          'I generated the report.',
          '',
          '[Agent optimizations](/workspaces/default/agents/suggestions)',
        ].join('\n'),
      },
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: 12s',
          'summary: Generated three optimization suggestions.',
          'details_label: worked for 12s',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    expect(getAssistantCompletedMessageParts(parts)).toMatchObject([
      {
        type: 'tool-call',
        toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
      },
      {
        type: 'text',
        text: [
          'Generated three optimization suggestions.',
          '',
          '[Agent optimizations](/workspaces/default/agents/suggestions)',
        ].join('\n'),
      },
    ]);
  });

  it('does not duplicate a detail link already included in the summary', () => {
    const link = '[Agent optimizations](/workspaces/default/agents/suggestions)';
    const parts: readonly ThreadAssistantMessagePart[] = [
      { type: 'text', text: `I generated the report.\n\n${link}` },
      {
        type: 'text',
        text: [
          STUDIO_MESSAGE_SUMMARY_START,
          'worked_for: 12s',
          `summary: Generated three optimization suggestions.\n\n${link}`,
          'details_label: worked for 12s',
          STUDIO_MESSAGE_SUMMARY_END,
        ].join('\n'),
      },
    ];

    const completedParts = getAssistantCompletedMessageParts(parts);
    const summaryPart = completedParts.at(-1);

    expect(summaryPart).toEqual({
      type: 'text',
      text: `Generated three optimization suggestions.\n\n${link}`,
    });
  });
});
