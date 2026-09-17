// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME } from '@studio/routes/agents/AssistantChatRoute/jobProgressConsts';
import {
  ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
  ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
  ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
  STUDIO_MESSAGE_SUMMARY_END,
  STUDIO_MESSAGE_SUMMARY_START,
} from '@studio/routes/agents/AssistantChatRoute/toolParts';
import type { AssistantSessionHistory } from '@studio/routes/agents/AssistantChatRoute/types';
import {
  getAssistantChatRouteForSession,
  getAssistantHistoryMessages,
  getSelectedAssistantSessionId,
} from '@studio/routes/agents/AssistantChatRoute/util';
import { getAssistantChatRoute } from '@studio/routes/utils';

describe('Assistant utilities', () => {
  it('builds and reads selected session URLs', () => {
    const workspace = 'default';
    const sessionId = '2dc6e5a6-acd7-43bf-b128-c9fd5cf6eb9a';

    expect(getAssistantChatRouteForSession(workspace, sessionId)).toBe(
      `${getAssistantChatRoute(workspace)}?session=${sessionId}`
    );
    expect(getSelectedAssistantSessionId(`?session=${sessionId}`)).toBe(sessionId);
    expect(getSelectedAssistantSessionId('?session=')).toBeUndefined();
  });

  it('restores a stored thinking part as a collapsed thinking block', () => {
    const history: AssistantSessionHistory = {
      session_id: 'session-thinking',
      chat_artifacts: { selections: [], files: [], links: [], jobs: [], tools: [] },
      items: [
        { kind: 'user', text: 'what is the capital of France?' },
        {
          kind: 'assistant',
          parts: [
            { type: 'thinking', thinking: 'The user wants a capital city.' },
            { type: 'text', text: 'Paris.' },
          ],
        },
      ],
    };

    expect(getAssistantHistoryMessages(history)[1]).toMatchObject({
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
          args: { text: 'The user wants a capital city.' },
        },
        { type: 'text', text: 'Paris.' },
      ],
    });
  });

  it('converts stored transcript items to assistant-ui messages', () => {
    const history: AssistantSessionHistory = {
      session_id: '2dc6e5a6-acd7-43bf-b128-c9fd5cf6eb9a',
      chat_artifacts: { selections: [], files: [], links: [], jobs: [], tools: [] },
      items: [
        { kind: 'user', text: 'check the repo' },
        {
          kind: 'assistant',
          parts: [
            { type: 'text', text: 'I found the route.' },
            { type: 'tool_use', name: 'AskUserQuestion', input: { question: 'Continue?' } },
            { type: 'tool_use', name: 'Bash', input: { command: 'pwd' } },
            { type: 'tool_use', name: 'Grep', input: { pattern: 'TODO' } },
            { type: 'tool_use', name: 'Read', input: { file_path: 'README.md' } },
            { type: 'text', text: 'I updated the route.\n\nTests passed.' },
          ],
        },
        {
          kind: 'assistant',
          parts: [
            { type: 'tool_use', name: 'TaskUpdate', input: { status: 'done' } },
            { type: 'tool_use', name: 'Grep', input: { pattern: 'TODO' } },
            { type: 'tool_use', name: 'AskUserQuestion', input: { question: 'Continue?' } },
            { type: 'tool_use', name: 'FindFiles', input: { query: 'TODO' } },
            { type: 'tool_use', name: 'TaskCreate', input: { task: 'check' } },
            { type: 'tool_use', name: 'ToolSearch', input: { query: 'read' } },
          ],
        },
        {
          kind: 'assistant',
          parts: [
            { type: 'text', text: 'I found another route.' },
            { type: 'tool_use', name: 'AskUserQuestion', input: { question: 'Continue?' } },
            { type: 'tool_use', name: 'Bash', input: { command: 'pwd' } },
            { type: 'tool_use', name: 'Grep', input: { pattern: 'TODO' } },
            { type: 'tool_use', name: 'Read', input: { file_path: 'package.json' } },
          ],
        },
      ],
    };

    const messages = getAssistantHistoryMessages(history);

    expect(messages).toHaveLength(4);
    expect(messages[0]).toEqual({
      id: '2dc6e5a6-acd7-43bf-b128-c9fd5cf6eb9a-0',
      role: 'user',
      content: [{ type: 'text', text: 'check the repo' }],
    });
    expect(messages[1]).toMatchObject({
      id: '2dc6e5a6-acd7-43bf-b128-c9fd5cf6eb9a-1',
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
          args: {
            text: 'I found the route.',
          },
        },
        {
          type: 'tool-call',
          toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
          args: {
            actions: [
              { args: { question: 'Continue?' }, toolName: 'AskUserQuestion' },
              { args: { command: 'pwd' }, toolName: 'Bash' },
              { args: { pattern: 'TODO' }, toolName: 'Grep' },
              { args: { file_path: 'README.md' }, toolName: 'Read' },
            ],
          },
        },
        { type: 'text', text: 'I updated the route.\n\nTests passed.' },
      ],
      status: { type: 'complete', reason: 'stop' },
    });
    expect(messages[2]).toMatchObject({
      id: '2dc6e5a6-acd7-43bf-b128-c9fd5cf6eb9a-2',
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
          args: {
            actions: [
              { args: { status: 'done' }, toolName: 'TaskUpdate' },
              { args: { pattern: 'TODO' }, toolName: 'Grep' },
              { args: { question: 'Continue?' }, toolName: 'AskUserQuestion' },
              { args: { query: 'TODO' }, toolName: 'FindFiles' },
              { args: { task: 'check' }, toolName: 'TaskCreate' },
              { args: { query: 'read' }, toolName: 'ToolSearch' },
            ],
          },
        },
      ],
      status: { type: 'complete', reason: 'stop' },
    });
    expect(messages[3]).toMatchObject({
      id: '2dc6e5a6-acd7-43bf-b128-c9fd5cf6eb9a-3',
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_COLLAPSED_THINKING_TOOL_NAME,
          args: {
            text: 'I found another route.',
          },
        },
        {
          type: 'tool-call',
          toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
          args: {
            actions: [
              { args: { question: 'Continue?' }, toolName: 'AskUserQuestion' },
              { args: { command: 'pwd' }, toolName: 'Bash' },
              { args: { pattern: 'TODO' }, toolName: 'Grep' },
              { args: { file_path: 'package.json' }, toolName: 'Read' },
            ],
          },
        },
      ],
      status: { type: 'complete', reason: 'stop' },
    });
  });

  it('combines consecutive tool-only assistant transcript items', () => {
    const history: AssistantSessionHistory = {
      session_id: 'session-1',
      chat_artifacts: { selections: [], files: [], links: [], jobs: [], tools: [] },
      items: [
        { kind: 'user', text: 'map the repo' },
        {
          kind: 'assistant',
          parts: [{ type: 'tool_use', name: 'Bash', input: { description: 'list files' } }],
        },
        {
          kind: 'assistant',
          parts: [{ type: 'tool_use', name: 'Glob', input: { pattern: '*' } }],
        },
        {
          kind: 'assistant',
          parts: [{ type: 'text', text: 'Glob hit node_modules.' }],
        },
        {
          kind: 'assistant',
          parts: [{ type: 'tool_use', name: 'Glob', input: { pattern: 'packages/*' } }],
        },
        {
          kind: 'assistant',
          parts: [{ type: 'tool_use', name: 'Glob', input: { pattern: 'services/*' } }],
        },
      ],
    };

    const messages = getAssistantHistoryMessages(history);

    expect(messages).toHaveLength(4);
    expect(messages[1]).toMatchObject({
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
          args: {
            actions: [
              { args: { description: 'list files' }, toolName: 'Bash' },
              { args: { pattern: '*' }, toolName: 'Glob' },
            ],
          },
        },
      ],
    });
    expect(messages[2]).toMatchObject({
      role: 'assistant',
      content: [{ type: 'text', text: 'Glob hit node_modules.' }],
    });
    expect(messages[3]).toMatchObject({
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
          args: {
            actions: [
              { args: { pattern: 'packages/*' }, toolName: 'Glob' },
              { args: { pattern: 'services/*' }, toolName: 'Glob' },
            ],
          },
        },
      ],
    });
  });

  it('collapses all assistant history before a Studio summary block on refresh', () => {
    const history: AssistantSessionHistory = {
      session_id: 'session-1',
      chat_artifacts: { selections: [], files: [], links: [], jobs: [], tools: [] },
      items: [
        { kind: 'user', text: 'optimize calculator-agent' },
        {
          kind: 'assistant',
          parts: [
            {
              type: 'text',
              text: "I'll start by invoking the nemo-agents-optimize skill.",
            },
            { type: 'tool_use', name: 'ToolSearch', input: { query: 'select_agent' } },
            { type: 'tool_use', name: 'mcp__nemo_studio__select_agent', input: {} },
          ],
        },
        {
          kind: 'assistant',
          parts: [
            {
              type: 'text',
              text: "The user selected calculator-agent. Now I'll run the workflow.",
            },
            { type: 'tool_use', name: 'Skill', input: { name: 'nemo-agents-optimize' } },
          ],
        },
        {
          kind: 'assistant',
          parts: [
            { type: 'text', text: 'Starting parallel data fetches.' },
            { type: 'tool_use', name: 'Bash', input: { description: 'List deployed agents' } },
            { type: 'tool_use', name: 'Bash', input: { description: 'Read model catalog' } },
          ],
        },
        {
          kind: 'assistant',
          parts: [
            { type: 'text', text: 'Both files uploaded. Let me get the Studio link.' },
            { type: 'tool_use', name: 'mcp__nemo_studio__studio_link', input: {} },
            {
              type: 'text',
              text: `${STUDIO_MESSAGE_SUMMARY_START} worked_for: ~3 minutes summary: Analyzed calculator-agent and generated 3 optimization suggestions. Snapshot and suggestions persisted. details_label: worked for ~3 minutes ${STUDIO_MESSAGE_SUMMARY_END}`,
            },
          ],
        },
      ],
    };

    const messages = getAssistantHistoryMessages(history);

    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({
      id: 'session-1-0',
      role: 'user',
      content: [{ type: 'text', text: 'optimize calculator-agent' }],
    });
    expect(messages[1]).toMatchObject({
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
          args: {
            label: 'worked for ~3 minutes',
            parts: [
              {
                type: 'text',
                text: "I'll start by invoking the nemo-agents-optimize skill.",
              },
              {
                type: 'tool-call',
                toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
              },
              {
                type: 'text',
                text: "The user selected calculator-agent. Now I'll run the workflow.",
              },
              {
                type: 'tool-call',
                toolName: 'Skill',
              },
              {
                type: 'text',
                text: 'Starting parallel data fetches.',
              },
              {
                type: 'tool-call',
                toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
              },
              {
                type: 'text',
                text: 'Both files uploaded. Let me get the Studio link.',
              },
              {
                type: 'tool-call',
                toolName: 'mcp__nemo_studio__studio_link',
              },
            ],
          },
        },
        {
          type: 'text',
          text: 'Analyzed calculator-agent and generated 3 optimization suggestions. Snapshot and suggestions persisted.',
        },
      ],
    });
  });

  it('preserves interactive user answers as message boundaries on refresh', () => {
    const history: AssistantSessionHistory = {
      session_id: 'session-1',
      chat_artifacts: { selections: [], files: [], links: [], jobs: [], tools: [] },
      items: [
        { kind: 'user', text: 'optimize an agent' },
        {
          kind: 'assistant',
          parts: [
            { type: 'text', text: 'Which agent should I optimize?' },
            { type: 'tool_use', name: 'mcp__nemo_studio__select_agent', input: {} },
          ],
        },
        { kind: 'user', text: 'Selected agent: calculator-agent' },
        {
          kind: 'assistant',
          parts: [
            { type: 'text', text: 'I analyzed calculator-agent.' },
            {
              type: 'text',
              text: `${STUDIO_MESSAGE_SUMMARY_START} summary: Generated optimization suggestions. details_label: worked briefly ${STUDIO_MESSAGE_SUMMARY_END}`,
            },
          ],
        },
      ],
    };

    const messages = getAssistantHistoryMessages(history);

    expect(messages).toHaveLength(4);
    expect(messages[1]).toMatchObject({ role: 'assistant' });
    expect(messages[2]).toEqual({
      id: 'session-1-2',
      role: 'user',
      content: [{ type: 'text', text: 'Selected agent: calculator-agent' }],
    });
    expect(messages[3]).toMatchObject({
      role: 'assistant',
      content: [
        {
          type: 'tool-call',
          toolName: ASSISTANT_COLLAPSED_STUDIO_DETAILS_TOOL_NAME,
        },
        { type: 'text', text: 'Generated optimization suggestions.' },
      ],
    });
  });

  it('keeps stored job progress tool calls visible in history', () => {
    const history: AssistantSessionHistory = {
      session_id: 'session-1',
      chat_artifacts: { selections: [], files: [], links: [], jobs: [], tools: [] },
      items: [
        { kind: 'user', text: 'evaluate my agent' },
        {
          kind: 'assistant',
          parts: [
            { type: 'tool_use', name: 'Bash', input: { command: 'pwd' } },
            {
              type: 'tool_use',
              id: ' toolu_job ',
              name: ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
              input: { job_name: 'studio-job-1' },
            },
            { type: 'tool_use', name: 'Read', input: { file_path: 'README.md' } },
          ],
        },
      ],
    };

    const messages = getAssistantHistoryMessages(history);

    expect(messages[1]).toMatchObject({
      role: 'assistant',
      content: [
        { type: 'tool-call', toolName: 'Bash' },
        {
          type: 'tool-call',
          toolCallId: 'toolu_job',
          toolName: ASSISTANT_JOB_PROGRESS_MCP_TOOL_NAME,
          args: { job_name: 'studio-job-1' },
        },
        { type: 'tool-call', toolName: 'Read' },
      ],
    });
  });
});
