// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import { ASSISTANT_SUBTLE_TOOL_GROUP_NAME } from '@nemo/common/src/components/AssistantChat/toolParts';
import {
  didLastReplySucceed,
  getFelixActivity,
  getFelixActivityForTool,
} from '@studio/components/FelixAssistant/felixActivity';

type Part = Exclude<ThreadMessageLike['content'], string>[number];

const toolCall = (
  toolName: string,
  args: Record<string, string | number | boolean> = {},
  extra: { result?: string } = {}
): Part => ({
  type: 'tool-call',
  toolCallId: `call-${toolName}`,
  toolName,
  args,
  argsText: JSON.stringify(args),
  ...extra,
});

const text = (value: string): Part => ({ type: 'text', text: value });
const reasoning = (value: string): Part => ({ type: 'reasoning', text: value });

const assistant = (...parts: Part[]): ThreadMessageLike => ({ role: 'assistant', content: parts });
const user = (value: string): ThreadMessageLike => ({ role: 'user', content: value });

describe('getFelixActivityForTool', () => {
  it.each(['create', 'update', 'delete', 'deploy', 'submit', 'cancel', 'upload'])(
    'treats nemo_api %s as building',
    (action) => {
      expect(getFelixActivityForTool('nemo_api', { action })).toBe('building');
    }
  );

  it.each(['list', 'get', 'retrieve', 'search', 'check', 'get_status', 'get_logs', 'read'])(
    'treats read-only nemo_api %s as docs',
    (action) => {
      expect(getFelixActivityForTool('nemo_api', { action })).toBe('docs');
    }
  );

  it('ignores the case and padding of a nemo_api action', () => {
    expect(getFelixActivityForTool('nemo_api', { action: '  LIST ' })).toBe('docs');
    expect(getFelixActivityForTool('nemo_api', { action: ' Create' })).toBe('building');
  });

  it('gives no pose to a nemo_api call with no action', () => {
    expect(getFelixActivityForTool('nemo_api', {})).toBeNull();
    expect(getFelixActivityForTool('nemo_api', { action: '' })).toBeNull();
    expect(getFelixActivityForTool('nemo_api')).toBeNull();
  });

  it('understands MCP-prefixed names', () => {
    expect(getFelixActivityForTool('mcp__nemo_studio__nemo_api', { action: 'create' })).toBe(
      'building'
    );
    expect(getFelixActivityForTool('mcp__nemo_studio__nemo_api', { action: 'list' })).toBe('docs');
    expect(getFelixActivityForTool('mcp__nemo_studio__deploy_guardrail')).toBe('building');
    expect(getFelixActivityForTool('mcp__nemo_studio__check_status')).toBe('docs');
  });

  it.each(['Write', 'Edit', 'MultiEdit', 'Bash', 'NotebookEdit', 'deploy_guardrail'])(
    'treats %s as building',
    (toolName) => {
      expect(getFelixActivityForTool(toolName, {})).toBe('building');
    }
  );

  it.each(['write_file', 'edit_file', 'execute'])(
    'treats the Deep Agents tool %s as building',
    (toolName) => {
      expect(getFelixActivityForTool(toolName, {})).toBe('building');
    }
  );

  it.each([
    'Read',
    'Grep',
    'Glob',
    'FindFiles',
    'ToolSearch',
    'WebFetch',
    'WebSearch',
    'Skill',
    'check_status',
  ])('treats %s as docs', (toolName) => {
    expect(getFelixActivityForTool(toolName, {})).toBe('docs');
  });

  it.each(['read_file', 'ls', 'glob', 'grep'])(
    'treats the Deep Agents tool %s as docs',
    (toolName) => {
      expect(getFelixActivityForTool(toolName, {})).toBe('docs');
    }
  );

  it.each([
    'studio_link',
    'ask_user_question',
    'AskUserQuestion',
    'select_model',
    'select_agent',
    'write_todos',
    'TodoWrite',
    'task',
    'job_progress',
    'SomethingNew',
  ])('gives no pose to %s', (toolName) => {
    expect(getFelixActivityForTool(toolName, {})).toBeNull();
  });
});

describe('getFelixActivity', () => {
  it('is null for an empty thread', () => {
    expect(getFelixActivity([])).toBeNull();
  });

  it('is null while the last message is the user waiting for a reply', () => {
    expect(getFelixActivity([assistant(toolCall('Write')), user('thanks')])).toBeNull();
  });

  it('is null for a plain-string assistant message', () => {
    expect(getFelixActivity([{ role: 'assistant', content: 'All done.' }])).toBeNull();
  });

  it('follows a tool call that is the last thing in the reply', () => {
    expect(
      getFelixActivity([user('hi'), assistant(toolCall('nemo_api', { action: 'create' }))])
    ).toBe('building');
    expect(getFelixActivity([user('hi'), assistant(toolCall('Read'))])).toBe('docs');
  });

  it('still counts when text or reasoning came before the tool call', () => {
    expect(
      getFelixActivity([assistant(text('Let me check.'), reasoning('hm'), toolCall('Grep'))])
    ).toBe('docs');
  });

  it('ignores blank text after the tool call (streaming placeholder)', () => {
    expect(getFelixActivity([assistant(toolCall('Edit'), text('  \n'))])).toBe('building');
  });

  it('ends once the model moves on to writing text', () => {
    expect(getFelixActivity([assistant(toolCall('Edit'), text('Done, I changed it.'))])).toBeNull();
  });

  it('ends once the model moves on to reasoning', () => {
    expect(getFelixActivity([assistant(toolCall('Edit'), reasoning('Now what...'))])).toBeNull();
  });

  it('ends once the tool call has a result', () => {
    expect(getFelixActivity([assistant(toolCall('Edit', {}, { result: 'ok' }))])).toBeNull();
  });

  it('only looks at the latest tool call', () => {
    expect(
      getFelixActivity([assistant(toolCall('Write', {}, { result: 'ok' }), toolCall('Read'))])
    ).toBe('docs');
  });

  it('only looks at the latest assistant reply', () => {
    expect(
      getFelixActivity([
        assistant(toolCall('Write')),
        user('and now?'),
        assistant(text('Thinking about it')),
      ])
    ).toBeNull();
  });

  it('uses the last call inside a folded group of tool calls', () => {
    const group: Part = {
      type: 'tool-call',
      toolCallId: 'group-1',
      toolName: ASSISTANT_SUBTLE_TOOL_GROUP_NAME,
      args: {
        actions: [
          { toolCallId: 'a', toolName: 'Read', args: {} },
          { toolCallId: 'b', toolName: 'nemo_api', args: { action: 'delete' } },
        ],
      },
      argsText: '',
    };

    expect(getFelixActivity([assistant(group)])).toBe('building');
  });

  it('gives no pose to a tool Felix has no pose for', () => {
    expect(getFelixActivity([assistant(toolCall('studio_link'))])).toBeNull();
  });
});

describe('didLastReplySucceed', () => {
  const replyWith = (status: ThreadMessageLike['status']): ThreadMessageLike => ({
    role: 'assistant',
    content: [text('All done.')],
    status,
  });

  it('is true for a reply that completed', () => {
    expect(didLastReplySucceed([user('hi'), replyWith({ type: 'complete', reason: 'stop' })])).toBe(
      true
    );
  });

  it('is false for a reply that is still running', () => {
    expect(didLastReplySucceed([replyWith({ type: 'running' })])).toBe(false);
  });

  it('is false for a reply that failed', () => {
    expect(
      didLastReplySucceed([replyWith({ type: 'incomplete', reason: 'error', error: 'boom' })])
    ).toBe(false);
  });

  it('is false for a reply the user stopped', () => {
    expect(didLastReplySucceed([replyWith({ type: 'incomplete', reason: 'cancelled' })])).toBe(
      false
    );
  });

  it('is false when the reply carries no status', () => {
    expect(didLastReplySucceed([replyWith(undefined)])).toBe(false);
  });

  it('is false when the user is the last to speak, or there are no messages', () => {
    expect(
      didLastReplySucceed([replyWith({ type: 'complete', reason: 'stop' }), user('and then?')])
    ).toBe(false);
    expect(didLastReplySucceed([])).toBe(false);
  });
});
