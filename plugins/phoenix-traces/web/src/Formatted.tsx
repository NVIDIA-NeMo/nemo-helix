// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Stack, Text } from '@nvidia/foundations-react-core';
import { humanize, readMessages, readStructured, readToolCalls, type ReviewGroup } from './format';
import type { Span } from './traces';

// Inline styles throughout: Studio's Tailwind only generates classes Studio itself uses.
const MONO = { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 12 } as const;
const PROSE = { whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', lineHeight: 1.55 } as const;

const ATTENTION_COLOR: Record<string, 'gray' | 'blue' | 'yellow' | 'red'> = {
  cursory: 'gray',
  standard: 'blue',
  medium: 'blue',
  careful: 'yellow',
  deep: 'red',
};

const DIFF_LINE_STYLE = (line: string) => {
  if (line.startsWith('@@')) return { background: 'rgba(59,130,246,0.15)', color: '#60a5fa' };
  if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('diff ') || line.startsWith('index '))
    return { opacity: 0.6 };
  if (line.startsWith('+')) return { background: 'rgba(34,197,94,0.18)' };
  if (line.startsWith('-')) return { background: 'rgba(239,68,68,0.18)' };
  return {};
};

function Diff({ diff }: { diff: string }) {
  const lines = diff.split('\n');
  return (
    <details open={lines.length <= 80}>
      <summary className="cursor-pointer text-sm text-subtle">Diff · {lines.length} lines</summary>
      <pre className="rounded bg-surface-sunken" style={{ ...MONO, overflow: 'auto', maxHeight: 520, margin: '8px 0 0' }}>
        {lines.map((line, i) => (
          <div key={i} style={{ padding: '0 12px', minHeight: '1.4em', ...DIFF_LINE_STYLE(line) }}>
            {line || ' '}
          </div>
        ))}
      </pre>
    </details>
  );
}

function Group({ group }: { group: ReviewGroup }) {
  return (
    <Stack gap="2">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Text kind="label/bold/md">{group.title}</Text>
        {group.attention ? (
          <Badge color={ATTENTION_COLOR[group.attention] ?? 'blue'} kind="outline">
            {group.attention}
          </Badge>
        ) : null}
      </div>
      {group.description ? (
        <Text kind="body/regular/sm" color="secondary">
          {group.description}
        </Text>
      ) : null}
      {group.diff ? <Diff diff={group.diff} /> : null}
    </Stack>
  );
}

const cell = (value: unknown): string =>
  value === null || value === undefined ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);

export function InputView({ span }: { span: Span }) {
  const messages = readMessages(span.attributes['input.value']);
  const modelAttr = span.attributes['llm.model_name'];
  const model = typeof modelAttr === 'string' ? modelAttr : undefined;
  const agent = span.attributes['gen_ai.agent.name'];
  const subtitle = model ? `${typeof agent === 'string' ? agent : 'model'}: ${model}` : undefined;
  if (messages.length === 0) return <Text color="secondary">No input recorded.</Text>;
  return (
    <Stack gap="3">
      <Text kind="label/bold/lg">{subtitle ? `Input (${subtitle})` : 'Input'}</Text>
      {messages.map((m, i) => (
        <div key={i}>
          {m.role && messages.length > 1 ? (
            <Text kind="body/regular/xs" color="secondary">
              {m.role}
            </Text>
          ) : null}
          <div style={PROSE}>{m.text}</div>
        </div>
      ))}
    </Stack>
  );
}

export function OutputView({ span }: { span: Span }) {
  const raw = span.attributes['output.value'];
  const structured = readStructured(raw);
  const toolCalls = readToolCalls(raw);
  const messages = readMessages(raw);

  if (!structured) {
    return (
      <Stack gap="3">
        <Text kind="label/bold/lg">Output</Text>
        {messages.length === 0 && toolCalls.length === 0 ? <Text color="secondary">No output recorded.</Text> : null}
        {messages.map((m, i) => (
          <div key={i} style={PROSE}>
            {m.text}
          </div>
        ))}
        {toolCalls.length > 0 ? <pre style={{ ...MONO, whiteSpace: 'pre-wrap' }}>{JSON.stringify(toolCalls, null, 2)}</pre> : null}
      </Stack>
    );
  }

  return (
    <Stack gap="4">
      <Text kind="label/bold/lg">Output</Text>
      {structured.partial ? (
        <Text kind="body/regular/sm" color="warning">
          The exporter truncated this output; showing what was recorded.
        </Text>
      ) : null}
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <tbody>
          {Object.entries(structured.fields).map(([key, value]) => (
            <tr key={key} className="border-b border-subtle">
              <th scope="row" className="text-subtle" style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 16px 6px 0', whiteSpace: 'nowrap', fontWeight: 500, width: 1 }}>
                {humanize(key)}
              </th>
              <td style={{ padding: '6px 0', overflowWrap: 'anywhere', ...(/^[0-9a-f]{40}$/.test(String(value)) ? MONO : {}) }}>{cell(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {structured.groups.length > 0 ? (
        <Stack gap="4">
          <Text kind="label/bold/lg">Groups</Text>
          {structured.groups.map((group, i) => (
            <Group key={i} group={group} />
          ))}
        </Stack>
      ) : null}
    </Stack>
  );
}
