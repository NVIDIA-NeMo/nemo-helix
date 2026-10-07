// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asRecord } from '@studio/util/guards';

/** Strict Jinja subset: dot paths, length, array loops, if/else, comments and whitespace control.
 *  Reject unsupported syntax and missing fields rather than changing the evaluation's prompts. */

export class UnsupportedTemplateError extends Error {}

type Context = Record<string, unknown>;

type Node =
  | { kind: 'text'; value: string }
  | { kind: 'output'; path: string; filter?: 'length' }
  | { kind: 'for'; name: string; path: string; body: Node[] }
  | { kind: 'if'; path: string; body: Node[]; orElse: Node[] };

type Token = Extract<Node, { kind: 'text' | 'output' }> | { kind: 'tag'; value: string };

const PATH = /^[A-Za-z_]\w*(?:\.\w+)*$/;
const FOR_TAG = /^for\s+([A-Za-z_]\w*)\s+in\s+(\S+)$/;
const IF_TAG = /^if\s+(\S+)$/;

const unsupported = (detail: string): never => {
  throw new UnsupportedTemplateError(detail);
};

/** Split into text and tags, applying `-` whitespace control as Jinja does. */
const tokenize = (template: string): Token[] => {
  const tokens: Token[] = [];
  const pattern = /\{\{(-?)([\s\S]*?)(-?)\}\}|\{%(-?)([\s\S]*?)(-?)%\}|\{#(-?)[\s\S]*?(-?)#\}/g;
  let cursor = 0;
  let trimNext = false;

  for (let match = pattern.exec(template); match; match = pattern.exec(template)) {
    let text = template.slice(cursor, match.index);
    if (trimNext) text = text.replace(/^\s+/, '');
    const trimBefore = match[1] === '-' || match[4] === '-' || match[7] === '-';
    if (trimBefore) text = text.replace(/\s+$/, '');
    if (text) tokens.push({ kind: 'text', value: text });

    if (match[2] !== undefined) tokens.push(parseOutput(match[2].trim()));
    else if (match[5] !== undefined) tokens.push({ kind: 'tag', value: match[5].trim() });
    trimNext = match[3] === '-' || match[6] === '-' || match[8] === '-';
    cursor = pattern.lastIndex;
  }

  let rest = template.slice(cursor);
  if (trimNext) rest = rest.replace(/^\s+/, '');
  rest = rest.replace(/\n$/, '');
  if (rest) tokens.push({ kind: 'text', value: rest });
  return tokens;
};

const parseOutput = (expression: string): Extract<Node, { kind: 'output' }> => {
  const [path, filter, ...extra] = expression.split('|').map((part) => part.trim());
  if (extra.length > 0 || !PATH.test(path)) unsupported(`expression "{{ ${expression} }}"`);
  if (filter !== undefined && filter !== 'length') unsupported(`filter "${filter}"`);
  return { kind: 'output', path, filter: filter === 'length' ? 'length' : undefined };
};

/** Parse tokens up to one of `stopAt`, returning the nodes and the tag that stopped them. */
const parseNodes = (
  tokens: Token[],
  position: { index: number },
  stopAt: readonly string[]
): { nodes: Node[]; stoppedBy?: string } => {
  const nodes: Node[] = [];
  while (position.index < tokens.length) {
    const token = tokens[position.index++];
    if (token.kind !== 'tag') {
      nodes.push(token);
      continue;
    }
    if (stopAt.includes(token.value)) return { nodes, stoppedBy: token.value };

    const forTag = FOR_TAG.exec(token.value);
    if (forTag && PATH.test(forTag[2])) {
      const { nodes: body, stoppedBy } = parseNodes(tokens, position, ['endfor']);
      if (stoppedBy !== 'endfor') unsupported('unclosed {% for %}');
      nodes.push({ kind: 'for', name: forTag[1], path: forTag[2], body });
      continue;
    }

    const ifTag = IF_TAG.exec(token.value);
    if (ifTag && PATH.test(ifTag[1])) {
      const then = parseNodes(tokens, position, ['else', 'endif']);
      const orElse = then.stoppedBy === 'else' ? parseNodes(tokens, position, ['endif']) : then;
      if (orElse.stoppedBy !== 'endif') unsupported('unclosed {% if %}');
      nodes.push({
        kind: 'if',
        path: ifTag[1],
        body: then.nodes,
        orElse: then.stoppedBy === 'else' ? orElse.nodes : [],
      });
      continue;
    }

    unsupported(`tag "{% ${token.value} %}"`);
  }
  return { nodes };
};

const lookup = (context: Context, path: string): unknown => {
  let value: unknown = context;
  for (const key of path.split('.')) {
    const record = asRecord(value);
    if (!record || !Object.hasOwn(record, key)) return unsupported(`undefined value "${path}"`);
    value = record[key];
  }
  return value;
};

/** Python's `str()` for the JSON values a dataset row can hold. */
const pythonStr = (value: unknown): string => {
  if (typeof value === 'string') return value;
  if (value === null) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return String(value);
  return unsupported('rendering a list or mapping directly');
};

const truthy = (value: unknown): boolean => {
  if (Array.isArray(value)) return value.length > 0;
  const record = asRecord(value);
  if (record) return Object.keys(record).length > 0;
  return Boolean(value);
};

const renderNodes = (nodes: Node[], context: Context): string =>
  nodes
    .map((node) => {
      switch (node.kind) {
        case 'text':
          return node.value;
        case 'output': {
          const value = lookup(context, node.path);
          if (node.filter !== 'length') return pythonStr(value);
          if (typeof value === 'string' || Array.isArray(value)) return String(value.length);
          const record = asRecord(value);
          if (record) return String(Object.keys(record).length);
          return unsupported(`length of "${node.path}"`);
        }
        case 'for': {
          const items = lookup(context, node.path);
          if (!Array.isArray(items)) return unsupported(`looping over "${node.path}"`);
          return items
            .map((item, index) =>
              renderNodes(node.body, {
                ...context,
                [node.name]: item,
                loop: {
                  index: index + 1,
                  index0: index,
                  first: index === 0,
                  last: index === items.length - 1,
                  length: items.length,
                },
              })
            )
            .join('');
        }
        case 'if':
          return renderNodes(truthy(lookup(context, node.path)) ? node.body : node.orElse, context);
      }
    })
    .join('');

/** Match the evaluator's StrictUndefined behavior and `{**row, "item": row}` context. */
export const renderPromptTemplate = (template: string, row: Record<string, unknown>): string => {
  const { nodes } = parseNodes(tokenize(template), { index: 0 }, []);
  return renderNodes(nodes, { ...row, item: row });
};
