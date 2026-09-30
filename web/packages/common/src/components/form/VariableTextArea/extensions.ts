// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  autocompletion,
  type CompletionContext,
  type CompletionResult,
} from '@codemirror/autocomplete';
import {
  Compartment,
  EditorState,
  RangeSetBuilder,
  StateEffect,
  StateField,
} from '@codemirror/state';
import {
  Decoration,
  type DecorationSet,
  drawSelection,
  EditorView,
  tooltips,
  ViewPlugin,
  type ViewUpdate,
} from '@codemirror/view';

export const textareaLook = [
  EditorView.lineWrapping,
  EditorState.tabSize.of(2),
  drawSelection(),
  EditorView.theme({
    '&': { fontSize: '14px', backgroundColor: 'transparent' },
    '&.cm-focused': { outline: 'none' },
    '.cm-content': {
      fontFamily: 'var(--font-sans, inherit)',
      padding: '0',
    },
    '.cm-scroller': { overflow: 'auto', fontFamily: 'inherit' },
    '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'currentColor' },
  }),
];

/** Keeps the completion popup within the editor's own bounds. */
export const tooltipWithinEditor = tooltips({
  tooltipSpace: (view) => {
    const { left, right } = view.dom.getBoundingClientRect();
    return { left, right, top: 0, bottom: document.documentElement.clientHeight };
  },
});

export const setKnownVariables = StateEffect.define<Set<string>>();

export const knownVariablesField = StateField.define<Set<string>>({
  create: () => new Set<string>(),
  update(value, tr) {
    for (const e of tr.effects) if (e.is(setKnownVariables)) return e.value;
    return value;
  },
});

export const variablesCompartment = new Compartment();

/** Decides whether a token names something real. Null falls back to exact
 *  membership of `knownVariablesField`. */
export type VariableMatcher = (token: string) => boolean;

export const setVariableMatcher = StateEffect.define<VariableMatcher | null>();

export const variableMatcherField = StateField.define<VariableMatcher | null>({
  create: () => null,
  update(value, tr) {
    for (const e of tr.effects) if (e.is(setVariableMatcher)) return e.value;
    return value;
  },
});

const TOKEN_RE = /\{\{\s*([^}]*?)\s*\}\}/g;

const knownMark = Decoration.mark({ class: 'nv-variable-known' });
const unknownMark = Decoration.mark({ class: 'nv-variable-unknown' });

function buildDecorations(view: EditorView): DecorationSet {
  const known = view.state.field(knownVariablesField, false) ?? new Set<string>();
  const matcher = view.state.field(variableMatcherField, false) ?? null;
  const builder = new RangeSetBuilder<Decoration>();
  for (const { from, to } of view.visibleRanges) {
    const text = view.state.doc.sliceString(from, to);
    TOKEN_RE.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = TOKEN_RE.exec(text)) !== null) {
      const start = from + match.index;
      const end = start + match[0].length;
      const token = match[1];
      builder.add(
        start,
        end,
        (matcher ? matcher(token) : known.has(token)) ? knownMark : unknownMark
      );
    }
  }
  return builder.finish();
}

export const variableDecorations = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;
    constructor(view: EditorView) {
      this.decorations = buildDecorations(view);
    }
    update(u: ViewUpdate) {
      const effectsTouchedKnown = u.transactions.some((tr) =>
        tr.effects.some((e) => e.is(setKnownVariables) || e.is(setVariableMatcher))
      );
      if (u.docChanged || u.viewportChanged || effectsTouchedKnown) {
        this.decorations = buildDecorations(u.view);
      }
    }
  },
  { decorations: (v) => v.decorations }
);

export interface VariableCompletionsOptions {
  getVariables: () => readonly { name: string; description?: string }[];
}

export function variableCompletions({ getVariables }: VariableCompletionsOptions) {
  function source(context: CompletionContext): CompletionResult | null {
    const line = context.state.doc.lineAt(context.pos);
    const before = context.state.doc.sliceString(
      Math.max(line.from, context.pos - 64),
      context.pos
    );
    const open = before.lastIndexOf('{{');
    if (open === -1) return null;
    const between = before.slice(open + 2);
    if (between.includes('}}')) return null;
    if (!/^[\w.\-[\]\s]*$/.test(between)) return null;

    const from = context.pos - between.length;
    const variables = getVariables();
    return {
      from,
      to: context.pos,
      validFor: /^[\w.\-[\]\s]*$/,
      options: variables.map((v) => ({
        label: v.name,
        detail: v.description,
        apply: (view, _completion, applyFrom, applyTo) => {
          const after = view.state.doc.sliceString(
            applyTo,
            Math.min(view.state.doc.length, applyTo + 2)
          );
          const consumeClose = after.startsWith('}}') ? 2 : 0;
          const insert = `${v.name}}}`;
          view.dispatch({
            changes: { from: applyFrom, to: applyTo + consumeClose, insert },
            selection: { anchor: applyFrom + insert.length },
          });
        },
      })),
    };
  }
  return autocompletion({
    activateOnTyping: true,
    override: [source],
    tooltipClass: () => 'nv-kui-autocomplete',
  });
}
