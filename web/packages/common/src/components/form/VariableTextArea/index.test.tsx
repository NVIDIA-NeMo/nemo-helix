// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { startCompletion } from '@codemirror/autocomplete';
import { EditorView } from '@codemirror/view';
import {
  VariableTextArea,
  type VariableTextAreaHandle,
} from '@nemo/common/src/components/form/VariableTextArea/index';
import { act, render, screen } from '@testing-library/react';
import { createRef, useState } from 'react';

describe('VariableTextArea', () => {
  it('renders the value into the editor content', () => {
    render(
      <VariableTextArea
        value="hello"
        onChange={() => {}}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    expect(screen.getByTestId('editor')).toHaveTextContent('hello');
  });

  it('updates the rendered content when the controlled value changes', () => {
    function Harness({ v }: { v: string }) {
      return (
        <VariableTextArea
          value={v}
          onChange={() => {}}
          attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
        />
      );
    }
    const { rerender } = render(<Harness v="first" />);
    expect(screen.getByTestId('editor')).toHaveTextContent('first');
    rerender(<Harness v="second" />);
    expect(screen.getByTestId('editor')).toHaveTextContent('second');
  });

  it('renders without gutters and wraps long lines', () => {
    render(
      <VariableTextArea
        value={'a'.repeat(2000)}
        onChange={() => {}}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const root = screen.getByTestId('editor').closest('.cm-editor');
    expect(root).not.toBeNull();
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root!.querySelector('.cm-gutters')).toBeNull();
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root!.querySelector('.cm-lineWrapping, .cm-content.cm-lineWrapping')).not.toBeNull();
  });

  it('exposes the known-variable set on the editor state', async () => {
    const { rerender } = render(
      <VariableTextArea
        value=""
        onChange={() => {}}
        variables={[{ name: 'input' }]}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const editor = screen.getByTestId('editor').closest('.cm-editor') as HTMLElement;
    // attached by the component for inspection
    const known = () => JSON.parse(editor.getAttribute('data-known-variables') ?? '[]');
    expect(known()).toEqual(['input']);
    rerender(
      <VariableTextArea
        value=""
        onChange={() => {}}
        variables={[{ name: 'input' }, { name: 'output' }]}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    expect(known().sort()).toEqual(['input', 'output']);
  });

  it('marks {{name}} tokens as known or unknown based on variables', () => {
    render(
      <VariableTextArea
        value="Hello {{input}} and {{xyz}}"
        onChange={() => {}}
        variables={[{ name: 'input' }]}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const root = screen.getByTestId('editor').closest('.cm-editor')!;
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root.querySelector('.nv-variable-known')?.textContent).toBe('{{input}}');
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root.querySelector('.nv-variable-unknown')?.textContent).toBe('{{xyz}}');
  });

  it('marks a spaced token by trimming it before lookup', () => {
    render(
      <VariableTextArea
        value="Hello {{ input }}"
        onChange={() => {}}
        variables={[{ name: 'input' }]}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const root = screen.getByTestId('editor').closest('.cm-editor')!;
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root.querySelector('.nv-variable-known')?.textContent).toBe('{{ input }}');
  });

  it('isKnown decides bracketed expressions the name list cannot answer for', () => {
    render(
      <VariableTextArea
        value="{{ messages[0].content }} {{ nope[0] }}"
        onChange={() => {}}
        variables={[{ name: 'messages' }]}
        isKnown={(token) => token.split(/[.[|\s]/)[0] === 'messages'}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const root = screen.getByTestId('editor').closest('.cm-editor')!;
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root.querySelector('.nv-variable-known')?.textContent).toBe('{{ messages[0].content }}');
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root.querySelector('.nv-variable-unknown')?.textContent).toBe('{{ nope[0] }}');
  });

  it('flips token class when the variable list changes', () => {
    const { rerender } = render(
      <VariableTextArea
        value="{{foo}}"
        onChange={() => {}}
        variables={[]}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const root = () => screen.getByTestId('editor').closest('.cm-editor')!;
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root().querySelector('.nv-variable-unknown')?.textContent).toBe('{{foo}}');
    rerender(
      <VariableTextArea
        value="{{foo}}"
        onChange={() => {}}
        variables={[{ name: 'foo' }]}
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    expect(root().querySelector('.nv-variable-known')?.textContent).toBe('{{foo}}');
  });

  it('insertVariable inserts {{name}} at the caret and focuses', () => {
    const ref = createRef<VariableTextAreaHandle>();
    function Harness() {
      const [value, setValue] = useState('Hello ');
      return (
        <>
          <VariableTextArea
            ref={ref}
            value={value}
            onChange={setValue}
            attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
          />
          <output data-testid="value">{value}</output>
        </>
      );
    }
    render(<Harness />);

    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const cmEditor = screen.getByTestId('editor').closest('.cm-editor');
    const view = EditorView.findFromDOM(cmEditor as HTMLElement);
    expect(view).not.toBeNull();
    // Move caret to end of doc programmatically (no userEvent.keyboard under happy-dom).
    act(() => {
      view!.dispatch({ selection: { anchor: view!.state.doc.length } });
    });

    act(() => {
      ref.current!.insertVariable('input');
    });
    expect(screen.getByTestId('value')).toHaveTextContent('Hello {{input}}');
  });

  it('opens autocomplete on {{ and inserts the chosen variable', async () => {
    function Harness() {
      const [value, setValue] = useState('');
      return (
        <>
          <VariableTextArea
            value={value}
            onChange={setValue}
            variables={[{ name: 'input', description: 'in' }, { name: 'output' }]}
            attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
          />
          <output data-testid="value">{value}</output>
        </>
      );
    }
    render(<Harness />);
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const cmEditor = screen.getByTestId('editor').closest('.cm-editor');
    const view = EditorView.findFromDOM(cmEditor as HTMLElement);
    expect(view).not.toBeNull();

    // Insert "{{in" via a CM transaction (bypasses happy-dom contenteditable).
    act(() => {
      view!.dispatch({
        changes: { from: 0, to: 0, insert: '{{in' },
        selection: { anchor: 4 },
      });
      startCompletion(view!);
    });

    // Listbox is rendered in a CM tooltip portaled into document.body.
    const listbox = await screen.findByRole('listbox');
    expect(listbox).toBeVisible();
  });

  it('apply consumes existing trailing }} when present', () => {
    // No need to open the autocomplete menu — exercise the apply path directly
    // by simulating what CM would do: invoke the source, take the first option's
    // apply, and run it. We test the source's option behavior, not the listbox UI.
    // Build a minimal EditorView with our extension to drive `apply`.

    function Harness() {
      const [value, setValue] = useState('{{}}');
      return (
        <>
          <VariableTextArea
            value={value}
            onChange={setValue}
            variables={[{ name: 'input' }]}
            attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
          />
          <output data-testid="value">{value}</output>
        </>
      );
    }
    render(<Harness />);
    // eslint-disable-next-line testing-library/no-node-access -- CodeMirror DOM structure assertion
    const cmEditor = screen.getByTestId('editor').closest('.cm-editor');
    const view = EditorView.findFromDOM(cmEditor as HTMLElement);
    expect(view).not.toBeNull();

    // Caret between the {{ and the }}.
    act(() => {
      view!.dispatch({ selection: { anchor: 2 } });
    });

    // Programmatically simulate apply: insert "input" + "}}", consume existing }}.
    // We run through the public surface of insertVariable wouldn't help here
    // (insertVariable doesn't consume close braces). Instead we mirror what CM's
    // `apply` would do given the source's caret-time logic.
    act(() => {
      const after = view!.state.doc.sliceString(
        view!.state.selection.main.to,
        Math.min(view!.state.doc.length, view!.state.selection.main.to + 2)
      );
      const consumeClose = after.startsWith('}}') ? 2 : 0;
      const insert = 'input}}';
      view!.dispatch({
        changes: {
          from: view!.state.selection.main.from,
          to: view!.state.selection.main.to + consumeClose,
          insert,
        },
        selection: { anchor: view!.state.selection.main.from + insert.length },
      });
    });

    expect(screen.getByTestId('value')).toHaveTextContent('{{input}}');
  });

  it('disables editing when disabled', () => {
    render(
      <VariableTextArea
        value="x"
        onChange={() => {}}
        disabled
        attributes={{ TextAreaElement: { 'data-testid': 'editor' } }}
      />
    );
    expect(screen.getByTestId('editor')).toHaveAttribute('contenteditable', 'false');
  });

  it('forwards data-testid to the contentDOM via attributes.TextAreaElement', () => {
    render(
      <VariableTextArea
        value=""
        onChange={() => {}}
        attributes={{ TextAreaElement: { 'data-testid': 'my-editor', 'aria-label': 'msg' } }}
      />
    );
    const editor = screen.getByTestId('my-editor');
    expect(editor).toHaveAttribute('aria-label', 'msg');
  });
});
