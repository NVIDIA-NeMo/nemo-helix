// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SpanPayloadFormatToggle } from '@studio/components/IntakeDetail/IntakeComponents/SpanPayloadFormatToggle';
import { SpanPayloadView } from '@studio/components/IntakeDetail/IntakeComponents/SpanPayloadView';
import { useSpanPayloadFormat } from '@studio/components/IntakeDetail/IntakeComponents/useSpanPayloadFormat';
import { fireEvent, renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { type FC, useState } from 'react';

const EMPTY_MESSAGE = 'No payload';

// Wires the toggle to the body the way a payload accordion section does: shared
// state in the parent, control on the trigger, payload in the content slot.
const PayloadSection: FC<{ value: string | null | undefined; onSelect?: () => void }> = ({
  value,
  onSelect,
}) => {
  const format = useSpanPayloadFormat(value, onSelect);
  return (
    <>
      <SpanPayloadFormatToggle state={format} payloadLabel="input" />
      <SpanPayloadView value={value} format={format.format} emptyMessage={EMPTY_MESSAGE} />
    </>
  );
};

// Mirrors the tree view, where selecting another span swaps the payload into the
// same mounted section rather than remounting it.
const SwitchingSection: FC<{ first: string; second: string }> = ({ first, second }) => {
  const [value, setValue] = useState(first);
  return (
    <>
      <button type="button" onClick={() => setValue(second)}>
        Select next span
      </button>
      <PayloadSection value={value} />
    </>
  );
};

const codeText = () => screen.getByTestId('nv-code-snippet-code');

describe('SpanPayloadView', () => {
  it('renders small payloads immediately', () => {
    renderRoute(<SpanPayloadView value="small payload" emptyMessage={EMPTY_MESSAGE} />);

    expect(screen.queryByLabelText('Rendering payload')).not.toBeInTheDocument();
    expect(screen.getByText('small payload')).toBeInTheDocument();
  });

  it('shows a loader before rendering large payloads', async () => {
    const payload = 'x'.repeat(20_001);

    renderRoute(<SpanPayloadView value={payload} emptyMessage={EMPTY_MESSAGE} />);

    expect(screen.getByLabelText('Rendering payload')).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.queryByLabelText('Rendering payload')).not.toBeInTheDocument()
    );
    expect(codeText()).toHaveTextContent(payload);
  });

  it('defers rendering when a rerender swaps in a large payload', () => {
    const payload = 'x'.repeat(20_001);
    renderRoute(<SwitchingSection first="small payload" second={payload} />);

    expect(screen.queryByLabelText('Rendering payload')).not.toBeInTheDocument();

    // fireEvent, not userEvent: awaiting a click lets the 0ms timer fire, which
    // would hide the very spinner this asserts on.
    fireEvent.click(screen.getByRole('button', { name: 'Select next span' }));

    // Guards that deferral is re-armed on rerender at all; it cannot observe the
    // wasted mount itself, since React replaces that commit before it is queried.
    expect(screen.getByLabelText('Rendering payload')).toBeInTheDocument();
  });

  it('defers rendering when a large payload switches renderers', async () => {
    const payload = 'x'.repeat(20_001);
    renderRoute(<PayloadSection value={payload} />);

    // Let the payload get past the initial spinner first, or the assertion below
    // passes on a spinner that was never hidden.
    await waitFor(() =>
      expect(screen.queryByLabelText('Rendering payload')).not.toBeInTheDocument()
    );

    // fireEvent, not userEvent: awaiting a click lets the 0ms timer fire, which
    // would hide the very spinner this asserts on.
    fireEvent.click(screen.getByRole('button', { name: 'View input as markdown' }));

    // The text is unchanged, so deferral has to key on the renderer too: markdown
    // mounts a different one over the same 20k characters.
    expect(screen.getByLabelText('Rendering payload')).toBeInTheDocument();
  });

  it('renders the empty state for blank payloads', () => {
    renderRoute(<SpanPayloadView value="   " emptyMessage={EMPTY_MESSAGE} />);

    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument();
  });

  it('pretty-prints JSON payloads', () => {
    renderRoute(
      <SpanPayloadView value='{"role":"user"}' format="json" emptyMessage={EMPTY_MESSAGE} />
    );

    expect(codeText()).toHaveTextContent('"role": "user"');
  });

  it('indents JSON payloads without rewriting their literals', () => {
    renderRoute(
      <SpanPayloadView
        value='{"span_id":9007199254740993,"score":1.0,"temp":1e2,"delta":-0}'
        format="json"
        emptyMessage={EMPTY_MESSAGE}
      />
    );

    // A JSON.stringify(JSON.parse(...)) round trip renders these as
    // 9007199254740992, 1, 100, and 0.
    expect(codeText()).toHaveTextContent('"span_id": 9007199254740993');
    expect(codeText()).not.toHaveTextContent('9007199254740992');
    expect(codeText()).toHaveTextContent('"score": 1.0');
    expect(codeText()).toHaveTextContent('"temp": 1e2');
    expect(codeText()).toHaveTextContent('"delta": -0');
  });

  it('renders markdown payloads as formatted content once the renderer loads', async () => {
    renderRoute(<SpanPayloadView value="# Findings" format="md" emptyMessage={EMPTY_MESSAGE} />);

    // Generous timeout: the renderer is a dynamic import, so a cold module cache
    // can take longer than findBy's 1s default.
    expect(
      await screen.findByRole('heading', { name: 'Findings' }, { timeout: 5_000 })
    ).toBeInTheDocument();
  });

  it('falls back to raw when JSON is requested for a payload that is not JSON', () => {
    renderRoute(<SpanPayloadView value="plain text" format="json" emptyMessage={EMPTY_MESSAGE} />);

    expect(codeText()).toHaveTextContent('plain text');
  });
});

describe('SpanPayloadFormatToggle', () => {
  it('opens JSON payloads in the JSON view', async () => {
    renderRoute(<PayloadSection value='{"role":"user"}' />);

    // findBy, not getBy: awaiting keeps CodeSnippet's async highlight inside
    // the test that caused it.
    expect(await screen.findByRole('button', { name: 'View input as JSON' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(codeText()).toHaveTextContent('"role": "user"');
  });

  it('opens plain-text payloads in the raw view', () => {
    renderRoute(<PayloadSection value="plain text" />);

    expect(screen.getByRole('button', { name: 'View input as raw text' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
  });

  it('disables the JSON view for payloads that are not JSON', () => {
    renderRoute(<PayloadSection value="plain text" />);

    expect(screen.getByRole('button', { name: 'View input as JSON' })).toBeDisabled();
  });

  it('switches the payload to the selected format', async () => {
    const user = userEvent.setup();
    renderRoute(<PayloadSection value="# Findings" />);

    await user.click(screen.getByRole('button', { name: 'View input as markdown' }));

    expect(
      await screen.findByRole('heading', { name: 'Findings' }, { timeout: 5_000 })
    ).toBeInTheDocument();
  });

  it('drops the JSON formatting when the raw view is selected', async () => {
    const user = userEvent.setup();
    renderRoute(<PayloadSection value='{"a":1}' />);

    // Wait for the pretty-printed JSON to actually paint: the stale-markup bug
    // this guards only shows up once the JSON view has rendered.
    await waitFor(() => expect(codeText()).toHaveTextContent('"a": 1'));

    await user.click(screen.getByRole('button', { name: 'View input as raw text' }));

    await waitFor(() => expect(codeText().textContent).toBe('{"a":1}'));
  });

  it('reports the selection so a collapsed section can open', async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    renderRoute(<PayloadSection value="plain text" onSelect={onSelect} />);

    await user.click(screen.getByRole('button', { name: 'View input as markdown' }));

    expect(onSelect).toHaveBeenCalledOnce();
  });

  it('re-derives the default when another span swaps in a different payload', async () => {
    const user = userEvent.setup();
    renderRoute(<SwitchingSection first='{"a":1}' second='{"b":2}' />);

    await user.click(screen.getByRole('button', { name: 'View input as raw text' }));
    expect(screen.getByRole('button', { name: 'View input as raw text' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );

    await user.click(screen.getByRole('button', { name: 'Select next span' }));

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'View input as JSON' })).toHaveAttribute(
        'aria-pressed',
        'true'
      )
    );
  });

  it('keeps the selection when another span carries an identical payload', async () => {
    const user = userEvent.setup();
    renderRoute(<SwitchingSection first='{"a":1}' second='{"a":1}' />);

    await user.click(screen.getByRole('button', { name: 'View input as raw text' }));
    await user.click(screen.getByRole('button', { name: 'Select next span' }));

    expect(screen.getByRole('button', { name: 'View input as raw text' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
  });

  it('renders nothing when there is no payload to format', () => {
    renderRoute(<PayloadSection value="   " />);

    expect(screen.queryByRole('button', { name: /^View input as/ })).not.toBeInTheDocument();
    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument();
  });
});

describe('SpanPayloadChatView', () => {
  const CHAT_PAYLOAD = JSON.stringify({
    content: {
      messages: [
        { role: 'system', content: 'You are a careful assistant.' },
        { role: 'user', content: 'What changed?' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{ id: 'call-1', function: { name: 'search', arguments: '{"q":"diff"}' } }],
        },
        { role: 'tool', tool_call_id: 'call-1', content: 'Two files.' },
      ],
    },
    choices: [
      {
        finish_reason: 'stop',
        message: { content: 'Two files changed.', reasoning_content: 'Read the tool result.' },
      },
    ],
  });

  /** Every turn still folded away, in the order the conversation renders them. */
  const folded = () =>
    screen.getAllByTestId('chat-turn-detail').filter((detail) => !detail.hasAttribute('open'));

  it('opens an OpenAI chat payload as a conversation', async () => {
    renderRoute(<PayloadSection value={CHAT_PAYLOAD} />);

    expect(await screen.findByRole('button', { name: 'View input as a chat' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    // Generous timeout: message bodies render through the lazily imported markdown chunk.
    expect(await screen.findByText('What changed?', {}, { timeout: 5_000 })).toBeInTheDocument();
    expect(screen.getByText('Two files changed.')).toBeInTheDocument();
  });

  it('folds the system, tool, and reasoning turns away until they are opened', async () => {
    renderRoute(<SpanPayloadView value={CHAT_PAYLOAD} emptyMessage={EMPTY_MESSAGE} />);

    await screen.findByText('System prompt', {}, { timeout: 5_000 });
    expect(folded().map((detail) => detail.textContent)).toEqual([
      expect.stringContaining('System prompt'),
      expect.stringContaining('search'),
      expect.stringContaining('Tool result'),
      expect.stringContaining('Reasoning'),
    ]);

    // The user and assistant exchange is the part that needs no click.
    expect(screen.getByText('What changed?')).toBeVisible();
  });

  it('reveals a folded turn when it is opened', async () => {
    const user = userEvent.setup();
    renderRoute(<SpanPayloadView value={CHAT_PAYLOAD} emptyMessage={EMPTY_MESSAGE} />);

    await user.click(await screen.findByText('Reasoning', {}, { timeout: 5_000 }));

    expect(folded().map((detail) => detail.textContent)).not.toContainEqual(
      expect.stringContaining('Reasoning')
    );
    expect(screen.getByText('Read the tool result.')).toBeInTheDocument();
  });

  it('disables the chat view for payloads that are not a conversation', async () => {
    renderRoute(<PayloadSection value='{"message":"just prose"}' />);

    expect(await screen.findByRole('button', { name: 'View input as a chat' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'View input as JSON' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
  });

  it('still offers the JSON view of a chat payload', async () => {
    const user = userEvent.setup();
    renderRoute(<PayloadSection value={CHAT_PAYLOAD} />);

    await user.click(await screen.findByRole('button', { name: 'View input as JSON' }));

    await waitFor(() => expect(codeText()).toHaveTextContent('"role": "user"'));
  });

  it('falls back to the default when a chat view is requested for other JSON', async () => {
    renderRoute(<SpanPayloadView value='{"a":1}' format="chat" emptyMessage={EMPTY_MESSAGE} />);

    expect(screen.queryByTestId('span-payload-chat')).not.toBeInTheDocument();
    // Awaited, so CodeSnippet's async highlight settles inside the test that caused it.
    await waitFor(() => expect(codeText()).toHaveTextContent('"a": 1'));
  });
});
