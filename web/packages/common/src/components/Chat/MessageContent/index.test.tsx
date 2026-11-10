// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MessageContent } from '@nemo/common/src/components/Chat/MessageContent';
import { renderListItemChildren } from '@nemo/common/src/components/Chat/MessageContent/helpers';
import { MarkdownParagraph } from '@nemo/common/src/components/Chat/MessageContent/MarkdownParagraph';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Components } from 'react-markdown';

const expectNoDirectParagraphChild = (listItem: HTMLElement): void => {
  // The list marker alignment depends on the first paragraph being unwrapped inside list items.
  // eslint-disable-next-line testing-library/no-node-access
  expect(listItem.querySelector(':scope > p')).not.toBeInTheDocument();
};

const expectListParagraphDensity = (paragraph: HTMLElement): void => {
  expect(paragraph).toHaveClass('text-sm', 'leading-6');
  expect(paragraph).not.toHaveClass('leading-[160%]');
};

describe('MessageContent', () => {
  it('renders markdown headings and list items with visible list styling', () => {
    render(
      <MessageContent
        content={
          '# Overview\n\n## Plan\n\nRead the route carefully.\n\n- Read the route\n- Update the renderer\n\n### Details'
        }
      />
    );

    expect(screen.getByRole('heading', { level: 1, name: 'Overview' })).toHaveClass(
      'mb-density-sm',
      'mt-density-2xl'
    );
    expect(screen.getByTestId('chat-message-content-text')).toHaveClass('[&>:first-child]:mt-0');
    expect(screen.getByRole('heading', { level: 2, name: 'Plan' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 2, name: 'Plan' })).toHaveClass(
      'mb-density-sm',
      'mt-density-xl'
    );
    expect(screen.getByRole('heading', { level: 3, name: 'Details' })).toHaveClass(
      'mb-density-xs',
      'mt-density-lg'
    );
    expect(screen.getByTestId('chat-message-content-text')).toHaveClass('text-sm', 'leading-6');
    expect(screen.getByTestId('chat-message-content-text')).not.toHaveClass('text-base');
    expect(screen.getByText('Read the route carefully.')).toHaveClass('mb-density-md');
    expect(screen.getByRole('list')).toHaveClass('my-density-md', 'list-disc');
    expect(screen.getByRole('list')).not.toHaveClass('space-y-0.5');
    const listItems = screen.getAllByRole('listitem');
    expect(listItems).toHaveLength(2);
    expect(listItems[0]).toHaveTextContent('Read the route');
    expect(listItems[0]).toHaveClass(
      'mb-density-xs',
      'whitespace-normal',
      'pl-density-xs',
      'text-sm',
      'leading-6',
      'last:mb-0',
      '[&>p]:my-0'
    );
  });

  it('keeps the top margin on a heading that follows a code block', () => {
    render(<MessageContent content={'Intro.\n\n```bash\nls\n```\n\n### After the code'} />);

    expect(screen.getByRole('heading', { level: 3, name: 'After the code' })).toHaveClass(
      'mt-density-lg'
    );
    const [intro, afterCode] = screen.getAllByTestId('chat-message-content-text');
    expect(intro).toHaveClass('[&>:first-child]:mt-0');
    expect(afterCode).toHaveTextContent('After the code');
    expect(afterCode).not.toHaveClass('[&>:first-child]:mt-0');
  });

  it('drops the top margin on a heading that opens a blockquote or list item', () => {
    render(<MessageContent content={'> ## Note\n\n- ### Item'} />);

    expect(screen.getByTestId('chat-message-content-text')).toHaveClass(
      '[&_blockquote>:first-child]:mt-0',
      '[&_li>:first-child]:mt-0'
    );
    expect(
      within(screen.getByRole('blockquote')).getByRole('heading', { level: 2, name: 'Note' })
    ).toBeInTheDocument();
    expect(
      within(screen.getByRole('listitem')).getByRole('heading', { level: 3, name: 'Item' })
    ).toBeInTheDocument();
  });

  it('indents bullet lists like ordered lists so markers stay inside the reply', () => {
    render(<MessageContent content={'- One\n- Two'} />);

    expect(screen.getByRole('list')).toHaveClass('pl-density-2xl');
  });

  it('keeps loose ordered list content aligned with the number marker', () => {
    render(
      <MessageContent
        content={[
          '2.',
          '',
          '**First item** with a paragraph.',
          '',
          '3.',
          '',
          '**Second item** with a paragraph.',
        ].join('\n')}
      />
    );

    const orderedList = screen.getByRole('list');
    expect(screen.getByTestId('chat-message-content-text')).toHaveClass('whitespace-normal');
    expect(orderedList.tagName).toBe('OL');
    expect(orderedList).toHaveAttribute('start', '2');
    expect(orderedList).toHaveClass('my-density-md', 'list-decimal', 'pl-density-2xl');
    expect(orderedList).not.toHaveClass('pl-density-lg');

    const listItems = within(orderedList).getAllByRole('listitem');
    expect(listItems).toHaveLength(2);
    expectNoDirectParagraphChild(listItems[0]);
    expect(listItems[0]).toHaveTextContent('First item with a paragraph.');
    expectNoDirectParagraphChild(listItems[1]);
    expect(listItems[1]).toHaveTextContent('Second item with a paragraph.');
  });

  it('keeps unwrapped list item paragraphs at list item density', () => {
    render(
      <ul>
        <li>{renderListItemChildren(<MarkdownParagraph>Wrapped item</MarkdownParagraph>)}</li>
      </ul>
    );

    expectListParagraphDensity(screen.getByText('Wrapped item'));
  });

  it('keeps indented loose ordered list content aligned with the number marker', () => {
    render(
      <MessageContent
        content={[
          '2.',
          '',
          '   **TS client generation** via Orval.',
          '',
          '3.',
          '',
          '   **Custom axios fetcher.** Each generated service uses a mutator.',
        ].join('\n')}
      />
    );

    const orderedList = screen.getByRole('list');
    const listItems = within(orderedList).getAllByRole('listitem');

    expect(orderedList).toHaveAttribute('start', '2');
    expect(listItems).toHaveLength(2);
    expectNoDirectParagraphChild(listItems[0]);
    expect(listItems[0]).toHaveTextContent('TS client generation via Orval.');
    expectNoDirectParagraphChild(listItems[1]);
    expect(listItems[1]).toHaveTextContent(
      'Custom axios fetcher. Each generated service uses a mutator.'
    );
  });

  it('renders inline code with grey backgrounds and prose font', () => {
    render(<MessageContent content="Run `pnpm test` after editing." />);

    expect(screen.getByText('pnpm test')).toHaveClass(
      'bg-gray-050',
      'dark:bg-gray-800',
      'font-sans',
      'text-sm'
    );
    expect(screen.getByText('pnpm test')).not.toHaveClass('text-[0.95em]');
    expect(screen.getByText('pnpm test')).not.toHaveClass('text-base');
    expect(screen.getByText('pnpm test')).not.toHaveClass('font-mono');
  });

  it('keeps markdown links inert by default', () => {
    render(<MessageContent content="Open [Agents](/workspaces/default/agents)." />);

    expect(screen.queryByRole('link', { name: 'Agents' })).not.toBeInTheDocument();
    expect(screen.getByText('Agents')).toBeInTheDocument();
  });

  it('uses an opt-in markdown link renderer', () => {
    const LinkComponent: Components['a'] = ({ href, children }) => <a href={href}>{children}</a>;

    render(
      <MessageContent
        content="Open [Agents](/workspaces/default/agents)."
        markdownLinkComponent={LinkComponent}
      />
    );

    expect(screen.getByRole('link', { name: 'Agents' })).toHaveAttribute(
      'href',
      '/workspaces/default/agents'
    );
  });

  it('renders markdown tables with sortable data-view table components', async () => {
    render(
      <MessageContent
        content={[
          '| Name | Status |',
          '| --- | --- |',
          '| Code blocks | Pretty |',
          '| Agent chat | Ready |',
        ].join('\n')}
      />
    );

    const table = screen.getByRole('table');
    const user = userEvent.setup();

    expect(screen.getByTestId('data-view-content')).toBeInTheDocument();
    expect(table).toHaveClass('nv-table-root');
    expect(table).toHaveClass('nv-table-root--layout-auto');
    expect(within(table).getByRole('columnheader', { name: 'Name' })).toBeInTheDocument();
    expect(within(table).getByRole('columnheader', { name: 'Status' })).toBeInTheDocument();
    expect(within(table).getByRole('cell', { name: 'Agent chat' })).toBeInTheDocument();
    expect(within(table).getByRole('cell', { name: 'Ready' })).toBeInTheDocument();
    expect(within(table).getByRole('cell', { name: 'Code blocks' })).toBeInTheDocument();
    expect(within(table).getByRole('cell', { name: 'Pretty' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Filter/i })).not.toBeInTheDocument();

    expect(within(table).getAllByRole('row')[1]).toHaveTextContent('Code blocks');

    await user.click(screen.getByRole('button', { name: /Name/i }));

    expect(within(table).getAllByRole('row')[1]).toHaveTextContent('Agent chat');

    await user.type(screen.getByPlaceholderText('Search table'), 'Agent');

    await waitFor(() => {
      expect(within(table).getByRole('cell', { name: 'Agent chat' })).toBeInTheDocument();
      expect(within(table).queryByRole('cell', { name: 'Code blocks' })).not.toBeInTheDocument();
    });
  });

  it('expands clamped markdown table cells on row click', async () => {
    const longCellText =
      'This cell intentionally contains enough detail to overflow two lines in the compact assistant table view.';

    render(
      <MessageContent
        content={[
          '| ID | Summary | Very Long Header |',
          '| --- | --- | --- |',
          `| 1 | ${longCellText} | OK |`,
        ].join('\n')}
      />
    );

    const table = screen.getByRole('table');
    const user = userEvent.setup();
    const dataRow = within(table).getAllByRole('row')[1];
    const expandableCell = within(table).getByRole('button', { name: longCellText });

    expect(dataRow).toHaveAttribute('data-row-id', 'row-0');
    expect(expandableCell).toHaveClass('overflow-hidden');
    expect(expandableCell).toHaveAttribute('aria-expanded', 'false');
    const collapsedCellContent = within(expandableCell).getByText(longCellText);
    expect(collapsedCellContent).toHaveAttribute('data-collapsed', 'true');
    expect(collapsedCellContent).toHaveClass('line-clamp-2');

    await user.click(dataRow);

    await waitFor(() => {
      const expandedCell = within(table).getByRole('button', { name: longCellText });
      const expandedCellContent = within(expandedCell).getByText(longCellText);
      expect(expandedCell).toHaveAttribute('aria-expanded', 'true');
      expect(expandedCellContent).not.toHaveAttribute('data-collapsed');
      expect(expandedCellContent).not.toHaveClass('line-clamp-2');
    });
  });
});
