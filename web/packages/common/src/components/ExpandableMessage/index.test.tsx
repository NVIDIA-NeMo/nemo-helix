// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('Message', () => {
  it('renders the message text', () => {
    render(<ExpandableMessage message="Hello, world!" />);

    expect(screen.getByText('Hello, world!')).toBeInTheDocument();
  });

  it('renders error message with danger styling', () => {
    render(<ExpandableMessage errorMessage="Something went wrong" />);

    const errorText = screen.getByText('Something went wrong');
    expect(errorText).toBeInTheDocument();
    expect(errorText).toHaveClass('text-feedback-danger');
  });

  it('prioritizes error message over regular message', () => {
    render(<ExpandableMessage message="Regular message" errorMessage="Error message" />);

    expect(screen.getByText('Error message')).toBeInTheDocument();
    expect(screen.queryByText('Regular message')).not.toBeInTheDocument();
  });

  it('shows skeleton when loading', () => {
    render(<ExpandableMessage loading message="This should not appear" />);

    // Message text should not appear when loading
    expect(screen.queryByText('This should not appear')).not.toBeInTheDocument();
    // Skeleton is rendered (has progressbar role for accessibility)
    expect(screen.getByTestId('nv-skeleton')).toBeInTheDocument();
  });

  describe('show more/show less', () => {
    const longMessage = 'A'.repeat(350);
    const shortMessage = 'Short message';

    it('does not show show-more link for short messages', () => {
      render(<ExpandableMessage message={shortMessage} />);

      expect(screen.queryByText('Show more')).not.toBeInTheDocument();
      expect(screen.getByText(shortMessage)).toBeInTheDocument();
    });

    it('truncates long messages and shows "Show more" link', () => {
      render(<ExpandableMessage message={longMessage} />);

      expect(screen.getByText('Show more')).toBeInTheDocument();
      // Should show truncated text with ellipsis
      expect(screen.getByText(/\.\.\.$/)).toBeInTheDocument();
      // Should not show the full message
      expect(screen.queryByText(longMessage)).not.toBeInTheDocument();
    });

    it('expands message when "Show more" is clicked', async () => {
      const user = userEvent.setup();
      render(<ExpandableMessage message={longMessage} />);

      const showMoreLink = screen.getByText('Show more');
      await user.click(showMoreLink);

      // Should now show full message
      expect(screen.getByText(longMessage)).toBeInTheDocument();
      // Link should now say "Show less"
      expect(screen.getByText('Show less')).toBeInTheDocument();
    });

    it('collapses message when "Show less" is clicked', async () => {
      const user = userEvent.setup();
      render(<ExpandableMessage message={longMessage} />);

      // Expand first
      await user.click(screen.getByText('Show more'));
      // Then collapse
      await user.click(screen.getByText('Show less'));

      // Should be truncated again
      expect(screen.queryByText(longMessage)).not.toBeInTheDocument();
      expect(screen.getByText('Show more')).toBeInTheDocument();
    });

    it('toggles with the keyboard and reports aria-expanded', async () => {
      const user = userEvent.setup();
      render(<ExpandableMessage message={longMessage} />);

      const toggle = screen.getByRole('button', { name: 'Show more' });
      expect(toggle).toHaveAttribute('aria-expanded', 'false');

      await user.tab();
      expect(toggle).toHaveFocus();
      await user.keyboard('{Enter}');

      expect(screen.getByText(longMessage)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Show less' })).toHaveAttribute(
        'aria-expanded',
        'true'
      );
    });

    it('respects custom character limit', () => {
      const customLimit = 50;
      const messageJustOverLimit = 'B'.repeat(60);

      render(<ExpandableMessage message={messageJustOverLimit} characterLimit={customLimit} />);

      expect(screen.getByText('Show more')).toBeInTheDocument();
    });

    it('does not show show-more link when message equals character limit', () => {
      const exactLimitMessage = 'C'.repeat(300);

      render(<ExpandableMessage message={exactLimitMessage} characterLimit={300} />);

      expect(screen.queryByText('Show more')).not.toBeInTheDocument();
    });

    it('works with error messages too', async () => {
      const user = userEvent.setup();
      const longErrorMessage = 'E'.repeat(350);

      render(<ExpandableMessage errorMessage={longErrorMessage} />);

      expect(screen.getByText('Show more')).toBeInTheDocument();

      await user.click(screen.getByText('Show more'));

      expect(screen.getByText(longErrorMessage)).toBeInTheDocument();
      expect(screen.getByText(longErrorMessage)).toHaveClass('text-feedback-danger');
    });
  });
});
