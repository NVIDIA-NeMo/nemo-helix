// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChatEmptyState } from '@nemo/common/src/components/Chat/ChatEmptyState';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('ChatEmptyState', () => {
  it('renders without an action', () => {
    render(<ChatEmptyState slotHeading="Chat Unavailable" slotSubheading="No deployment." />);

    expect(screen.getByText('Chat Unavailable')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('renders slotAction and lets clicks through the nebula overlay', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <ChatEmptyState
        slotHeading="Chat Unavailable"
        slotAction={<button onClick={onClick}>Deploy this model</button>}
      />
    );

    await user.click(screen.getByRole('button', { name: 'Deploy this model' }));

    expect(onClick).toHaveBeenCalledOnce();
  });
});
