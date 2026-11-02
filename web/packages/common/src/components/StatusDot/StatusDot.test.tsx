// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusDot } from '@nemo/common/src/components/StatusDot';
import { render, screen } from '@testing-library/react';

describe('StatusDot', () => {
  it('colors and labels the dot from the shared status mapping', () => {
    render(<StatusDot status="running" />);
    expect(screen.getByRole('img', { name: 'Running' })).toHaveClass(
      'nv-status-indicator--color-green'
    );
  });

  it('falls back to accent tokens for colors StatusIndicator lacks', () => {
    render(<StatusDot status="deleted" />);
    expect(screen.getByRole('img', { name: 'Deleted' })).toHaveClass(
      '[--color:var(--text-color-accent-gray)]'
    );
  });

  it('honors a custom status config and label override', () => {
    render(
      <StatusDot status="x" statusConfig={{ x: { label: 'X', color: 'yellow' } }} label="Custom" />
    );
    expect(screen.getByRole('img', { name: 'Custom' })).toHaveClass(
      'nv-status-indicator--color-yellow'
    );
  });
});
