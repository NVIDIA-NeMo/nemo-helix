// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair/index';
import { render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';

describe('KVPair', () => {
  it('should render a label and a value', () => {
    render(<KVPair label="Test" value="Value" />);
    expect(screen.getByText('Test')).toBeInTheDocument();
    expect(screen.getByText('Value')).toBeInTheDocument();
  });

  it('should show the default value "—" when value is an empty string', () => {
    render(<KVPair label="Test" value="" />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('should show the default value "—" when value is undefined', () => {
    render(<KVPair label="Test" value={undefined as unknown as string} />);
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('should show a custom defaultValue when value is an empty string', () => {
    render(<KVPair label="Test" value="" defaultValue="N/A" />);
    expect(screen.getByText('N/A')).toBeInTheDocument();
  });

  it.each([
    ['narrow', 96],
    ['medium', 160],
    ['wide', 320],
  ])('should render with different widths for size and orientation horizontal', (size, width) => {
    render(
      <KVPair
        label="Test"
        value="Value"
        orientation="horizontal"
        size={size as ComponentProps<typeof KVPair>['size']}
      />
    );
    expect(screen.getByText('Test')).toHaveClass(`w-[${width}px]`);
  });
});
