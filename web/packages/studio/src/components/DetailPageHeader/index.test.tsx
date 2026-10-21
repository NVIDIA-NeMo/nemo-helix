// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, TabsRoot } from '@nvidia/foundations-react-core';
import { DetailPageHeader } from '@studio/components/DetailPageHeader';
import { render, screen } from '@testing-library/react';

const renderHeader = (props: Partial<Parameters<typeof DetailPageHeader>[0]> = {}) =>
  render(
    <TabsRoot value="details">
      <DetailPageHeader
        heading="my-model"
        tabs={[
          { value: 'details', label: 'Details' },
          { value: 'training', label: 'Training', disabled: true },
        ]}
        {...props}
      />
    </TabsRoot>
  );

describe('DetailPageHeader', () => {
  it('renders the heading, status, tags, description and actions', () => {
    renderHeader({
      slotStatus: <span>Deployed</span>,
      slotTags: <span>LoRA</span>,
      description: 'Tuned for support tickets',
      slotActions: <Button>Open Job</Button>,
    });

    expect(screen.getByRole('heading', { level: 1, name: 'my-model' })).toBeInTheDocument();
    expect(screen.getByText('Deployed')).toBeInTheDocument();
    expect(screen.getByText('LoRA')).toBeInTheDocument();
    expect(screen.getByText('Tuned for support tickets')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Job' })).toBeInTheDocument();
  });

  it('renders a tab per item and disables the disabled ones', () => {
    renderHeader();

    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Training' })).toBeDisabled();
  });
});
