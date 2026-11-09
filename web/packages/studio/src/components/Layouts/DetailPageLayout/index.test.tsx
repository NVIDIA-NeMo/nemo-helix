// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Button, TabsContent } from '@nvidia/foundations-react-core';
import {
  DetailPageLayout,
  type DetailPageLayoutProps,
} from '@studio/components/Layouts/DetailPageLayout';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

type Tab = 'details' | 'logs';

const TABS: DetailPageLayoutProps<Tab>['tabs'] = [
  { value: 'details', label: 'Details' },
  { value: 'logs', label: 'Logs' },
];

const renderLayout = (props: Partial<DetailPageLayoutProps<Tab>> = {}) =>
  render(
    <DetailPageLayout<Tab>
      title="my-entity"
      tabs={TABS}
      selectedTab="details"
      onTabChange={vi.fn()}
      {...props}
    >
      <TabsContent value="details">Details content</TabsContent>
      <TabsContent value="logs">Logs content</TabsContent>
    </DetailPageLayout>
  );

describe('DetailPageLayout', () => {
  it('renders the title, badges, and description in the header', () => {
    renderLayout({
      slotBadges: <Badge>Deployed</Badge>,
      slotDescription: 'An entity description',
    });

    expect(screen.getByTestId('nv-page-header-heading')).toHaveTextContent('my-entity');
    expect(screen.getByText('Deployed')).toBeInTheDocument();
    expect(screen.getByText('An entity description')).toBeInTheDocument();
  });

  it('renders a tab per entry and shows the selected tab content', () => {
    renderLayout({ selectedTab: 'logs' });

    expect(screen.getByRole('tab', { name: 'Details' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Logs' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('Logs content')).toBeVisible();
  });

  it('reports the clicked tab', async () => {
    const onTabChange = vi.fn();
    renderLayout({ onTabChange });

    await userEvent.click(screen.getByRole('tab', { name: 'Logs' }));

    expect(onTabChange).toHaveBeenCalledWith('logs');
  });

  it('disables tabs marked disabled', () => {
    renderLayout({ tabs: [TABS[0], { ...TABS[1], disabled: true }] });

    expect(screen.getByRole('tab', { name: 'Logs' })).toBeDisabled();
  });

  it('renders the actions slot', () => {
    renderLayout({ slotActions: <Button>Run Evaluation</Button> });

    expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeInTheDocument();
  });
});
