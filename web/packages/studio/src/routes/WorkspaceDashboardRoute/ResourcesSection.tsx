// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Book, Bug, OpenExternal, Route } from '@nemo/icons/icons';
import {
  Card,
  Divider,
  Flex,
  Grid,
  Stack,
  Text,
  VerticalNav,
  VerticalNavItem,
} from '@nvidia/foundations-react-core';
import { ReportTraceModal } from '@studio/components/ReportTraceModal';
import { TELEMETRY_ENABLED } from '@studio/constants/environment';
import { LINK_DOCS_SDK, LINK_DOCS_STUDIO, LINK_GITHUB_ISSUES } from '@studio/constants/links';
import { FC, useState } from 'react';

export const ResourcesSection: FC = () => {
  const [isTraceModalOpen, setIsTraceModalOpen] = useState(false);

  const helpItems = [
    { id: 'report-bug', children: 'Report a Bug', href: LINK_GITHUB_ISSUES },
    ...(TELEMETRY_ENABLED ? [{ id: 'report-trace', children: 'Report a Trace', href: '#' }] : []),
  ];

  return (
    <>
      <Card className="w-full">
        <Grid cols={{ sm: 1, md: 2 }} gap="density-5xl">
          {/* Getting Started Documentation */}
          <Stack gap="density-lg" className="min-w-0">
            <Text kind="label/bold/xl">Documentation</Text>
            <VerticalNav
              className="w-full bg-surface-raised border-0"
              items={[
                { id: 'studio-docs', children: 'Studio Documentation', href: LINK_DOCS_STUDIO },
                { id: 'sdk-docs', children: 'SDK Documentation', href: LINK_DOCS_SDK },
              ]}
              renderLink={(item) => (
                <>
                  <VerticalNavItem
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    slotStart={<Book width={16} height={16} />}
                  >
                    {item.children}
                    <Flex align="center" className="ml-auto">
                      <OpenExternal width={14} height={14} />
                    </Flex>
                  </VerticalNavItem>
                  {item.id === 'studio-docs' && <Divider />}
                </>
              )}
            />
          </Stack>

          {/* Help & Support */}
          {helpItems.length > 0 && (
            <Stack gap="density-lg" className="min-w-0">
              <Text kind="label/bold/xl">Help & Support</Text>
              <VerticalNav
                className="w-full bg-surface-raised border-0"
                items={helpItems}
                renderLink={(item) => {
                  if (item.id === 'report-bug') {
                    return (
                      <>
                        <VerticalNavItem
                          href={item.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          slotStart={<Bug width={16} height={16} />}
                        >
                          {item.children}
                          <Flex align="center" className="ml-auto">
                            <OpenExternal width={14} height={14} />
                          </Flex>
                        </VerticalNavItem>
                        {TELEMETRY_ENABLED && <Divider />}
                      </>
                    );
                  }
                  if (item.id === 'report-trace') {
                    return (
                      <VerticalNavItem
                        href="#"
                        onClick={(e) => {
                          e.preventDefault();
                          setIsTraceModalOpen(true);
                        }}
                        slotStart={<Route width={16} height={16} />}
                      >
                        {item.children}
                      </VerticalNavItem>
                    );
                  }
                  return null;
                }}
              />
            </Stack>
          )}
        </Grid>
      </Card>

      <ReportTraceModal open={isTraceModalOpen} onClose={() => setIsTraceModalOpen(false)} />
    </>
  );
};
