/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { useGuardrailsGetGuardrailConfigSuspense } from '@nemo/sdk/generated/platform/guardrails';
import { Flex, PageHeader, Stack, Tabs } from '@nvidia/foundations-react-core';
import { SuspenseOutlet } from '@studio/components/Layouts/SuspenseOutlet';
import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { GuardrailDetailActions } from '@studio/routes/guardrails/GuardrailDetailRoute/GuardrailDetailActions';
import { GuardrailFormProvider } from '@studio/routes/guardrails/GuardrailForm/GuardrailFormProvider';
import { GuardrailHeaderActions } from '@studio/routes/guardrails/GuardrailForm/GuardrailHeaderActions';
import {
  getGuardrailChecksRoute,
  getGuardrailConfigRoute,
  getGuardrailsRoute,
  getGuardrailVirtualModelsRoute,
} from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { type FC } from 'react';
import { Link, matchPath, useLocation } from 'react-router';

export const GuardrailDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { guardrailConfigName } = useRequiredPathParams([ROUTE_PARAMS.guardrailConfigName]);

  const location = useLocation();
  const match = matchPath(
    { path: `${ROUTES.workspace.guardrailDetail}/:selectedTab`, end: false },
    location.pathname
  );
  const {
    params: { selectedTab },
  } = match ?? { params: { selectedTab: 'config' } };

  useBreadcrumbs({
    items: [
      { href: getGuardrailsRoute(workspace), slotLabel: 'Guardrails' },
      { slotLabel: guardrailConfigName },
    ],
  });

  const { data: config } = useGuardrailsGetGuardrailConfigSuspense(workspace, guardrailConfigName);

  return (
    <AccessibleTitle title={`Guardrail config ${guardrailConfigName}`}>
      <GuardrailFormProvider config={config}>
        <Stack className="w-full min-h-full p-density-2xl" gap="density-xl">
          <PageHeader
            slotHeading={
              <Flex gap="density-sm" align="center" justify="between">
                <span className="min-w-0 truncate" title={config.name}>
                  {config.name}
                </span>
                <Flex gap="density-sm" align="center">
                  <GuardrailHeaderActions />
                  <GuardrailDetailActions config={config} />
                </Flex>
              </Flex>
            }
          />

          <Tabs
            // Tabs used purely for navigation (renderLink); override KUI's default overflow:hidden.
            className="overflow-visible"
            value={selectedTab}
            items={[
              {
                value: 'config',
                children: 'Configuration',
                href: getGuardrailConfigRoute(workspace, guardrailConfigName),
              },
              {
                value: 'checks',
                children: 'Test and Validate',
                href: getGuardrailChecksRoute(workspace, guardrailConfigName),
              },
              {
                value: 'virtual-models',
                children: 'Virtual Models',
                href: getGuardrailVirtualModelsRoute(workspace, guardrailConfigName),
              },
            ]}
            renderLink={(item) => <Link to={item.href!}>{item.children}</Link>}
          />

          <SuspenseOutlet />
        </Stack>
      </GuardrailFormProvider>
    </AccessibleTitle>
  );
};
