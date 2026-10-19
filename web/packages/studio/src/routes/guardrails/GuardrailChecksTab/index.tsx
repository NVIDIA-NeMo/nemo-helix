// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useGuardrailsGetGuardrailConfig } from '@nemo/sdk/generated/platform/guardrails';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { useGuardrailChecksForConfig } from '@studio/api/guardrail-checks/hooks';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  GUARDRAIL_CHECKS_DEFAULT_SUB_TAB,
  isGuardrailChecksSubTab,
} from '@studio/routes/guardrails/GuardrailChecksTab/constants';
import { GuardrailTestCasesEditor } from '@studio/routes/guardrails/GuardrailChecksTab/GuardrailTestCasesEditor';
import { useDraftRailsConfig } from '@studio/routes/guardrails/GuardrailForm/useDraftRailsConfig';
import { getGuardrailChecksSubTabRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { Navigate, useParams } from 'react-router';

export const GuardrailChecksTab: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { guardrailConfigName } = useRequiredPathParams([ROUTE_PARAMS.guardrailConfigName]);
  const subTab = useParams()[ROUTE_PARAMS.guardrailChecksSubTab];
  const isValidSubTab = isGuardrailChecksSubTab(subTab);

  // The unsaved edits from the Configuration tab; both tabs sit inside GuardrailFormProvider.
  const { isDirty, draftConfig } = useDraftRailsConfig();

  const { data: config, isPending: isConfigPending } = useGuardrailsGetGuardrailConfig(
    workspace,
    guardrailConfigName,
    { query: { enabled: isValidSubTab && Boolean(workspace && guardrailConfigName) } }
  );

  const {
    data: checksPage,
    isPending: isChecksPending,
    isError: isChecksError,
  } = useGuardrailChecksForConfig(
    workspace,
    config?.id ?? '',
    { page_size: 1000 },
    { enabled: isValidSubTab && Boolean(config?.id) }
  );

  // Must stay above the loading gate: both queries are disabled for a hand-typed sub-tab,
  // and a disabled query reports `isPending`, so falling through would park the redirect
  // behind a "Loading checks..." spinner that never resolves.
  if (!isValidSubTab) {
    return (
      <Navigate
        replace
        to={getGuardrailChecksSubTabRoute(
          workspace,
          guardrailConfigName,
          GUARDRAIL_CHECKS_DEFAULT_SUB_TAB
        )}
      />
    );
  }

  if (isConfigPending || isChecksPending) {
    return <Loading description="Loading checks..." />;
  }
  // Unreachable in-app: GuardrailDetailRoute runs the same config query and renders its own
  // error state instead of this <Outlet />. Kept so `config.id` narrows below.
  if (!config?.name) {
    return null;
  }
  // Never fall back to an empty list here — an editor showing "Run 0 Tests" is indistinguishable
  // from a config that genuinely has no test cases, so a failed fetch would read as data loss.
  if (isChecksError || !checksPage) {
    return (
      <Flex
        className="w-full min-h-80"
        align="center"
        justify="center"
        data-testid="guardrail-checks-tab"
      >
        <Text className="text-feedback-danger">Failed to load guardrail tests.</Text>
      </Flex>
    );
  }

  return (
    <GuardrailTestCasesEditor
      workspace={workspace}
      configId={config.id}
      configData={config.data}
      isDirty={isDirty}
      draftConfig={draftConfig}
      checks={checksPage.data}
      subTab={subTab}
    />
  );
};
