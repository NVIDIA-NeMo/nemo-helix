/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Button, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { InferenceProvidersDataView } from '@studio/components/dataViews/InferenceProvidersDataView';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { CreateInferenceProviderSidePanel } from '@studio/routes/InferenceProvidersListRoute/CreateInferenceProviderSidePanel';
import type { InferenceProviderPresetId } from '@studio/routes/InferenceProvidersListRoute/CreateInferenceProviderSidePanel/inferenceProviderPresets';
import { getWorkspaceInferenceProvidersRoute } from '@studio/routes/utils';
import { type FC, useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router';

export const InferenceProvidersListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Read deep-link params once on mount
  const initialParams = useRef({
    create: searchParams.get('create') === 'true',
    preset: (searchParams.get('preset') as InferenceProviderPresetId | null) ?? undefined,
  });

  const [isCreatePanelOpen, setIsCreatePanelOpen] = useState(initialParams.current.create);

  // Clean up URL params after consuming them
  useEffect(() => {
    if (initialParams.current.create) {
      navigate(getWorkspaceInferenceProvidersRoute(workspace), { replace: true });
    }
  }, [navigate, workspace]);

  useBreadcrumbs({
    items: [
      {
        href: getWorkspaceInferenceProvidersRoute(workspace),
        slotLabel: 'Inference Providers',
      },
    ],
  });

  const addProviderButton = (
    <Button color="brand" onClick={() => setIsCreatePanelOpen(true)}>
      Add Provider
    </Button>
  );

  return (
    <AccessibleTitle title="Inference Providers">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Inference Providers"
          slotDescription="Manage inference endpoints like NVIDIA Build, OpenAI, and NIMs."
          slotActions={addProviderButton}
        />
        <InferenceProvidersDataView
          workspace={workspace}
          onCreate={() => setIsCreatePanelOpen(true)}
          attributes={{
            Stack: {
              className: 'flex-1 min-h-0',
            },
          }}
        />
      </Stack>
      <CreateInferenceProviderSidePanel
        workspace={workspace}
        open={isCreatePanelOpen}
        onClose={() => setIsCreatePanelOpen(false)}
        defaultPreset={initialParams.current.preset}
      />
    </AccessibleTitle>
  );
};
