// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NewCustomizationForm } from '@studio/components/NewCustomizationForm';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  getNewCustomizationJobRoute,
  getWorkspaceCustomizationJobListRoute,
} from '@studio/routes/utils';
import { getInitialFormValuesFromState } from '@studio/util/forms/customization';
import { useMemo } from 'react';
import { useLocation, useSearchParams } from 'react-router';

/** The full fine-tuning form on its own URL. */
export const NewCustomizationFormRoute = () => {
  const workspace = useWorkspaceFromPath();
  const [searchParams] = useSearchParams();
  const initialModel = searchParams.get('model') ?? undefined;

  const { state } = useLocation();
  const stateValues = useMemo(() => getInitialFormValuesFromState(state), [state]);

  useBreadcrumbs({
    items: [
      {
        href: getWorkspaceCustomizationJobListRoute(workspace),
        slotLabel: 'Models',
      },
      {
        href: getNewCustomizationJobRoute(workspace),
        slotLabel: 'New Fine-Tuned Model',
      },
      {
        slotLabel: 'Configure',
      },
    ],
  });

  return (
    <NewCustomizationForm
      workspace={workspace}
      initialModel={initialModel}
      initialValues={stateValues}
    />
  );
};
