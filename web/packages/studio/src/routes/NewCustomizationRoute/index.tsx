// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CreateCustomizationStart } from '@studio/components/CreateCustomizationStart';
import type { StartSelection } from '@studio/components/CreateCustomizationStart/types';
import { NewCustomizationForm } from '@studio/components/NewCustomizationForm';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getWorkspaceCustomizationJobListRoute } from '@studio/routes/utils';
import {
  getInitialFormValuesFromState,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { useMemo, useState } from 'react';
import { useLocation, useSearchParams } from 'react-router';

export const NewCustomizationRoute = () => {
  const workspace = useWorkspaceFromPath();
  const [searchParams] = useSearchParams();
  const initialModel = searchParams.get('model') ?? undefined;

  const { state } = useLocation();
  const stateValues = useMemo(() => getInitialFormValuesFromState(state), [state]);

  /**
   * What the start page resolved to. `null` means it is still on screen; a value — possibly
   * the `undefined` of "build from scratch" — means the form has taken over.
   */
  const [started, setStarted] = useState<{ values?: CustomizationFormFields } | null>(null);

  // Arriving with a model to customize or a job to clone already answers how to start, so
  // those entry points go straight to the form rather than asking a question twice.
  const isDeepLink = stateValues !== undefined || initialModel !== undefined;

  useBreadcrumbs({
    items: [
      {
        href: getWorkspaceCustomizationJobListRoute(workspace),
        slotLabel: 'Models',
      },
      {
        slotLabel: 'New Fine-Tuned Model',
      },
    ],
  });

  const handleContinue = (selection: StartSelection) => {
    setStarted({ values: selection.optionId === 'scratch' ? undefined : selection.initialValues });
  };

  if (!isDeepLink && started === null) {
    return <CreateCustomizationStart workspace={workspace} onContinue={handleContinue} />;
  }

  return (
    <NewCustomizationForm
      workspace={workspace}
      initialModel={initialModel}
      initialValues={stateValues ?? started?.values}
    />
  );
};
