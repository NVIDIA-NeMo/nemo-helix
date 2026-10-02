// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CreateCustomizationStart } from '@studio/components/CreateCustomizationStart';
import type { StartSelection } from '@studio/components/CreateCustomizationStart/types';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  getNewCustomizationFormRoute,
  getWorkspaceCustomizationJobListRoute,
} from '@studio/routes/utils';
import { useNavigate } from 'react-router';

/** The "how do you want to start?" page. The form it leads to lives at `/new/scratch`. */
export const NewCustomizationRoute = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

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
    // A template's values go as router state rather than in the URL: they are a whole form
    // payload, and the provisioning behind them has already happened by the time we get here.
    navigate(
      getNewCustomizationFormRoute(workspace),
      selection.optionId === 'template'
        ? { state: { initialValues: selection.initialValues } }
        : undefined
    );
  };

  return <CreateCustomizationStart workspace={workspace} onContinue={handleContinue} />;
};
