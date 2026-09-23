// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DataDesignerStart } from '@studio/components/DataDesignerStart';
import type { StartSelection } from '@studio/components/DataDesignerStart/types';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import type { DataDesignerGeneratedState } from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import { getDataDesignerJobBuildRoute, getDataDesignerJobListRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

export const NewDataDesignerJobRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

  useBreadcrumbs({
    items: [
      { href: getDataDesignerJobListRoute(workspace), slotLabel: 'Data Designer' },
      { slotLabel: 'New fileset' },
    ],
  });

  const handleContinue = (selection: StartSelection) => {
    switch (selection.optionId) {
      case 'scratch':
        navigate(getDataDesignerJobBuildRoute(workspace));
        break;
      case 'template':
        navigate(`${getDataDesignerJobBuildRoute(workspace)}?template=${selection.templateId}`);
        break;
      case 'ai': {
        const state: DataDesignerGeneratedState = { generatedJobRequest: selection.jobRequest };
        navigate(getDataDesignerJobBuildRoute(workspace), { state });
        break;
      }
    }
  };

  return (
    <AccessibleTitle title="Create a fileset">
      <DataDesignerStart workspace={workspace} onContinue={handleContinue} />
    </AccessibleTitle>
  );
};
