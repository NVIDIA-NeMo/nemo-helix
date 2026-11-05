// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, type ButtonProps } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getEvaluationResultsRoute } from '@studio/routes/utils';
import { useNavigate } from 'react-router';

/**
 * Button that navigates to the evaluation results page.
 */
export const ButtonLaunchEvaluation = ({
  children = 'Launch Evaluation',
  ...buttonProps
}: ButtonProps) => {
  const navigate = useNavigate();
  const workspace = useWorkspaceFromPath();

  return (
    <Button
      color="brand"
      {...buttonProps}
      onClick={() => navigate(getEvaluationResultsRoute(workspace))}
      data-testid="button-launch-evaluation"
    >
      {children}
    </Button>
  );
};
