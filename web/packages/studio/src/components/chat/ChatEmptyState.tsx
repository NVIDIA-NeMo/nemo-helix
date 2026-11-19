// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, StatusMessage } from '@nvidia/foundations-react-core';
import { ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import type { FC } from 'react';
import { generatePath, useNavigate } from 'react-router';

interface ChatEmptyStateProps {
  /** Shows connect/deploy actions when no models are available. */
  hasModels: boolean;
}

export const ChatEmptyState: FC<ChatEmptyStateProps> = ({ hasModels }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

  const headline = hasModels ? 'Ready' : 'No Models Available';
  const subhead = hasModels
    ? 'Prompt your model to get started.'
    : 'Connect an inference provider or create a deployment to start chatting.';

  return (
    <div className="flex h-full w-full items-center justify-center p-8">
      <StatusMessage
        size="medium"
        slotHeading={headline}
        slotSubheading={subhead}
        slotFooter={
          !hasModels && (
            <Flex gap="density-md" justify="center" wrap="wrap">
              <Button
                kind="secondary"
                onClick={() => navigate(generatePath(ROUTES.workspace.deployments, { workspace }))}
              >
                Create Deployment
              </Button>
              <Button
                kind="primary"
                color="brand"
                onClick={() =>
                  navigate(generatePath(ROUTES.workspace.inferenceProviders, { workspace }))
                }
              >
                Connect Inference Provider
              </Button>
            </Flex>
          )
        }
      />
    </div>
  );
};
