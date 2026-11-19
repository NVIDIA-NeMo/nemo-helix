// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceContext } from '@studio/providers/workspace/WorkspaceContext';
import { type FC, type ReactNode, useState } from 'react';

interface MockWorkspaceProviderProps {
  children: ReactNode;
  defaultWorkspace?: string;
  isWorkspaceUnauthorized?: boolean;
  isWorkspaceLoading?: boolean;
}

/**
 * Mock WorkspaceProvider for tests that provides a simple workspace context
 * without dependencies on auth or localStorage.
 */
export const MockWorkspaceProvider: FC<MockWorkspaceProviderProps> = ({
  children,
  defaultWorkspace = 'test-workspace',
  isWorkspaceUnauthorized = false,
  isWorkspaceLoading = false,
}) => {
  const [selectedWorkspace, setSelectedWorkspace] = useState<string | undefined>(defaultWorkspace);

  return (
    <WorkspaceContext.Provider
      value={{
        selectedWorkspace,
        setSelectedWorkspace,
        isWorkspaceUnauthorized,
        isWorkspaceLoading,
      }}
    >
      {children}
    </WorkspaceContext.Provider>
  );
};
