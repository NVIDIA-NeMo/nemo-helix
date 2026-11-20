// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { QueryClient } from '@tanstack/react-query';

/** Lists, details, groups, and metrics share the workspace Intake URL prefix. */
export const invalidateSessionDeletion = (client: QueryClient, workspace: string) => {
  const root = `/apis/intake/v2/workspaces/${workspace}/`;
  return client.invalidateQueries({
    predicate: ({ queryKey }) => typeof queryKey[0] === 'string' && queryKey[0].startsWith(root),
  });
};
