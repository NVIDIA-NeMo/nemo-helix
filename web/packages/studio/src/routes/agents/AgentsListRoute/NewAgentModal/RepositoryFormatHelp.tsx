// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import { REPOSITORY_EXAMPLES } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type { FC } from 'react';

/** What the Repository field accepts, shown from its info icon. */
export const RepositoryFormatHelp: FC = () => (
  <Stack gap="density-sm" className="max-w-[460px]">
    <Text kind="body/regular/sm">
      A GitHub repository, or an SSH URL for any other Git host. Add <code>@branch</code>,{' '}
      <code>@tag</code>, or <code>@commit</code> after the repository, and <code>#directory</code>{' '}
      for the folder holding agent.yaml.
    </Text>
    {REPOSITORY_EXAMPLES.map(({ label, value }) => (
      <Stack key={value} gap="0">
        <Text kind="body/semibold/xs">{label}</Text>
        <code className="break-all text-xs">{value}</code>
      </Stack>
    ))}
  </Stack>
);
