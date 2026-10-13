// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

export interface ArtifactFileRowProps {
  children: ReactNode;
  onClick: () => void;
}

export const ArtifactFileRow: FC<ArtifactFileRowProps> = ({ children, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    className="cursor-pointer flex items-center justify-between w-full text-left border-b border-base pb-density-md last:border-0 last:pb-0 hover:bg-subtle px-density-xs"
  >
    <Stack gap="density-xs">{children}</Stack>
  </button>
);
