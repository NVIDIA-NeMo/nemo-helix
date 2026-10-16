// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusMessage } from '@nvidia/foundations-react-core';
import { MousePointerClick } from 'lucide-react';
import { type FC } from 'react';

export const NoSpanSelected: FC = () => (
  <div className="flex min-h-[320px] items-center justify-center rounded-lg border border-dashed border-base">
    <StatusMessage
      size="medium"
      slotMedia={<MousePointerClick aria-hidden className="h-12 w-12 text-secondary" />}
      slotHeading="No span selected"
      slotSubheading="Select a span in the tree to see its input, output and timing."
    />
  </div>
);
