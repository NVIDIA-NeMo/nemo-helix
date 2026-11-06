// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Anchor, Text } from '@nvidia/foundations-react-core';
import voyagerArt from '@studio/assets/voyager.svg';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getOptimizerInsightRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

interface OriginatingInsightLinkProps {
  insightId: string;
}

/**
 * The Voyager mark and an "Originating insight" label on one line, both linking to the insight
 * detail page. Sits in the top-right corner of the insight cards that reference it.
 */
export const OriginatingInsightLink: FC<OriginatingInsightLinkProps> = ({ insightId }) => {
  const workspace = useWorkspaceFromPath();
  return (
    <Anchor asChild>
      <Link
        to={getOptimizerInsightRoute(workspace, insightId)}
        className="flex shrink-0 items-center gap-density-sm no-underline hover:!bg-transparent hover:underline"
      >
        {/* The artwork is authored at 90x78 and stretches to its box, so keep that ratio. */}
        <img src={voyagerArt} alt="" aria-hidden className="h-6 w-[28px]" />
        <Text kind="body/bold/md" color="brand">
          Originating insight
        </Text>
      </Link>
    </Anchor>
  );
};
