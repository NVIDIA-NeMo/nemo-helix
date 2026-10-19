// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGraphMotionDuration } from '@studio/components/DagCanvas/motion';
import { MIN_GRAPH_ZOOM } from '@studio/components/DagCanvas/viewport';
import { useNodes, useReactFlow } from '@xyflow/react';
import { type FC, useEffect, useRef } from 'react';

interface FitNodesControllerProps {
  readonly nodeIds?: readonly string[];
}

export const FitNodesController: FC<FitNodesControllerProps> = ({ nodeIds }) => {
  const { fitView } = useReactFlow();
  const nodes = useNodes();
  const fitKey = nodeIds?.join('|') ?? '';
  const lastFitKey = useRef('');

  useEffect(() => {
    if (!fitKey) {
      lastFitKey.current = '';
      return;
    }
    if (fitKey === lastFitKey.current) return;
    const visibleNodes = nodes.filter(({ id }) => nodeIds?.includes(id));
    if (visibleNodes.length !== nodeIds?.length) return;
    lastFitKey.current = fitKey;
    requestAnimationFrame(() => {
      fitView({
        nodes: visibleNodes.map(({ id }) => ({ id })),
        duration: getGraphMotionDuration(),
        padding: 0.18,
        minZoom: MIN_GRAPH_ZOOM,
        maxZoom: 1,
      });
    });
  }, [fitKey, fitView, nodeIds, nodes]);

  return null;
};
