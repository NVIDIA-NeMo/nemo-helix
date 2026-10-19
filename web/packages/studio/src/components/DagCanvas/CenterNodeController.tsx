// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGraphMotionDuration } from '@studio/components/DagCanvas/motion';
import { useNodes, useReactFlow } from '@xyflow/react';
import { type FC, useEffect, useRef } from 'react';

export const CenterNodeController: FC<{
  centerNodeId?: string | null;
  requestNonce?: number;
}> = ({ centerNodeId, requestNonce }) => {
  const { getNodesBounds, getZoom, setCenter } = useReactFlow();
  const nodes = useNodes();
  const lastRequest = useRef('');

  useEffect(() => {
    if (!centerNodeId || requestNonce === undefined) return;
    const request = `${centerNodeId}:${requestNonce}`;
    if (request === lastRequest.current || !nodes.some(({ id }) => id === centerNodeId)) return;
    const bounds = getNodesBounds([centerNodeId]);
    const zoom = getZoom();
    if (
      ![bounds.x, bounds.y, bounds.width, bounds.height, zoom].every(Number.isFinite) ||
      bounds.width <= 0 ||
      bounds.height <= 0
    ) {
      return;
    }
    lastRequest.current = request;
    setCenter(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2, {
      zoom,
      duration: getGraphMotionDuration(),
    });
  }, [centerNodeId, getNodesBounds, getZoom, nodes, requestNonce, setCenter]);

  return null;
};
