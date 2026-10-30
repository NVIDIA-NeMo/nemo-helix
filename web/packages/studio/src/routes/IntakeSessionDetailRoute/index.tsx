// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type SessionDetailRouteContext,
  SessionDetailView,
} from '@studio/components/IntakeDetail/SessionDetailView';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';

interface SessionDetailContentProps {
  sessionId: string;
  routeContext?: SessionDetailRouteContext;
}

export const SessionDetailContent: FC<SessionDetailContentProps> = (props) => {
  const workspace = useWorkspaceFromPath();
  return <SessionDetailView workspace={workspace} {...props} />;
};

export const IntakeSessionDetailRoute: FC = () => {
  const { sessionId } = useRequiredPathParams([ROUTE_PARAMS.sessionId]);
  return <SessionDetailContent sessionId={sessionId} />;
};
