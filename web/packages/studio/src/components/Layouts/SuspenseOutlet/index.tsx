// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Loading } from '@studio/components/Layouts/Loading';
import { type FC, type ReactNode, Suspense, useContext } from 'react';
import { Outlet, UNSAFE_RouteContext, useMatches } from 'react-router';

interface SuspenseOutletProps {
  fallback?: ReactNode;
}

const useChildRouteId = () => {
  const { matches: parentMatches } = useContext(UNSAFE_RouteContext);
  const matches = useMatches();
  const parentRouteId = parentMatches.at(-1)?.route.id;
  const parentIndex = matches.findIndex((match) => match.id === parentRouteId);
  if (parentIndex === -1) {
    return undefined;
  }
  return matches[parentIndex + 1]?.id;
};

// Navigations run in a transition, so a revealed boundary only shows its fallback again under a new key.
export const SuspenseOutlet: FC<SuspenseOutletProps> = ({ fallback = <Loading /> }) => {
  const childRouteId = useChildRouteId();

  return (
    <Suspense key={childRouteId} fallback={fallback}>
      <Outlet />
    </Suspense>
  );
};
