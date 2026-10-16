// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Loading } from '@studio/components/Layouts/Loading';
import { type FC, type ReactNode, Suspense, useContext } from 'react';
import { Outlet, UNSAFE_RouteContext, useMatches } from 'react-router';

/**
 * Renders the child route inside a Suspense boundary keyed by that child route's id.
 *
 * Navigations run in a transition, and React will not swap an already-revealed boundary
 * back to its fallback during a transition; it keeps the previous page on screen until
 * the next one (lazy chunk or suspense query) is ready. Keying by the child route mounts
 * a fresh boundary when the child route changes, so its fallback shows immediately.
 * Param or search changes within the same child route keep the boundary, so they still
 * get the transition behavior of leaving the current content visible.
 */
export const SuspenseOutlet: FC<{ fallback?: ReactNode }> = ({ fallback = <Loading /> }) => {
  const { matches: parentMatches } = useContext(UNSAFE_RouteContext);
  const childRouteId = useMatches()[parentMatches.length]?.id;

  return (
    <Suspense key={childRouteId} fallback={fallback}>
      <Outlet />
    </Suspense>
  );
};
