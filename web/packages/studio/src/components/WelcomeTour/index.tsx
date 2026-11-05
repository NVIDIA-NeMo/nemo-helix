// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { steps } from '@studio/components/WelcomeTour/steps';
import { TourController } from '@studio/components/WelcomeTour/TourController';
import '@studio/components/WelcomeTour/theme.css';
import { TourProvider } from 'modern-tour';
import { type FC, useMemo } from 'react';
import { matchPath, useLocation } from 'react-router';

export const WelcomeTour: FC = () => {
  const { pathname } = useLocation();

  const routeSteps = useMemo(
    () => steps.find((group) => matchPath(group.route, pathname))?.steps ?? [],
    [pathname]
  );

  if (routeSteps.length === 0) return null;

  const tourOptions = {
    steps: routeSteps,
    showNavigation: false,
    showProgress: false,
    showCloseButton: false,
    closeOnOverlayClick: false,
    closeOnEscape: true,
    animation: 'smooth' as const,
    spotlightPadding: 8,
  };

  return (
    <TourProvider options={tourOptions}>
      <TourController />
    </TourProvider>
  );
};
