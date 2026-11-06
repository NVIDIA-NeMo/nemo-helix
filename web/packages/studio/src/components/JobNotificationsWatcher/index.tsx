// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobNotifications } from '@studio/hooks/useJobNotifications';
import { useSelectedWorkspace } from '@studio/providers/workspace';

/** Renders nothing; keeps job-finished notifications running on every page of the app. */
export const JobNotificationsWatcher = () => {
  const { selectedWorkspace } = useSelectedWorkspace();
  useJobNotifications(selectedWorkspace);
  return null;
};
