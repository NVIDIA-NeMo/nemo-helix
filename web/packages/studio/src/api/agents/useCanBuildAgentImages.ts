// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobsGetExecutionProfiles } from '@nemo/sdk/generated/platform/jobs';

// Matches _DEFAULT_PROFILE in nemo_agents_plugin/jobs/package_agent.py, which requires a subprocess backend.
const PACKAGE_JOB_PROFILE = 'default';

// Undefined while loading or unreadable, so a build the platform might accept is never blocked.
export const useCanBuildAgentImages = (): boolean | undefined => {
  const { data: profiles } = useJobsGetExecutionProfiles();
  if (!profiles) return undefined;
  return profiles.some(
    (profile) => profile.backend === 'subprocess' && profile.profile === PACKAGE_JOB_PROFILE
  );
};
