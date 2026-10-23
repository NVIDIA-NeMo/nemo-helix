// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InfoCircle } from '@nemo/icons/icons';
import { Badge, Text, Tooltip } from '@nvidia/foundations-react-core';
import { featureFlags } from '@studio/constants/featureFlags';
import { FeatureFlags } from '@studio/constants/featureFlags/featureFlags';
import { PREVIEW } from '@studio/constants/featureFlags/utils';
import { tooltipClassName } from '@studio/styles/common';
import { FC } from 'react';

interface FeatureFlagBadgeProps {
  flag: keyof FeatureFlags;
}

export const FeatureFlagBadge: FC<FeatureFlagBadgeProps> = ({ flag }) => {
  if (featureFlags[flag] !== PREVIEW) return null;

  return (
    <Tooltip
      slotContent={
        <div className={tooltipClassName}>
          <Text>
            This feature is only enabled for internal deployments and is not available in production
            releases.
          </Text>
        </div>
      }
      side="bottom"
    >
      <Badge color="gray" kind="solid" className="ml-2 align-middle">
        <InfoCircle width="12px" height="12px" />
        Early Preview
      </Badge>
    </Tooltip>
  );
};
