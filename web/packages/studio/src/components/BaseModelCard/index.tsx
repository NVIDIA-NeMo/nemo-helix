/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { EntityCard } from '@nemo/common/src/components/EntityCard';
import { getModelMetadata } from '@nemo/common/src/components/ModelDetailsTooltip/utils';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { ChatMulti, IconDocument, World } from '@nemo/icons/icons';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Badge } from '@nvidia/foundations-react-core';
import { useMemo } from 'react';

export interface BaseModelCardProps {
  model: ModelEntity;
  isChatAvailable?: boolean;
  showCustomizationBadges?: boolean;
  onClick?: () => void;
}

const formatParameters = (numParams: number): string => {
  if (numParams >= 1_000_000_000) {
    return `${Math.round(numParams / 1_000_000_000)}B`;
  }
  if (numParams >= 1_000_000) {
    return `${Math.round(numParams / 1_000_000)}M`;
  }
  return `${numParams}`;
};

const formatContextSize = (contextSize: number): string => {
  if (contextSize >= 1_000) {
    return `${Math.round(contextSize / 1_000)}K`;
  }
  return `${contextSize}`;
};

const STAT_ICON = 'w-4 h-4 text-secondary';

export const BaseModelCard = ({
  model,
  isChatAvailable = false,
  showCustomizationBadges = true,
  onClick,
}: BaseModelCardProps) => {
  const metadata = useMemo(() => getModelMetadata(model), [model]);

  const creator = metadata?.creator;
  const description = model.description ?? metadata?.description;
  const parameters = model.spec?.base_num_parameters
    ? formatParameters(model.spec.base_num_parameters)
    : undefined;
  const contextSize = model.spec?.context_size
    ? formatContextSize(model.spec.context_size)
    : undefined;
  const isFineTuneable = Boolean(model.fileset);
  const providers = model.model_providers ?? [];

  return (
    <EntityCard
      title={model.name}
      subtitle={
        creator
          ? {
              icon: creatorToIcon(creator, { className: 'w-4 h-4 flex-shrink-0' }),
              label: creator,
            }
          : undefined
      }
      description={description}
      badges={
        showCustomizationBadges && isFineTuneable ? (
          <Badge color="purple" kind="solid">
            Fine-tunable
          </Badge>
        ) : undefined
      }
      tags={providers.map((provider) => ({
        key: provider,
        label: getPartsFromReference(provider).name,
      }))}
      stats={[
        ...(parameters
          ? [{ key: 'parameters', icon: <IconDocument className={STAT_ICON} />, label: parameters }]
          : []),
        ...(contextSize
          ? [{ key: 'context', icon: <World className={STAT_ICON} />, label: contextSize }]
          : []),
        ...(isChatAvailable
          ? [{ key: 'chat', icon: <ChatMulti className={STAT_ICON} />, label: 'Chat' }]
          : []),
      ]}
      onClick={onClick}
    />
  );
};
