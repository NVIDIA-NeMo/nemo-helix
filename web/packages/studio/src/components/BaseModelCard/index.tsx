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

import { getModelMetadata } from '@nemo/common/src/components/ModelDetailsTooltip/utils';
import { OverflowGroup } from '@nemo/common/src/components/OverflowGroup';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Badge, Button, Card, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import { MessagesSquare, File, Globe } from 'lucide-react';
import React, { useMemo } from 'react';

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
    <Card
      asChild
      interactive
      className="[&_.nv-card-content-header]:overflow-hidden"
      slotHeader={
        <Stack gap="density-sm" className="min-w-0">
          <Text
            kind="label/semibold/lg"
            className="leading-normal truncate block"
            title={model.name}
          >
            {model.name}
          </Text>
          {/* Creator */}
          {creator && (
            <Flex gap="density-sm" align="center">
              {creatorToIcon(creator, { className: 'w-4 h-4 flex-shrink-0' })}
              <Text kind="body/regular/md" className="text-placeholder truncate">
                {creator}
              </Text>
            </Flex>
          )}
        </Stack>
      }
    >
      <button onClick={onClick} className="text-left w-full h-full flex flex-col gap-2">
        {/* Description */}
        {description && (
          <Text kind="body/regular/md" className="text-secondary line-clamp-2 leading-normal">
            {description}
          </Text>
        )}
        {showCustomizationBadges && (
          <Flex gap="density-sm">
            {/* Capabilities */}
            {isFineTuneable && (
              <Badge color="purple" kind="solid">
                Fine-tunable
              </Badge>
            )}
          </Flex>
        )}

        {/* Footer */}
        <div className="grid grid-cols-[1fr_auto] items-center gap-4 border-t -mx-6 px-6 pt-4 -mb-2 border-base">
          {/* Providers (left) */}
          {providers.length > 0 ? (
            <div className="min-w-0 overflow-hidden">
              <OverflowGroup
                lines={1}
                gap={4}
                kind="popover"
                renderSeeMoreButton={({ hiddenChildren, ref }) => (
                  <Button
                    ref={ref as React.Ref<HTMLButtonElement>}
                    kind="tertiary"
                    size="small"
                    onClick={(e) => e.stopPropagation()}
                  >
                    +{hiddenChildren.length}
                  </Button>
                )}
              >
                {providers.map((provider) => (
                  <Tag key={provider} kind="outline" color="gray" readOnly>
                    {getPartsFromReference(provider).name}
                  </Tag>
                ))}
              </OverflowGroup>
            </div>
          ) : (
            <div />
          )}

          {/* Spec icons (right) */}
          <Flex gap="3" align="center">
            {parameters && (
              <Flex gap="1" align="center">
                <File className="w-4 h-4 text-secondary" />
                <Text kind="label/regular/sm">{parameters}</Text>
              </Flex>
            )}
            {contextSize && (
              <Flex gap="1" align="center">
                <Globe className="w-4 h-4 text-secondary" />
                <Text kind="label/regular/sm">{contextSize}</Text>
              </Flex>
            )}
            {isChatAvailable && (
              <Flex gap="1" align="center">
                <MessagesSquare className="w-4 h-4 text-secondary" />
                <Text kind="label/regular/sm">Chat</Text>
              </Flex>
            )}
          </Flex>
        </div>
      </button>
    </Card>
  );
};
