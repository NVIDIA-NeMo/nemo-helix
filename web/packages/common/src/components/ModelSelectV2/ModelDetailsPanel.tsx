// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { useRelativeTimeSince } from '@nemo/common/src/components/RelativeTime';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { formatFinetuningType } from '@nemo/common/src/utils/formatters';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Divider, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';

interface ModelDetailsPanelProps {
  model: ModelEntity;
  adapter?: Adapter;
}

export const ModelDetailsPanel: FC<ModelDetailsPanelProps> = ({ model, adapter }) => {
  const name = adapter?.name ?? model.name?.split('@')[0];
  const description = adapter?.description ?? model.description;
  const finetuningType = adapter?.finetuning_type ?? model.finetuning_type;
  const createdAt = adapter?.created_at ?? model.created_at;
  const updatedAt = adapter?.updated_at ?? model.updated_at;
  const isChatModel = model.spec?.is_chat;
  const headType = model.spec?.head_type;
  const isEmbeddingModel = headType === 'embedding' || Boolean(model.spec?.is_embedding_model);
  const isCrossEncoder = headType === 'cross_encoder';

  // Hooks must be called unconditionally per React rules; empty string is safe fallback
  const createdRelative = useRelativeTimeSince(createdAt ?? '');
  const updatedRelative = useRelativeTimeSince(updatedAt ?? '');

  return (
    <Stack className="p-3" gap="density-md">
      <Flex gap="density-sm" align="center">
        {creatorToIcon(model.workspace ?? '', { className: 'flex-shrink-0' })}
        <Text fontWeight="bold">{name}</Text>
      </Flex>
      {description && (
        <Text className="text-secondary leading-normal" kind="body/regular/sm">
          {description}
        </Text>
      )}
      {(isChatModel || isEmbeddingModel || isCrossEncoder) && (
        <Flex gap="density-sm" wrap="wrap">
          {isChatModel && (
            <Tag color="gray" kind="outline" readOnly>
              Chat Model
            </Tag>
          )}
          {isEmbeddingModel && (
            <Tag color="gray" kind="outline" readOnly>
              Embedding Model
            </Tag>
          )}
          {isCrossEncoder && (
            <Tag color="gray" kind="outline" readOnly>
              Cross-Encoder Model
            </Tag>
          )}
        </Flex>
      )}
      <Divider />
      <Stack gap="density-sm">
        <KVPair
          label="Fine-tuning Type"
          value={finetuningType ? formatFinetuningType(finetuningType) : undefined}
          size="narrow"
          orientation="horizontal"
        />
        <KVPair
          label="Created"
          value={createdAt ? createdRelative : undefined}
          size="narrow"
          orientation="horizontal"
        />
        <KVPair
          label="Updated"
          value={updatedAt ? updatedRelative : undefined}
          size="narrow"
          orientation="horizontal"
        />
      </Stack>
    </Stack>
  );
};
