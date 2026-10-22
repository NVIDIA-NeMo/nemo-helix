// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { ScaleBalance } from '@nvidia/react-gui-icons/icons';
import { type FC, type ReactNode } from 'react';

const METRIC_TYPE_LABELS: Record<string, string> = {
  'llm-judge': 'LLM Judge',
};

const resolveTypeLabel = (type: string): string =>
  METRIC_TYPE_LABELS[type] ??
  type
    .split('-')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');

interface EvalCardProps {
  name: string;
  description?: string;
  type?: string | null;
  /** Icon rendered inside the accent box. Defaults to a ScaleBalance icon. */
  icon?: ReactNode;
}

export const EvalCard: FC<EvalCardProps> = ({ name, description, type, icon }) => {
  const typeLabel = type ? resolveTypeLabel(type) : null;

  return (
    <Stack gap="2" className="border border-base rounded-lg p-4">
      <Flex align="center" gap="2" className="w-full">
        <Flex
          align="center"
          justify="center"
          className="size-6 rounded shrink-0 bg-accent-green text-accent-green"
        >
          {icon ?? <ScaleBalance size="12" />}
        </Flex>
        <Text kind="label/semibold/xl" className="flex-1 truncate min-w-0">
          {name}
        </Text>
        {typeLabel && (
          <Badge kind="solid" color="gray" className="shrink-0">
            <ScaleBalance size="10" />
            {typeLabel}
          </Badge>
        )}
      </Flex>
      {description && (
        <Text kind="body/regular/md" className="text-secondary truncate pl-8">
          {description}
        </Text>
      )}
    </Stack>
  );
};
