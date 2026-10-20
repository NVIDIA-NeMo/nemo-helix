// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile } from '@nemo/common/src/components/StatTile';
import { Divider, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import {
  type DetailField,
  type DetailTile,
  hasDetailValue,
} from '@studio/routes/ModelDetailRoute/DetailsTab/fields';
import { FieldGrid } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/FieldGrid';
import type { FC } from 'react';

export interface DetailPanelProps {
  heading: string;
  description?: string;
  tiles: DetailTile[];
  fields: DetailField[];
}

/** A raised panel: headline numbers as stat tiles above the remaining fields. */
export const DetailPanel: FC<DetailPanelProps> = ({ heading, description, tiles, fields }) => {
  const hasFields = fields.some((field) => hasDetailValue(field.value));
  return (
    <Panel elevation="high" className="h-full">
      <Stack gap="density-xl">
        <Stack gap="density-xs">
          <Text kind="label/bold/lg" asChild>
            <h2>{heading}</h2>
          </Text>
          {description && (
            <Text kind="body/regular/sm" className="text-secondary">
              {description}
            </Text>
          )}
        </Stack>
        {tiles.length > 0 && (
          <Flex gap="density-3xl" wrap="wrap">
            {tiles.map((tile) => (
              <StatTile key={tile.label} {...tile} bordered={false} />
            ))}
          </Flex>
        )}
        {tiles.length > 0 && hasFields && <Divider />}
        {hasFields && <FieldGrid fields={fields} />}
      </Stack>
    </Panel>
  );
};
