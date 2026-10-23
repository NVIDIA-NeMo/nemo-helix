// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { formatFinetuningType } from '@nemo/common/src/utils/formatters';
import type { Adapter } from '@nemo/sdk/generated/platform/schema';
import { Badge, Card, Flex, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import { DetailSection } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/DetailSection';
import { getModelDetailRoute } from '@studio/routes/utils';
import type { FC, ReactNode } from 'react';
import { Link } from 'react-router';

export interface AdaptersSectionProps {
  workspace: string;
  modelName: string;
  adapters: Adapter[];
  description?: string;
  /** A badge shown under each adapter's name, such as whether a deployment has it loaded. */
  renderStatus?: (adapter: Adapter) => ReactNode;
}

export const AdaptersSection: FC<AdaptersSectionProps> = ({
  workspace,
  modelName,
  adapters,
  description = 'Adapters trained on top of this model.',
  renderStatus,
}) => (
  <DetailSection heading={`Adapters (${adapters.length})`} description={description}>
    <Grid colMinWidth="260px" gap="density-lg">
      {adapters.map((adapter) => {
        const summary = [
          adapter.lora_config ? `Rank ${adapter.lora_config.rank}` : '',
          adapter.created_at ? formatAbsoluteTimestamp(adapter.created_at) : '',
        ].filter(Boolean);
        return (
          <Card key={adapter.name} asChild interactive>
            <Link to={getModelDetailRoute(workspace, modelName, { adapterName: adapter.name })}>
              <Stack gap="density-sm" className="min-w-0">
                <Flex align="center" justify="between" gap="density-sm">
                  <Text kind="body/semibold/md" className="truncate" title={adapter.name}>
                    {adapter.name}
                  </Text>
                  <Badge kind="outline" color="gray">
                    {formatFinetuningType(adapter.finetuning_type)}
                  </Badge>
                </Flex>
                {summary.length > 0 && (
                  <Text kind="body/regular/sm" className="text-secondary">
                    {summary.join(' · ')}
                  </Text>
                )}
                {renderStatus && <Flex>{renderStatus(adapter)}</Flex>}
                {adapter.enabled === false && (
                  <Text kind="body/regular/sm" className="text-secondary">
                    Disabled
                  </Text>
                )}
              </Stack>
            </Link>
          </Card>
        );
      })}
    </Grid>
  </DetailSection>
);
