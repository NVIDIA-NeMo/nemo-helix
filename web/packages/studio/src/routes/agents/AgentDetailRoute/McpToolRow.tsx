// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
import { Badge, Flex, Grid, GridItem, Text } from '@nvidia/foundations-react-core';
import type { McpToolStatus } from '@studio/api/agents/types';
import cn from 'classnames';
import type { FC, ReactNode } from 'react';

/** About two lines of the Tools pane before "Show more". */
const DESCRIPTION_CHARACTER_LIMIT = 90;

interface McpToolRowProps {
  tool: McpToolStatus;
  /** Leading control column, e.g. a switch to enable the tool in chat. */
  slotControl?: ReactNode;
}

/** One MCP tool in the Tools pane, laid out like a guardrail rail row. */
export const McpToolRow: FC<McpToolRowProps> = ({ tool, slotControl }) => {
  const column = slotControl ? 2 : 1;

  return (
    <Grid
      role="listitem"
      className={cn(
        'border-border-subtle items-start gap-x-density-lg gap-y-density-xs border-b py-density-sm last:border-b-0',
        slotControl ? 'grid-cols-[auto_1fr]' : 'grid-cols-1'
      )}
    >
      {slotControl ? (
        <GridItem colStart={1} rowStart={1} className="flex self-center">
          {slotControl}
        </GridItem>
      ) : null}

      <GridItem colStart={column} rowStart={1} className="min-w-0">
        <Flex align="center" justify="between" gap="density-sm">
          <Text
            kind="body/semibold/sm"
            className={cn('truncate font-mono', !tool.exposed && 'text-secondary')}
            title={tool.name}
          >
            {tool.name}
          </Text>
          {tool.exposed ? null : (
            <Badge kind="outline" color="gray" size="small" className="shrink-0">
              Blocked
            </Badge>
          )}
        </Flex>
      </GridItem>

      {tool.description ? (
        <GridItem colStart={column} rowStart={2} className="min-w-0">
          <ExpandableMessage
            message={tool.description}
            characterLimit={DESCRIPTION_CHARACTER_LIMIT}
            attributes={{
              Text: {
                kind: 'body/regular/sm',
                className: 'text-secondary [overflow-wrap:anywhere]',
              },
              Anchor: { className: 'text-sm w-fit' },
            }}
          />
        </GridItem>
      ) : null}
    </Grid>
  );
};
