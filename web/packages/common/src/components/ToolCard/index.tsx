// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import type { ChatCompletionTool } from '@nemo/sdk/generated/platform/schema';
import { Badge, Card, Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { Wrench } from 'lucide-react';
import type { FC } from 'react';

export interface ToolParameter {
  readonly name: string;
  readonly type?: string;
  readonly description?: string;
  readonly required: boolean;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

const schemaType = (schema: Record<string, unknown>): string | undefined => {
  const { type } = schema;
  if (Array.isArray(type)) {
    return type.filter((t) => typeof t === 'string').join(' | ') || undefined;
  }
  return asString(type);
};

/** Flattens a function's top-level JSON Schema `properties` into display rows. */
export const getToolParameters = (parameters: unknown): ToolParameter[] => {
  if (!isRecord(parameters) || !isRecord(parameters.properties)) {
    return [];
  }
  const required = new Set(
    Array.isArray(parameters.required)
      ? parameters.required.filter((r) => typeof r === 'string')
      : []
  );
  return Object.entries(parameters.properties).map(([name, schema]) => ({
    name,
    type: isRecord(schema) ? schemaType(schema) : undefined,
    description: isRecord(schema) ? asString(schema.description) : undefined,
    required: required.has(name),
  }));
};

const ParameterRow: FC<{ parameter: ToolParameter }> = ({ parameter }) => (
  <Stack gap="density-xs" className="rounded-md bg-surface-sunken p-density-md">
    <Flex align="center" gap="density-sm" wrap="wrap">
      <Text kind="mono/sm">{parameter.name}</Text>
      {parameter.type && (
        <Badge kind="outline" color="gray" size="small">
          {parameter.type}
        </Badge>
      )}
      {parameter.required && (
        <Badge kind="outline" color="red" size="small">
          required
        </Badge>
      )}
    </Flex>
    {parameter.description && (
      <Text kind="body/regular/sm" className="text-secondary">
        {parameter.description}
      </Text>
    )}
  </Stack>
);

interface ToolCardProps {
  tool: ChatCompletionTool;
  className?: string;
}

/**
 * Summarizes an OpenAI-compatible tool definition: name, description,
 * parameters from its JSON Schema, and strict mode.
 */
export const ToolCard: FC<ToolCardProps> = ({ tool, className }) => {
  const { name, description, parameters, strict } = tool.function;
  const params = getToolParameters(parameters);
  const requiredCount = params.filter((p) => p.required).length;

  return (
    <Card className={cn('h-fit', className)}>
      <Stack gap="density-lg">
        <Flex align="center" gap="density-md">
          <Flex
            align="center"
            justify="center"
            className="shrink-0 rounded-md bg-surface-sunken p-density-sm"
          >
            <Wrench width={20} height={20} aria-hidden />
          </Flex>
          <Text kind="mono/md" className="min-w-0 flex-1 truncate">
            {name}
          </Text>
          <Badge kind="outline" color="green" size="small">
            {tool.type === 'function' ? 'Function' : tool.type}
          </Badge>
        </Flex>
        {description && <Text kind="body/regular/md">{description}</Text>}
        <Divider />
        <Stack gap="density-sm">
          <Text kind="label/bold/sm" className="text-secondary">
            Parameters
          </Text>
          {params.length > 0 ? (
            params.map((p) => <ParameterRow key={p.name} parameter={p} />)
          ) : (
            <Text kind="body/regular/sm" className="text-secondary">
              No parameters
            </Text>
          )}
        </Stack>
        <Divider />
        <Stack gap="density-sm">
          <Text kind="label/bold/sm" className="text-secondary">
            Metadata
          </Text>
          <KeyValueGrid
            items={[
              {
                key: 'strict',
                label: 'Strict',
                value: strict === undefined ? '—' : String(strict),
              },
              {
                key: 'parameters',
                label: 'Parameters',
                value: `${params.length} (${requiredCount} required)`,
              },
            ]}
          />
        </Stack>
      </Stack>
    </Card>
  );
};
