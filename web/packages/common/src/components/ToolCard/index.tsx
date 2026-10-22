// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EntityCard } from '@nemo/common/src/components/EntityCard';
import type { ToolParameter } from '@nemo/common/src/components/ToolCard/types';
import type { ChatCompletionTool } from '@nemo/sdk/generated/platform/schema';
import { Badge } from '@nvidia/foundations-react-core';
import { Asterisk, Braces, Wrench } from 'lucide-react';
import type { FC } from 'react';

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

const parameterTitle = ({ type, description, required }: ToolParameter): string =>
  [type, required ? 'required' : undefined, description].filter(Boolean).join(' · ');

interface ToolCardProps {
  tool: ChatCompletionTool;
  onClick?: () => void;
  className?: string;
}

const STAT_ICON = 'w-4 h-4 text-secondary';

/**
 * Summarizes an OpenAI-compatible tool definition in the same layout as model cards:
 * name, type, description, strict mode, and its JSON Schema parameters.
 */
export const ToolCard: FC<ToolCardProps> = ({ tool, onClick, className }) => {
  const { name, description, parameters, strict } = tool.function;
  const params = getToolParameters(parameters);
  const requiredCount = params.filter((p) => p.required).length;

  return (
    <EntityCard
      title={name}
      titleKind="mono/md"
      subtitle={{
        icon: <Wrench className="w-4 h-4 flex-shrink-0" aria-hidden />,
        label: tool.type === 'function' ? 'Function' : tool.type,
        trailing: strict && (
          <Badge color="green" kind="solid" size="small">
            Strict
          </Badge>
        ),
      }}
      description={description}
      tags={params.map((p) => ({ key: p.name, label: p.name, title: parameterTitle(p) }))}
      stats={[
        {
          key: 'parameters',
          icon: <Braces className={STAT_ICON} />,
          label: `${params.length} ${params.length === 1 ? 'param' : 'params'}`,
        },
        ...(requiredCount > 0
          ? [
              {
                key: 'required',
                icon: <Asterisk className={STAT_ICON} />,
                label: `${requiredCount} required`,
              },
            ]
          : []),
      ]}
      onClick={onClick}
      className={className}
    />
  );
};
