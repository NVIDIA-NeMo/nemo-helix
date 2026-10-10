// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import { Badge, Button, Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { SwitchyardFormValues } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/formValues';
import { ArrowDown, ArrowUp, X } from 'lucide-react';
import { type FC } from 'react';
import { useController } from 'react-hook-form';

export interface ModelOrderFieldProps {
  workspace: string;
  disabled?: boolean;
}

const move = (models: string[], from: number, to: number): string[] => {
  const next = [...models];
  [next[from], next[to]] = [next[to], next[from]];
  return next;
};

const endLabel = (index: number, count: number): string | undefined => {
  if (count < 2) return undefined;
  if (index === 0) return 'Most capable';
  if (index === count - 1) return 'Most efficient';
  return undefined;
};

/**
 * The form's `models`, in the order the router pairs them: every model is the capable side of a
 * pair with each model below it. Order is the whole input here, so it is edited directly rather
 * than inferred from the models themselves.
 */
export const ModelOrderField: FC<ModelOrderFieldProps> = ({ workspace, disabled }) => {
  const {
    field: { value, onChange },
  } = useController<SwitchyardFormValues, 'models'>({ name: 'models' });

  return (
    <Stack gap="density-md" className="w-full">
      {value.length > 0 && (
        <ol
          aria-label="Models, most capable first"
          className="divide-base bg-surface-raised border-base w-full divide-y overflow-hidden rounded-lg border"
        >
          {value.map((model, index) => {
            const label = endLabel(index, value.length);
            return (
              <li key={model} className="flex min-h-12 items-center gap-3 px-3 py-1.5">
                <span
                  aria-hidden
                  className="bg-surface-sunken text-secondary flex size-6 shrink-0 items-center justify-center rounded-full text-xs"
                >
                  {index + 1}
                </span>
                <Flex align="center" gap="density-sm" className="min-w-0 flex-1">
                  <Text kind="body/regular/md" className="min-w-0 truncate" title={model}>
                    {model}
                  </Text>
                  {label && (
                    <Badge color="gray" kind="outline" className="shrink-0">
                      {label}
                    </Badge>
                  )}
                </Flex>
                <Flex align="center" gap="density-xxs" className="shrink-0">
                  <Button
                    kind="tertiary"
                    size="small"
                    disabled={disabled || index === 0}
                    aria-label={`Move ${model} up`}
                    onClick={() => onChange(move(value, index, index - 1))}
                  >
                    <ArrowUp size={16} />
                  </Button>
                  <Button
                    kind="tertiary"
                    size="small"
                    disabled={disabled || index === value.length - 1}
                    aria-label={`Move ${model} down`}
                    onClick={() => onChange(move(value, index, index + 1))}
                  >
                    <ArrowDown size={16} />
                  </Button>
                  <Divider orientation="vertical" className="mx-1 h-5" />
                  <Button
                    kind="tertiary"
                    size="small"
                    disabled={disabled}
                    aria-label={`Remove ${model}`}
                    onClick={() => onChange(value.filter((_, other) => other !== index))}
                  >
                    <X size={16} />
                  </Button>
                </Flex>
              </li>
            );
          })}
        </ol>
      )}

      <WorkspaceModelSelect
        workspace={workspace}
        value={null}
        onValueChange={({ model }) => {
          if (!value.includes(model)) onChange([...value, model]);
        }}
        include={(model) => !value.includes(getURNFromNamedEntityRef(model) ?? '')}
        placeholder={
          value.length === 0
            ? 'Add the most capable model'
            : value.length === 1
              ? 'Add a more efficient model'
              : 'Add another model'
        }
        disabled={disabled}
        hideAdapters
        fullWidth
        aria-label="Add a model"
      />
    </Stack>
  );
};
