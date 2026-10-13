// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { VariableDef } from '@nemo/common/src/components/form/VariableTextArea';
import { Button, Popover, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { Plus } from 'lucide-react';
import { type ComponentProps, useState } from 'react';

export interface VariableButtonProps {
  variables: VariableDef[];
  onSelect: (variable: VariableDef) => void;
  disabled?: boolean;
  className?: string;
  attributes?: {
    Button?: ComponentProps<typeof Button>;
    Popover?: Partial<ComponentProps<typeof Popover>>;
  };
}

/** Offers the variables a template may reference, inserting the chosen one. */
export function VariableButton({
  variables,
  onSelect,
  disabled,
  className,
  attributes,
}: VariableButtonProps) {
  const empty = variables.length === 0;
  const isDisabled = Boolean(disabled) || empty;
  const [open, setOpen] = useState(false);

  const trigger = (
    <Button
      type="button"
      kind="tertiary"
      size="tiny"
      disabled={isDisabled}
      className={className}
      {...attributes?.Button}
    >
      <Plus className="size-3" aria-hidden />
      Variable
    </Button>
  );

  const triggerWithTooltip = empty ? (
    <Tooltip slotContent="No variables available" side="top">
      {trigger}
    </Tooltip>
  ) : (
    trigger
  );

  if (isDisabled) {
    return triggerWithTooltip;
  }

  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      align="start"
      slotContent={
        <Stack gap="density-xxs" className="w-[320px] p-density-sm">
          {variables.map((v) => (
            <Button
              key={v.name}
              type="button"
              kind="tertiary"
              className="w-full justify-start"
              onClick={() => {
                onSelect(v);
                setOpen(false);
              }}
            >
              <Stack gap="density-xxs" className="min-w-0 items-start text-left">
                <Text kind="label/regular/md">{v.name}</Text>
                {v.description ? (
                  <Text kind="body/regular/sm" className="text-secondary">
                    {v.description}
                  </Text>
                ) : null}
              </Stack>
            </Button>
          ))}
        </Stack>
      }
      {...attributes?.Popover}
    >
      {triggerWithTooltip}
    </Popover>
  );
}
