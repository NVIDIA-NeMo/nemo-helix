// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  InferenceParameters,
  type InferenceParametersProps,
} from '@nemo/common/src/components/ModelSelectV2/InferenceParameters';
import { ChevronDown, Sliders } from '@nemo/icons/icons';
import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  DropdownContent,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  Stack,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export interface ParamsDropdownProps {
  disabled?: boolean;
  /** Controlled open state. Leave both unset to let the dropdown manage it. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  inferenceParams?: Partial<InferenceParams>;
  onInferenceParamsChange?: (params: Partial<InferenceParams>) => void;
  fieldMetadata?: InferenceParametersProps['fieldMetadata'];
  fields?: InferenceParametersProps['fields'];
  /** Show only the icon, with the name in a tooltip, for tight layouts. */
  iconOnly?: boolean;
}

export const ParamsDropdown: FC<ParamsDropdownProps> = ({
  disabled = false,
  open,
  onOpenChange,
  inferenceParams = {},
  onInferenceParamsChange,
  fieldMetadata,
  fields,
  iconOnly = false,
}) => {
  const trigger = (
    <DropdownTrigger asChild showChevron={false}>
      <Button
        kind="secondary"
        disabled={disabled}
        aria-label="Model parameters"
        data-testid="params-dropdown-trigger"
        className="shrink-0"
      >
        {iconOnly ? (
          <Sliders size="16" />
        ) : (
          <Flex align="center" gap="density-sm">
            <Sliders size="16" />
            <Text>Params</Text>
            <ChevronDown size="16" />
          </Flex>
        )}
      </Button>
    </DropdownTrigger>
  );

  return (
    <DropdownRoot open={open} onOpenChange={onOpenChange}>
      {iconOnly ? <Tooltip slotContent="Model parameters">{trigger}</Tooltip> : trigger}
      <DropdownContent
        align="end"
        side="bottom"
        className="w-[250px] max-h-[600px] overflow-y-auto p-4"
        data-testid="params-dropdown-content"
      >
        <Stack gap="4">
          <Text kind="body/semibold/md" className="text-secondary">
            Inference Params
          </Text>
          <InferenceParameters
            value={inferenceParams}
            onChange={onInferenceParamsChange ?? (() => {})}
            disabled={disabled}
            fieldMetadata={fieldMetadata}
            fields={fields}
          />
        </Stack>
      </DropdownContent>
    </DropdownRoot>
  );
};
