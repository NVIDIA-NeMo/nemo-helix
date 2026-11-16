// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import type { Control, FieldValues, Path } from 'react-hook-form';

export interface JobBasicsProps<T extends FieldValues = FieldValues> {
  control: Control<T>;
  nameName: Path<T>;
  rowsName: Path<T>;
  descriptionName: Path<T>;
  disabled?: boolean;
}

/**
 * "Job basics" card: name the dataset, set the full-run record count, and describe the job.
 */
export function JobBasics<T extends FieldValues>({
  control,
  nameName,
  rowsName,
  descriptionName,
  disabled = false,
}: JobBasicsProps<T>) {
  return (
    <Panel elevation="high" density="standard">
      <Stack gap="density-lg">
        <Stack gap="density-xs">
          <Text kind="label/bold/lg">Job basics</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            Name your fileset and set the full-run size.
          </Text>
        </Stack>

        <Flex gap="density-md" align="start" className="w-full">
          <Stack className="min-w-0 flex-1">
            <ControlledTextInput
              label="Fileset name"
              disabled={disabled}
              useControllerProps={{ name: nameName, control }}
            />
          </Stack>
          <Stack className="w-[200px] shrink-0">
            <ControlledTextInput
              label="Records to generate"
              type="number"
              min={1}
              step={1}
              required
              disabled={disabled}
              useControllerProps={{ name: rowsName, control }}
            />
          </Stack>
        </Flex>

        <ControlledTextArea
          label="Description"
          rows={3}
          placeholder="What this fileset is for and how it will be used…"
          className="w-full"
          disabled={disabled}
          useControllerProps={{ name: descriptionName, control }}
        />
      </Stack>
    </Panel>
  );
}
