// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Accordion, Text } from '@nvidia/foundations-react-core';
import type { AdvancedFileSplitOptions } from '@studio/components/FilesTable/CreateFileSplitsModal/types';
import { tooltipClassName } from '@studio/styles/common';
import { type FC, useMemo } from 'react';
import { useFormContext } from 'react-hook-form';

interface Props {
  distributionType: 'random' | 'sequential';
}
/**
 * A component that allows the user to specify advanced options for the file split.
 * Supports options for random and sequential distribution.
 */
export const FileSplitAdvancedOptions: FC<Props> = ({ distributionType }) => {
  const { control } = useFormContext<AdvancedFileSplitOptions>();
  const slotContent = useMemo(() => {
    if (distributionType === 'random') {
      return (
        <ControlledTextInput
          useControllerProps={{ name: 'seed', control }}
          label="Seed Value (Optional)"
          formFieldProps={{
            attributes: {
              Popover: { className: tooltipClassName },
              FormFieldLabelGroup: {
                className: 'justify-between',
              },
            },
            slotInfo:
              'Setting a seed number guarantees that the random sampling process is reproducible, generating the exact same sample each time to ensure consistency.',
          }}
        />
      );
    } else if (distributionType === 'sequential') {
      return (
        <ControlledTextInput
          useControllerProps={{ name: 'sortKey', control }}
          label="Sort Key (Optional)"
          formFieldProps={{
            attributes: {
              Popover: { className: tooltipClassName },
              FormFieldLabelGroup: {
                className: 'justify-between',
              },
            },
            slotInfo:
              "Specify the field name to sort by before splitting the data sequentially (e.g., 'timestamp', 'date', 'id').",
          }}
        />
      );
    }
    return null;
  }, [control, distributionType]);

  return (
    <Accordion
      title="Advanced Options"
      multiple
      className="border-none bg-transparent"
      items={[
        {
          attributes: {
            AccordionItem: {
              className: 'border-none bg-transparent',
            },
            AccordionTrigger: {
              className: 'px-0',
            },
            AccordionContent: {
              className: 'px-0',
            },
          },
          value: 'advanced-options',
          slotTrigger: <Text kind="label/bold/sm">Advanced Options</Text>,
          slotContent,
          chevronPosition: 'start',
        },
      ]}
    />
  );
};
