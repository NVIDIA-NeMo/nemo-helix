// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import type { ProgressReportingConfig } from '@nemo/sdk/generated/customizer/schema';
import { ControlledStringListInput } from '@studio/components/NewCustomizationForm/ControlledStringListInput';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { specSliderProps } from '@studio/util/forms/specDefaults';
import { useFormContext } from 'react-hook-form';

/** Every parent that carries a `progress_reporting` object. */
type ProgressReportingParent = 'automodel.schedule' | 'unsloth.schedule' | 'rl.training';

interface ProgressReportingFieldsProps {
  /** Parent the `progress_reporting` object hangs off. */
  prefix: ProgressReportingParent;
  /** Flattened spec table for the backend. */
  defaults: ReadonlyMap<string, unknown>;
  defaultsPrefix: string;
  disabled: boolean;
}

/**
 * The shared `ProgressReportingConfig`, which every backend carries but hangs off a
 * different parent — `schedule` on automodel and unsloth, `training` on RL.
 */
export const ProgressReportingFields = ({
  prefix,
  defaults,
  defaultsPrefix,
  disabled,
}: ProgressReportingFieldsProps) => {
  const { control } = useFormContext<CustomizationFormFields>();
  const field = <N extends keyof ProgressReportingConfig & string>(name: N) =>
    `${prefix}.progress_reporting.${name}` as const;

  return (
    <>
      <ControlledSliderWithTextInput
        useControllerProps={{ name: field('min_report_interval_seconds'), control }}
        formFieldProps={{
          slotLabel: 'Min Report Interval (s)',
          slotInfo:
            'Least time between progress reports. Metrics are still recorded at full resolution, so raising this only makes charts update less often while leaving training less time blocked on reporting.',
        }}
        {...specSliderProps(defaults, `${defaultsPrefix}_min_report_interval_seconds`)}
        min={1}
        max={600}
        step={1}
        disabled={disabled}
      />
      <ControlledStringListInput
        useControllerProps={{ name: field('time_series_metrics'), control }}
        formFieldProps={{
          slotLabel: 'Time Series Metrics',
          slotInfo:
            'Metric names to keep history for, matching the job status details and so prefixed by phase. Glob patterns work, so *_loss covers every loss. Unmatched metrics are still reported as a current value, just without history.',
        }}
        placeholder="train_loss, val_loss"
        disabled={disabled}
      />
    </>
  );
};
