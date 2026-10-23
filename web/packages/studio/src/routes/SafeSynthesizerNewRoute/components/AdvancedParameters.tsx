// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { InfoCircle, Retry } from '@nemo/icons/icons';
import {
  Text,
  Stack,
  Divider,
  Flex,
  Checkbox,
  Tooltip,
  Button,
} from '@nvidia/foundations-react-core';
import {
  DEFAULT_NUM_INPUT_RECORDS_TO_SAMPLE,
  PRETRAINED_MODEL_OPTIONS,
} from '@studio/routes/SafeSynthesizerNewRoute/constants';
import {
  SafeSynthesizerFormData,
  getSafeSynthesizerFormDefaults,
} from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { useFormContext } from 'react-hook-form';

export const AdvancedParameters = () => {
  const { control, setValue, watch, trigger } = useFormContext<SafeSynthesizerFormData>();

  // Get default values from schema for reset functionality
  const defaults = getSafeSynthesizerFormDefaults();

  // UI-only state for showing/hiding automatic sampling and scaling
  const numInputRecordsToSample = watch('spec.config.training.num_input_records_to_sample');
  const ropeScalingFactor = watch('spec.config.training.rope_scaling_factor');

  const useAutomaticSampling = numInputRecordsToSample === 'auto';
  const useAutomaticScaling = ropeScalingFactor === 'auto';

  return (
    <Stack gap="density-2xl">
      <Text kind="body/regular/md">
        Training hyperparameters control how the model learns from your tabular data. These
        parameters directly affect training performance, model quality, and resource usage.
      </Text>
      <Divider orientation="horizontal" width="small" />
      <Stack gap="density-2xl">
        <Text kind="label/bold/lg">Core Generation Settings</Text>
        <ControlledSliderWithTextInput
          useControllerProps={{
            name: 'spec.config.generation.temperature',
            control,
          }}
          displayName="temperature"
          defaultValue={defaults.spec.config.generation?.temperature ?? 1}
          min={0}
          max={2}
          step={0.01}
          formFieldProps={{
            slotLabel: (
              <>
                temperature
                <Tooltip
                  slotContent="Controls the randomness of responses. Lower values make output more focused and deterministic, while higher values increase creativity and variability."
                  side="bottom"
                >
                  <InfoCircle className="inline ml-density-sm" />
                </Tooltip>
              </>
            ),
            labelPosition: 'left',
            attributes: {
              FormFieldLabelGroup: { className: '!w-[200px]' },
              FormFieldHelper: { className: '!ml-[210px]' },
            },
          }}
          attributes={{
            Slider: {
              stepInterval: 1,
            },
            TextInput: {
              className: 'max-w-[75px]',
            },
          }}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{
            name: 'spec.config.generation.top_p',
            control,
          }}
          defaultValue={defaults.spec.config.generation?.top_p ?? 0.5}
          min={0}
          max={1}
          step={0.01}
          displayName="top_p"
          formFieldProps={{
            slotLabel: (
              <>
                top_p
                <Tooltip
                  slotContent="Controls output diversity by only considering the most likely words until their combined probability reaches this percentage, filtering out improbable options."
                  side="bottom"
                >
                  <InfoCircle className="inline ml-density-sm" />
                </Tooltip>
              </>
            ),
            labelPosition: 'left',
            attributes: {
              FormFieldLabelGroup: { className: '!w-[200px]' },
              FormFieldHelper: { className: '!ml-[210px]' },
            },
          }}
          attributes={{
            Slider: {
              stepInterval: 1,
            },
            TextInput: {
              className: 'max-w-[75px]',
            },
          }}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{
            name: 'spec.config.generation.repetition_penalty',
            control,
          }}
          defaultValue={defaults.spec.config.generation?.repetition_penalty ?? 1}
          min={1}
          max={2}
          step={0.01}
          displayName="repetition_penalty"
          formFieldProps={{
            slotLabel: (
              <>
                repetition_penalty
                <Tooltip
                  slotContent="Penalty for the model repeating the same token on generation. A value of 1.0 means no penalty. Values greater than 1.0 reduce the likelihood of repeating the same token."
                  side="bottom"
                >
                  <InfoCircle className="inline ml-density-sm" />
                </Tooltip>
              </>
            ),
            labelPosition: 'left',
            attributes: {
              FormFieldLabelGroup: { className: '!w-[200px]' },
              FormFieldHelper: { className: '!ml-[210px]' },
            },
          }}
          attributes={{
            Slider: {
              stepInterval: 1,
            },
            TextInput: {
              className: 'max-w-[75px]',
            },
          }}
        />
      </Stack>
      <Divider orientation="horizontal" width="small" />
      <Stack gap="density-2xl">
        <Text kind="label/bold/lg">Training Data Configuration</Text>
        <ControlledSelect
          useControllerProps={{
            name: 'spec.config.training.pretrained_model',
            control,
          }}
          items={PRETRAINED_MODEL_OPTIONS.map((model) => ({
            value: model,
            children: model,
          }))}
          formFieldProps={{
            slotLabel: (
              <>
                Model
                <Tooltip
                  slotContent="Model that gets fine-tuned to generate synthetic data."
                  side="bottom"
                >
                  <InfoCircle className="inline ml-density-sm" />
                </Tooltip>
              </>
            ),
            labelPosition: 'left',
            attributes: {
              FormFieldLabelGroup: { className: '!w-[225px]' },
              FormFieldHelper: { className: '!ml-[235px]' },
            },
          }}
        />
        <Stack gap="density-md">
          <ControlledTextInput
            useControllerProps={{
              name: 'spec.config.training.num_input_records_to_sample',
              control,
            }}
            label={
              <>
                num_input_records_to_sample
                <Tooltip
                  slotContent="Total number of non-unique records seen by the model. It is effectively the product of training data size and the number of epochs. If 'num_input_records_to_sample' is greater than the sample size, the model is trained on each record multiple times; otherwise, the model is trained on a subset of the records. Recommended value 10,000 or more."
                  side="bottom"
                >
                  <InfoCircle className="inline ml-density-sm" />
                </Tooltip>
              </>
            }
            type="number"
            min={1}
            step={1}
            disabled={useAutomaticSampling}
            value={
              !useAutomaticSampling
                ? typeof numInputRecordsToSample === 'number'
                  ? numInputRecordsToSample.toString()
                  : ''
                : ''
            }
            formFieldProps={{
              labelPosition: 'left',
              attributes: {
                FormFieldLabelGroup: { className: '!w-[225px]' },
                FormFieldHelper: { className: '!ml-[235px]' },
              },
            }}
            slotEnd={
              <Button
                kind="tertiary"
                size="small"
                aria-label="Reset num_input_records_to_sample to default value"
                disabled={useAutomaticSampling}
                onClick={async () => {
                  setValue(
                    'spec.config.training.num_input_records_to_sample',
                    defaults.spec.config.training?.num_input_records_to_sample ?? 'auto'
                  );
                  await trigger('spec.config.training.num_input_records_to_sample');
                }}
                className="shrink-0"
                type="button"
              >
                <Retry />
              </Button>
            }
          />
          <Flex direction="row" align="center" gap="density-md" className="w-full">
            <div className="w-[225px]" />
            <div className="flex-1">
              <Flex gap="density-sm" direction="row" align="center">
                <Checkbox
                  aria-labelledby="auto-sampling-label"
                  checked={useAutomaticSampling}
                  onCheckedChange={async (checked) => {
                    setValue(
                      'spec.config.training.num_input_records_to_sample',
                      checked ? 'auto' : DEFAULT_NUM_INPUT_RECORDS_TO_SAMPLE
                    );
                    await trigger('spec.config.training.num_input_records_to_sample');
                  }}
                />
                <span id="auto-sampling-label">Use Automatic Sampling</span>
              </Flex>
            </div>
          </Flex>
        </Stack>
      </Stack>
      <Divider orientation="horizontal" width="small" />
      <Stack gap="density-2xl">
        <Text kind="label/bold/lg">Context Length Scaling</Text>
        <ControlledSliderWithTextInput
          useControllerProps={{
            name: 'spec.config.training.rope_scaling_factor',
            control,
          }}
          defaultValue={typeof ropeScalingFactor === 'number' ? ropeScalingFactor : 1}
          min={1}
          max={6}
          step={1}
          disabled={useAutomaticScaling}
          displayName="rope_scaling_factor"
          formFieldProps={{
            slotLabel: (
              <>
                rope_scaling_factor
                <Tooltip
                  slotContent="Scaling factor for the model's context window; an integer >=1. 1 means no additional scaling of the model's context window. Lower is better for quality, but higher may be required if your records (or groups of records, in case of event-driven data) are too large to fit in the original context window. Higher values require more GPU RAM, so reduce if hitting OOM errors. Up to 6 typically works and higher values may be possible with large GPUs."
                  side="bottom"
                >
                  <InfoCircle className="inline ml-density-sm" />
                </Tooltip>
              </>
            ),
            labelPosition: 'left',
            attributes: {
              FormFieldLabelGroup: { className: '!w-[200px]' },
              FormFieldHelper: { className: '!ml-[210px]' },
            },
          }}
          attributes={{
            Slider: {
              stepInterval: 6,
            },
            TextInput: {
              className: 'max-w-[75px]',
            },
          }}
          showReset={false}
          slotEnd={
            <Button
              kind="tertiary"
              size="small"
              aria-label="Reset Context Length Scaling to default value"
              disabled={ropeScalingFactor === 'auto'}
              onClick={() =>
                setValue(
                  'spec.config.training.rope_scaling_factor',
                  defaults.spec.config.training?.rope_scaling_factor ?? 'auto'
                )
              }
              className="shrink-0"
              type="button"
            >
              <Retry />
            </Button>
          }
        />
        <Flex direction="row" align="center" gap="density-md" className="w-full">
          <div className="min-w-[200px]" />
          <div className="flex-1">
            <Flex gap="density-sm" direction="row" align="center">
              <Checkbox
                aria-labelledby="auto-scaling-label"
                checked={useAutomaticScaling}
                onCheckedChange={(checked) => {
                  if (checked) {
                    setValue('spec.config.training.rope_scaling_factor', 'auto');
                  } else {
                    setValue('spec.config.training.rope_scaling_factor', 1);
                  }
                }}
              />
              <span id="auto-scaling-label">Use Automatic Scaling</span>
            </Flex>
          </div>
        </Flex>
      </Stack>
      <Divider orientation="horizontal" width="small" />
      <Stack gap="density-2xl">
        <Text kind="label/bold/lg">PII Replacement Configuration</Text>
        <Flex direction="row" align="center" gap="density-md" className="w-full">
          <Text kind="label/regular/md" className="min-w-[200px]">
            enable_replace_pii
            <Tooltip
              slotContent="Automatically redact or replace Personally Identifiable Information (PII) prior to training the model. Highly recommended to ensure the model has no chance to learn this sensitive information."
              side="bottom"
            >
              <InfoCircle className="inline ml-density-sm" />
            </Tooltip>
          </Text>
          <div className="flex-1">
            <ControlledSwitch
              useControllerProps={{
                name: 'spec.config.enable_replace_pii',
                control,
              }}
            />
          </div>
        </Flex>
      </Stack>
    </Stack>
  );
};
