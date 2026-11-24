// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { InfoCircle } from '@nemo/icons/icons';
import { useModelsListProviders } from '@nemo/sdk/generated/platform/model-providers';
import { Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import type { SafeSynthesizerFormData } from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const FIELD_LAYOUT = {
  labelPosition: 'left',
  attributes: {
    FormFieldLabelGroup: { className: '!w-[225px]' },
    FormFieldHelper: { className: '!ml-[235px]' },
  },
} as const;

const PiiDiscoveryModelFields = () => {
  const { control } = useFormContext<SafeSynthesizerFormData>();
  const workspace = useWorkspaceFromPath();
  const { data: providersPage, isLoading } = useModelsListProviders(workspace, {
    page_size: DEFAULT_LARGE_PAGE_SIZE,
  });

  const providerItems = useMemo(
    () =>
      (providersPage?.data ?? []).map((provider) => {
        const reference = getEntityReference(provider);
        return { value: reference, children: reference };
      }),
    [providersPage?.data]
  );
  const noProviders = !isLoading && providerItems.length === 0;

  return (
    <>
      <ControlledSelect
        useControllerProps={{
          name: 'spec.pii_model_provider',
          control,
        }}
        items={providerItems}
        loading={isLoading}
        placeholder="Select a model provider"
        formFieldProps={{
          ...FIELD_LAYOUT,
          slotLabel: (
            <>
              Model provider
              <Tooltip
                slotContent="Inference Gateway model provider that serves the LLM used to find PII in your data."
                side="bottom"
              >
                <InfoCircle className="inline ml-density-sm" />
              </Tooltip>
            </>
          ),
          slotHelp: noProviders
            ? 'No model providers in this workspace. Add one under Inference Providers.'
            : undefined,
        }}
      />
      <ControlledTextInput
        useControllerProps={{
          name: 'spec.config.replace_pii.llm.model_id',
          control,
        }}
        label={
          <>
            Model ID
            <Tooltip
              slotContent="Model the selected provider serves, used to find PII in your data."
              side="bottom"
            >
              <InfoCircle className="inline ml-density-sm" />
            </Tooltip>
          </>
        }
        formFieldProps={FIELD_LAYOUT}
      />
    </>
  );
};

export const PiiReplacementSettings = () => {
  const { control } = useFormContext<SafeSynthesizerFormData>();
  const enableReplacePii = useWatch({ control, name: 'spec.config.enable_replace_pii' });

  return (
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
      {enableReplacePii ? <PiiDiscoveryModelFields /> : null}
    </Stack>
  );
};
