// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledCheckbox } from '@nemo/common/src/components/form/ControlledCheckbox';
import { ControlledCombobox } from '@nemo/common/src/components/form/ControlledCombobox';
import { ControlledSegmentedControl } from '@nemo/common/src/components/form/ControlledSegmentedControl';
import { useAnonymizerListEntityLabels } from '@nemo/sdk/generated/anonymizer/anonymizer';
import { Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  defaultEntitiesLabel,
  ENTITY_MODE_CUSTOM,
  ENTITY_MODE_DESCRIPTIONS,
  ENTITY_MODE_OPTIONS,
  entityTagColor,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import {
  buildEntitySections,
  customLabelCandidate,
} from '@studio/routes/AnonymizerBuilderRoute/entityItems';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import { X } from 'lucide-react';
import { FC, useMemo, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export const EntitiesSection: FC = () => {
  const { control, setValue } = useFormContext<AnonymizerFormData>();
  const workspace = useWorkspaceFromPath();
  const entityMode = useWatch({ control, name: 'entityMode' });
  const [inputValue, setInputValue] = useState('');

  const selectedLabels = useWatch({ control, name: 'entityLabels' });

  const { data, isLoading } = useAnonymizerListEntityLabels(workspace, { query: {} });
  const available = useMemo(() => data?.data ?? [], [data?.data]);

  const isCustom = entityMode === ENTITY_MODE_CUSTOM;
  const selected = useMemo(() => selectedLabels ?? [], [selectedLabels]);
  const defaultsLabel = defaultEntitiesLabel(available.length);

  const items = useMemo(() => {
    const sections = buildEntitySections([...available]).map((section) => ({
      kind: 'section' as const,
      slotHeading: section.heading,
      items: section.items,
    }));
    const candidate = customLabelCandidate(inputValue, [...available], selected);
    return candidate
      ? [{ kind: 'section' as const, slotHeading: 'Custom label', items: [candidate] }, ...sections]
      : sections;
  }, [available, inputValue, selected]);

  const removeLabel = (label: string) =>
    setValue(
      'entityLabels',
      selected.filter((value) => value !== label),
      { shouldValidate: true }
    );

  return (
    <Stack gap="density-lg">
      <Text kind="label/bold/lg">Entities</Text>
      <ControlledSegmentedControl
        className="w-full"
        size="tiny"
        items={ENTITY_MODE_OPTIONS}
        useControllerProps={{ name: 'entityMode', control }}
      />
      <Text kind="body/regular/md">{ENTITY_MODE_DESCRIPTIONS[entityMode](defaultsLabel)}</Text>
      {isCustom && (
        <Stack gap="density-md">
          <ControlledCombobox
            kind="multiple"
            aria-label="Entity labels"
            items={items}
            inputValue={inputValue}
            onInputValueChange={setInputValue}
            onChange={() => setInputValue('')}
            placeholder="Select labels..."
            emptyStateMessage={isLoading ? 'Loading labels...' : 'No matching labels.'}
            multipleMode="count"
            formatSummaryLabel={(count) => `${count} selected`}
            useControllerProps={{ name: 'entityLabels', control }}
            formFieldProps={{
              slotLabel: 'Entity Labels',
              slotInfo:
                'Pick from the detected entity types, or type your own label and select it.',
            }}
          />
          {selected.length > 0 && (
            <Flex className="flex-wrap" gap="density-sm">
              {selected.map((label) => (
                <Tag
                  key={label}
                  color={entityTagColor(label)}
                  kind="outline"
                  aria-label={`Remove ${label}`}
                  onClick={() => removeLabel(label)}
                >
                  {label}
                  <X size={14} />
                </Tag>
              ))}
            </Flex>
          )}
        </Stack>
      )}
      {isCustom && (
        <ControlledCheckbox
          slotLabel={`Also include ${defaultsLabel}`}
          disabled={isLoading}
          useControllerProps={{ name: 'includeDefaultEntities', control }}
        />
      )}
    </Stack>
  );
};
