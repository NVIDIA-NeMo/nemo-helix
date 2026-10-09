/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import {
  Block,
  Divider,
  Flex,
  SelectContent,
  SelectItem,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import {
  CUSTOM_ROW,
  getInferenceProviderPresetIcon,
  getInferenceProviderPresetLabel,
  MODEL_PROVIDER_ROWS,
  type InferenceProviderPresetId,
} from '@studio/routes/InferenceProvidersListRoute/CreateInferenceProviderSidePanel/inferenceProviderPresets';
import { Search } from 'lucide-react';
import { type ChangeEvent, type FC, useCallback, useMemo, useRef, useState } from 'react';

export interface InferenceModelProviderSelectProps {
  value: InferenceProviderPresetId;
  onValueChange: (value: InferenceProviderPresetId) => void;
  disabled?: boolean;
  isPresetDisabled: (id: Exclude<InferenceProviderPresetId, 'custom'>) => boolean;
  triggerPlaceholder?: string;
}

export const InferenceModelProviderSelect: FC<InferenceModelProviderSelectProps> = ({
  value,
  onValueChange,
  disabled,
  isPresetDisabled,
  triggerPlaceholder = 'Select a provider',
}) => {
  const [localSearch, setLocalSearch] = useState('');
  const searchInputRef = useRef<HTMLInputElement>(null);

  const searchLower = localSearch.trim().toLowerCase();

  const filteredModelRows = useMemo(() => {
    if (!searchLower) return MODEL_PROVIDER_ROWS;
    return MODEL_PROVIDER_ROWS.filter((row) => {
      const hay = `${row.label} ${row.filterText}`.toLowerCase();
      return hay.includes(searchLower);
    });
  }, [searchLower]);

  const customMatches = useMemo(() => {
    if (!searchLower) return true;
    const hay = `${CUSTOM_ROW.label} ${CUSTOM_ROW.filterText}`.toLowerCase();
    return hay.includes(searchLower);
  }, [searchLower]);

  const handleOpenChange = useCallback((isOpen: boolean) => {
    if (isOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 0);
    } else {
      setLocalSearch('');
    }
  }, []);

  const showEmpty = filteredModelRows.length === 0 && !customMatches;

  return (
    <SelectRoot
      disabled={disabled}
      value={value}
      onValueChange={(v: string) => onValueChange(v as InferenceProviderPresetId)}
      onOpenChange={handleOpenChange}
    >
      <SelectTrigger
        className="w-full border-1 nv-input"
        placeholder={triggerPlaceholder}
        renderValue={(v: string | string[] | undefined) => {
          const valueStr = Array.isArray(v) ? v[0] : v;
          if (valueStr == null || valueStr === '') {
            return null;
          }
          const id = valueStr as InferenceProviderPresetId;
          const Icon = getInferenceProviderPresetIcon(id);
          return (
            <Flex align="center" gap="1" className="min-w-0">
              <Icon aria-hidden className="size-4 shrink-0 text-base" />
              <Text kind="body/regular/md" className="truncate">
                {getInferenceProviderPresetLabel(id)}
              </Text>
            </Flex>
          );
        }}
      />
      <SelectContent className="min-w-(--radix-select-trigger-width) ">
        <Block className="bg-surface-raised sticky top-0 z-10 w-full p-2">
          <TextInput
            ref={searchInputRef}
            name="model-provider-search"
            className="overflow-hidden"
            slotStart={<Search aria-hidden className="size-4 shrink-0 opacity-70" />}
            placeholder="Search by name..."
            value={localSearch}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setLocalSearch(e.target.value)}
            attributes={{
              Input: {
                ['data-testid' as never]: 'model-provider-search',
              },
            }}
          />
        </Block>
        <Stack className="max-h-[320px] w-full overflow-auto overscroll-contain" gap="0">
          {showEmpty ? (
            <Flex align="center" justify="center" className="py-4">
              <Text kind="body/regular/sm" className="text-subtle">
                No matching providers
              </Text>
            </Flex>
          ) : (
            <>
              {filteredModelRows.length > 0 ? (
                <Stack gap="0">
                  <Block className="w-full px-3 pb-1 pt-2">
                    <Text kind="label/bold/sm" className="text-secondary-foreground">
                      Pre-configured Providers
                    </Text>
                  </Block>
                  {filteredModelRows.map(({ id, label, Icon }) => (
                    <SelectItem
                      key={id}
                      value={id}
                      disabled={isPresetDisabled(id)}
                      filterValue={`${label} ${MODEL_PROVIDER_ROWS.find((r) => r.id === id)?.filterText ?? ''}`}
                      slotStart={<Icon aria-hidden className="size-4 shrink-0 text-base" />}
                    >
                      {label}
                    </SelectItem>
                  ))}
                </Stack>
              ) : null}

              {filteredModelRows.length > 0 && customMatches ? (
                <Divider orientation="horizontal" className="my-1" />
              ) : null}

              {customMatches ? (
                <Stack gap="0">
                  <Block className="w-full px-3 pb-1 pt-2">
                    <Text kind="label/bold/sm" className="text-secondary-foreground">
                      Custom
                    </Text>
                  </Block>
                  <SelectItem
                    value={CUSTOM_ROW.id}
                    filterValue={CUSTOM_ROW.filterText}
                    slotStart={
                      <CUSTOM_ROW.Icon aria-hidden className="size-4 shrink-0 text-base" />
                    }
                  >
                    {CUSTOM_ROW.label}
                  </SelectItem>
                </Stack>
              ) : null}
            </>
          )}
        </Stack>
      </SelectContent>
    </SelectRoot>
  );
};
