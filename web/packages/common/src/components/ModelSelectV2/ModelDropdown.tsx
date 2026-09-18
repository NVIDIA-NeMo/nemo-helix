// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelDropdownList } from '@nemo/common/src/components/ModelSelectV2/ModelDropdownList';
import { ModelDropdownSearch } from '@nemo/common/src/components/ModelSelectV2/ModelDropdownSearch';
import type {
  ModelSelectV2Props,
  ModelSelection,
  ModelType,
} from '@nemo/common/src/components/ModelSelectV2/types';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { getPartsFromReference, getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import { filterModel, isBaseModel } from '@nemo/common/src/utils/models';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  DropdownContent,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  SegmentedControl,
  Text,
} from '@nvidia/foundations-react-core';
import { ChevronDown, LoaderCircle } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState, type FC } from 'react';
import { useDebounce } from 'use-debounce';

const MODEL_TYPE_ITEMS = [
  { value: 'custom', children: 'Custom Models' },
  { value: 'base', children: 'Base Models' },
];

const DEFAULT_SEARCH_DEBOUNCE_MS = 300;

const isCustomModel = (model: ModelEntity): boolean => !isBaseModel(model);

type ModelDropdownProps = Omit<
  ModelSelectV2Props,
  'showParams' | 'inferenceParams' | 'onInferenceParamsChange' | 'onOpenChange' | 'aria-label'
> & {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export const ModelDropdown: FC<ModelDropdownProps> = ({
  value,
  onValueChange,
  groups,
  onSearchChange,
  searchDebounceMs = DEFAULT_SEARCH_DEBOUNCE_MS,
  onLoadMore,
  hasMore = false,
  isLoadingMore = false,
  doneLoadingMessage,
  emptyMessage,
  loading = false,
  disabled = false,
  placeholder = 'Select a model',
  showModelTypeToggle = false,
  defaultModelType = 'custom',
  onModelTypeChange,
  hideAdapters = false,
  fullWidth = false,
  dropdownSide = 'bottom',
  open,
  onOpenChange,
}) => {
  const [search, setSearch] = useState('');
  const [modelType, setModelType] = useState<ModelType>(defaultModelType);
  const [debouncedSearch] = useDebounce(search, searchDebounceMs);

  useEffect(() => {
    onSearchChange?.(debouncedSearch);
  }, [debouncedSearch, onSearchChange]);

  const localGroups = useMemo(() => groups ?? [], [groups]);

  const selectedModel = useMemo(() => {
    if (!value) return undefined;
    if (value.entity) return value.entity;
    return localGroups
      .flatMap((g) => g.models)
      .find((m) => getURNFromNamedEntityRef(m) === value.model);
  }, [localGroups, value]);

  const filteredGroups = useMemo(() => {
    const filterType = showModelTypeToggle && !onModelTypeChange;
    const filterSearch = !onSearchChange && debouncedSearch.length > 0;
    if (!filterType && !filterSearch) return localGroups;

    return localGroups
      .map((group) => {
        let models = group.models;
        if (filterType) {
          models = modelType === 'base' ? models.filter(isBaseModel) : models.filter(isCustomModel);
        }
        if (filterSearch) {
          models = models.filter((m) => filterModel(m, debouncedSearch));
        }
        return { ...group, models };
      })
      .filter((group) => group.models.length > 0);
  }, [
    debouncedSearch,
    localGroups,
    modelType,
    onModelTypeChange,
    onSearchChange,
    showModelTypeToggle,
  ]);

  const handleSelect = useCallback(
    (selection: ModelSelection) => {
      onValueChange(selection);
      onOpenChange(false);
    },
    [onOpenChange, onValueChange]
  );

  const handleModelTypeChange = (val: string) => {
    setModelType(val as ModelType);
    onModelTypeChange?.(val as ModelType);
  };

  const handleOpenChange = (nextOpen: boolean) => {
    onOpenChange(nextOpen);
    if (!nextOpen) {
      setSearch('');
    }
  };

  const selectedParts = value?.model ? getPartsFromReference(value.model) : undefined;
  const selectedName = selectedModel?.name ?? (selectedParts?.name || value?.model);
  const selectedWorkspace =
    selectedModel?.workspace ?? (selectedParts?.name ? selectedParts.workspace : undefined);
  const selectedBaseName = selectedName ? (selectedName.split('@')[0] ?? selectedName) : undefined;
  const triggerLabel =
    selectedBaseName && value?.adapter
      ? `${selectedBaseName} / ${value.adapter}`
      : (selectedBaseName ?? placeholder);

  return (
    <DropdownRoot open={open} onOpenChange={handleOpenChange}>
      <DropdownTrigger
        asChild
        showChevron={false}
        className={fullWidth ? 'flex-1 w-full min-w-0' : undefined}
      >
        <Button
          kind="secondary"
          disabled={disabled}
          aria-label="Select a model"
          data-testid="model-select-v2-trigger"
          className="overflow-hidden data-[state=open]:border-(--border-color-feedback-success) data-[state=open]:bg-(--background-color-interaction-base)"
        >
          <Flex
            align="center"
            gap="density-sm"
            className={`min-w-0 ${fullWidth ? 'w-full justify-between' : ''}`}
          >
            <Flex align="center" gap="density-sm" className="min-w-0 flex-1">
              {selectedWorkspace &&
                creatorToIcon(selectedWorkspace, { className: 'text-base flex-shrink-0' })}
              {loading && !selectedName ? (
                <>
                  <LoaderCircle size={16} className="animate-spin shrink-0" />
                  <Text className="truncate">{placeholder}</Text>
                </>
              ) : (
                <Text className="truncate">{triggerLabel}</Text>
              )}
            </Flex>
            <ChevronDown size={16} className="shrink-0" />
          </Flex>
        </Button>
      </DropdownTrigger>
      <DropdownContent
        align="start"
        side={dropdownSide}
        data-testid="model-select-v2-content"
        className="min-w-[360px]"
        style={{ width: 360 }} // eslint-disable-line no-restricted-syntax -- KUI DropdownContent needs explicit width
      >
        {open && (
          <>
            <ModelDropdownSearch value={search} onChange={setSearch} />
            {showModelTypeToggle && (
              <Flex className="px-2 pb-2 w-full">
                <SegmentedControl
                  className="w-full"
                  value={modelType}
                  items={MODEL_TYPE_ITEMS}
                  onValueChange={handleModelTypeChange}
                />
              </Flex>
            )}
            <ModelDropdownList
              groups={filteredGroups}
              value={value}
              onSelect={handleSelect}
              hideAdapters={hideAdapters}
              loading={loading}
              onLoadMore={onLoadMore}
              hasMore={hasMore}
              isLoadingMore={isLoadingMore}
              doneLoadingMessage={doneLoadingMessage}
              emptyMessage={emptyMessage}
            />
          </>
        )}
      </DropdownContent>
    </DropdownRoot>
  );
};
