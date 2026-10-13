// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import { ModelDetailsTooltip } from '@nemo/common/src/components/ModelDetailsTooltip';
import { creatorToIcon } from '@nemo/common/src/constants/modelMetadata';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import { UseControllerComponentProps } from '@nemo/common/src/types';
import { filterModel } from '@nemo/common/src/utils/models';
import { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  Block,
  DropdownHeading,
  DropdownSection,
  Flex,
  FormField,
  SelectContent,
  SelectItem,
  SelectProps,
  SelectRoot,
  SelectTrigger,
  Spinner,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { Filter } from 'lucide-react';
import {
  type ChangeEvent,
  type FC,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useController } from 'react-hook-form';

const DEFAULT_SEARCH_DEBOUNCE_MS = 300;
const DEFAULT_MAX_HEIGHT = '300px';

export type ModelSelectProps = Omit<SelectProps, 'items' | 'onChange'> &
  UseControllerComponentProps & {
    disableClearable?: boolean;
    helperText?: string;
    loading?: boolean;
    /** Flat list of models. Used for single-workspace dropdowns (legacy). */
    models?: ModelEntity[];
    /** Grouped models (workspace → namespace → models). Used for two-workspace dropdowns. */
    groups?: ModelWorkspaceGroup[];
    onChange?: (value: string) => void;
    onOpen?: (value: boolean) => void;
    tooltip?: string;
    value?: string;
    /** When provided, search is server-driven: debounced callback and no local filter */
    onSearchChange?: (searchValue: string) => void;
    /** Callback to load next page when user scrolls to bottom */
    onLoadMore?: () => Promise<unknown>;
    /** Whether there are more pages to load */
    hasMore?: boolean;
    /** Whether the next page is currently loading */
    isLoadingMore?: boolean;
    /** Debounce delay for search in ms (when onSearchChange is used) */
    searchDebounceMs?: number;
    /** Max height of the options list */
    maxHeight?: string;
    /** Message shown at the bottom when all pages have been loaded (infinite scroll) */
    doneLoadingMessage?: string;
  };

const MODEL_LIST_LIMIT = 500;

export const ModelSelect: FC<ModelSelectProps> = ({
  disabled,
  loading,
  models,
  groups,
  onBlur,
  onChange,
  placeholder,
  required,
  status,
  tooltip,
  formFieldProps = {},
  useControllerProps,
  onOpen,
  onSearchChange,
  onLoadMore,
  hasMore = false,
  isLoadingMore = false,
  searchDebounceMs = DEFAULT_SEARCH_DEBOUNCE_MS,
  maxHeight = DEFAULT_MAX_HEIGHT,
  doneLoadingMessage = 'All models loaded',
}) => {
  const [localSearch, setLocalSearch] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const filterInputRef = useRef<HTMLInputElement>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  const [isLoadingMoreLocal, setIsLoadingMoreLocal] = useState(false);

  const {
    field: { onBlur: onBlurField, onChange: onChangeField, value, disabled: fieldDisabled },
    fieldState: { error },
  } = useController(useControllerProps);

  const serverDrivenSearch = !!onSearchChange;

  // Debounce search and notify parent when using server-driven search
  useEffect(() => {
    if (!serverDrivenSearch) return;
    const timer = setTimeout(() => {
      onSearchChange(localSearch);
    }, searchDebounceMs);
    return () => clearTimeout(timer);
  }, [localSearch, searchDebounceMs, onSearchChange, serverDrivenSearch]);

  const handleBlur = (event: React.FocusEvent<HTMLDivElement>) => {
    onBlurField();
    onBlur?.(event);
  };

  const handleChange = (value: string) => {
    onChangeField(value);
    onChange?.(value);
  };

  const loadMoreItems = useCallback(async () => {
    if (isLoadingMoreLocal || !hasMore || !onLoadMore) return;
    setIsLoadingMoreLocal(true);
    try {
      await onLoadMore();
    } finally {
      setIsLoadingMoreLocal(false);
    }
  }, [isLoadingMoreLocal, hasMore, onLoadMore]);

  useEffect(() => {
    if (!onLoadMore || !loaderRef.current || !isDropdownOpen) return;
    const currentLoaderRef = loaderRef.current;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) loadMoreItems();
      },
      { threshold: 0.1 }
    );
    observer.observe(currentLoaderRef);
    return () => observer.unobserve(currentLoaderRef);
  }, [loadMoreItems, onLoadMore, isDropdownOpen]);

  const modelsToShow = useMemo(() => {
    if (!models) return [];
    if (serverDrivenSearch) return models;
    return models.filter((model) => filterModel(model, localSearch)).slice(0, MODEL_LIST_LIMIT);
  }, [models, serverDrivenSearch, localSearch]);

  // Flat models path: group by workspace (client-side filter applied)
  const namespaceToModels = useMemo(
    () =>
      modelsToShow.reduce(
        (acc, model) => {
          const namespace = model.workspace;
          if (!namespace) return acc;
          if (!acc[namespace]) acc[namespace] = [];
          acc[namespace].push(model);
          return acc;
        },
        {} as Record<string, ModelEntity[]>
      ),
    [modelsToShow]
  );

  const sortedNamespaces: string[] = useMemo(() => {
    return Object.keys(namespaceToModels).sort((a, b) => a.localeCompare(b));
  }, [namespaceToModels]);
  // Determine selected model's creator for trigger icon
  const foundModel = useMemo(() => {
    if (groups) {
      return groups.flatMap((g) => g.models).find((m) => getEntityReference(m) === value);
    }
    return models?.find((model) => getEntityReference(model) === value);
  }, [groups, models, value]);

  const creator = foundModel?.workspace;

  const hasModels = groups ? groups.some((g) => g.models.length > 0) : sortedNamespaces.length > 0;

  const handleOpenChange = (open: boolean) => {
    setIsDropdownOpen(open);
    onOpen?.(open);
    if (!open) {
      setLocalSearch('');
      return;
    }
    setTimeout(() => filterInputRef.current?.focus(), 0);
  };

  const handleSearchChange = (e: ChangeEvent<HTMLInputElement>) => {
    const next = e.target.value;
    setLocalSearch(next);
  };

  const showLoadingMore = isLoadingMore || isLoadingMoreLocal;

  return (
    <FormField
      attributes={{
        FormFieldLabelGroup: {
          className: 'justify-between',
        },
      }}
      slotInfo={tooltip}
      status={status || (error ? 'error' : undefined)}
      required={required}
      slotError={error?.message}
      {...formFieldProps}
    >
      <SelectRoot
        disabled={fieldDisabled || disabled}
        value={value ?? null}
        onValueChange={(value: string) => {
          handleChange(value);
        }}
        onOpenChange={handleOpenChange}
      >
        <SelectTrigger
          className={`w-full border-1 ${disabled ? 'nv-input-disabled' : 'nv-input'} relative`}
          onBlur={handleBlur}
          placeholder={
            loading ? 'Loading Models...' : placeholder || 'Select a model to get started'
          }
          slotStart={creator && creatorToIcon(creator, { className: 'text-base' })}
          slotEnd={loading && <TextInputSpinner />}
          required={required}
          status={status || (error ? 'error' : undefined)}
        />
        <SelectContent className="w-(--radix-popper-anchor-width)">
          <Block className="p-2 w-full sticky top-0 bg-surface z-10">
            <TextInput
              ref={filterInputRef}
              name="model-filter"
              className="overflow-hidden"
              slotStart={<Filter />}
              placeholder={serverDrivenSearch ? 'Search...' : 'Filter'}
              value={localSearch}
              onChange={handleSearchChange}
              attributes={{
                Input: {
                  ['data-testid' as never]: 'model-filter',
                },
              }}
            />
          </Block>
          {/* eslint-disable-next-line no-restricted-syntax */}
          <Stack className="overflow-auto overscroll-contain w-full" style={{ maxHeight }}>
            {hasModels ? (
              groups ? (
                // Grouped path: workspace → models (server-side search, no client filter)
                groups.map((workspaceGroup) => (
                  <DropdownSection key={workspaceGroup.workspace}>
                    <DropdownHeading>
                      <Flex gap="density-sm" align="center">
                        {creatorToIcon(workspaceGroup.workspace, { className: 'text-base' })}
                        <Text className="font-bold">{workspaceGroup.workspace}</Text>
                      </Flex>
                    </DropdownHeading>
                    {workspaceGroup.models.map((model) => (
                      <ModelDetailsTooltip
                        key={model.id}
                        model={model}
                        triggerProps={{ className: 'w-full text-left', type: 'button' }}
                      >
                        <SelectItem className="relative" value={getEntityReference(model)}>
                          <Flex className="w-full" align="center" justify="between">
                            <Text className="truncate flex-1">{model.name?.split('@')[0]}</Text>
                            {model.name?.includes('@') && (
                              <Text
                                className="text-subtle truncate ml-2 max-w-16 text-left"
                                // eslint-disable-next-line no-restricted-syntax -- RTL does not work with className, this shows ellipsis at beginning of text
                                style={{ direction: 'rtl' }}
                              >
                                {model.name?.split('@')[1]}
                              </Text>
                            )}
                          </Flex>
                        </SelectItem>
                      </ModelDetailsTooltip>
                    ))}
                  </DropdownSection>
                ))
              ) : (
                <>
                  {/* Flat path: namespace (=workspace) → models (client-side search) */}
                  {sortedNamespaces.map((namespace) => (
                    <DropdownSection key={namespace}>
                      <DropdownHeading>
                        <Flex gap="density-sm" align="center">
                          {creatorToIcon(namespace, { className: 'text-base' })}
                          <Text className="font-bold">{namespace}</Text>
                        </Flex>
                      </DropdownHeading>
                      {namespaceToModels[namespace].map((model) => (
                        <ModelDetailsTooltip
                          key={model.id}
                          model={model}
                          triggerProps={{ className: 'w-full text-left', type: 'button' }}
                        >
                          <SelectItem className="relative" value={getEntityReference(model)}>
                            <Flex className="w-full" align="center" justify="between">
                              <Text className="truncate flex-1">{model.name?.split('@')[0]}</Text>
                              {model.name?.includes('@') && (
                                <Text
                                  className="text-subtle truncate ml-2 max-w-16 text-left"
                                  // eslint-disable-next-line no-restricted-syntax -- RTL does not work with className, this shows ellipsis at beginning of text
                                  style={{ direction: 'rtl' }}
                                >
                                  {model.name?.split('@')[1]}
                                </Text>
                              )}
                            </Flex>
                          </SelectItem>
                        </ModelDetailsTooltip>
                      ))}
                    </DropdownSection>
                  ))}
                  {onLoadMore != null && (
                    <Flex ref={loaderRef} align="center" justify="center" className="py-2 min-h-8">
                      {showLoadingMore && <Spinner aria-label="Loading more models" size="small" />}
                      {!showLoadingMore && !hasMore && (
                        <Text kind="body/regular/sm" className="text-subtle">
                          {doneLoadingMessage}
                        </Text>
                      )}
                    </Flex>
                  )}
                </>
              )
            ) : (
              <DropdownSection>
                {!loading && (
                  <DropdownHeading>
                    <Text>No Models Found</Text>
                  </DropdownHeading>
                )}
              </DropdownSection>
            )}
          </Stack>
        </SelectContent>
      </SelectRoot>
    </FormField>
  );
};
