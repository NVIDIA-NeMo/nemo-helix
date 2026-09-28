// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import { UseControllerComponentProps } from '@nemo/common/src/types';
import {
  Block,
  Flex,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectProps,
  SelectRoot,
  SelectTrigger,
  Spinner,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { Filter } from 'lucide-react';
import { ChangeEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useController } from 'react-hook-form';

/**
 * SelectTrigger resolves display text with `value || placeholder || "Select item"`, so a real empty
 * string is treated as “unset”. Use ZWSP when the consumer wants no visible placeholder.
 */
const INVISIBLE_TRIGGER_PLACEHOLDER = '\u200b';

export interface SelectItemOption {
  /** The value to be stored in the form */
  value: string;
  /** The display label for the option */
  label: string;
  /** Optional custom render for the option */
  render?: ReactNode;
  /** Optional group key — options sharing one render under a single section header. */
  group?: string;
  /**
   * Text the local typeahead matches against instead of `label`. Use it to make an option
   * findable by more than its own name — e.g. an evaluation that should also match its
   * experiment's name, so typing the experiment narrows the list to that one section.
   */
  searchText?: string;
}

export interface ControlledSearchableSelectProps
  extends
    Omit<SelectProps, 'items' | 'onChange' | 'defaultValue' | 'value' | 'renderValue' | 'kind'>,
    UseControllerComponentProps {
  /** Array of options to display */
  options: SelectItemOption[];
  /** Display labels for option groups, keyed by ``SelectItemOption.group``. Groups in this map render in the order they appear here. */
  groupLabels?: Record<string, string>;
  /** Callback fired when search input changes (for server-side search) */
  onSearchChange?: (searchValue: string) => void;
  /** Callback to load more items when scrolling to bottom */
  onLoadMore?: () => Promise<void>;
  /** Whether there are more items to load */
  hasMore?: boolean;
  /** Whether currently loading initial data */
  isLoading?: boolean;
  /** Whether currently loading more items */
  isLoadingMore?: boolean;
  /** Debounce delay for search input in milliseconds */
  searchDebounceMs?: number;
  /** Placeholder text for the search input */
  searchPlaceholder?: string;
  /**
   * Placeholder text for the select trigger. Pass `''` for no visible placeholder (default copy is
   * avoided via a zero-width space; the field label still describes the control).
   */
  triggerPlaceholder?: string;
  /** Message to show when no options match the search */
  emptyMessage?: string;
  /** Message to show when all items have been loaded */
  doneLoadingMessage?: string;
  /**
   * Sticky footer below the scrollable list (e.g. “Create new”). Use `close` to dismiss the menu
   * before opening another dialog.
   */
  listFooter?: (ctx: { close: () => void }) => ReactNode;
  /** Hide error message */
  hideError?: boolean;
  /** Callback when value changes */
  onChange?: (value: string) => void;
  /** Callback when dropdown opens/closes */
  onOpenChange?: (isOpen: boolean) => void;
  /** Maximum height for the options list */
  maxHeight?: string;
  /** Custom render function for the selected value in the trigger */
  renderValue?: (
    value: string | string[] | undefined,
    setValue:
      | ((nextValue: string | string[]) => void)
      | ((nextValueFunc: (prevValue: string | string[]) => string | string[]) => void)
  ) => ReactNode;
}

export const ControlledSearchableSelect = ({
  options,
  groupLabels,
  onSearchChange,
  onLoadMore,
  hasMore = false,
  isLoading = false,
  isLoadingMore = false,
  searchDebounceMs = 300,
  searchPlaceholder = 'Search...',
  triggerPlaceholder = 'Select an option',
  emptyMessage = 'No results found',
  doneLoadingMessage,
  listFooter,
  hideError = false,
  onChange,
  onOpenChange,
  maxHeight = '300px',
  formFieldProps,
  useControllerProps,
  disabled,
  required,
  status,
  renderValue,
  ...selectProps
}: ControlledSearchableSelectProps) => {
  const [localSearch, setLocalSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [selectOpen, setSelectOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  const [isLoadingMoreLocal, setIsLoadingMoreLocal] = useState(false);

  const {
    field: { onBlur, onChange: onChangeControl, value },
    fieldState: { error },
  } = useController(useControllerProps);

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(localSearch);
      onSearchChange?.(localSearch);
    }, searchDebounceMs);

    return () => clearTimeout(timer);
  }, [localSearch, searchDebounceMs, onSearchChange]);

  // Filter options locally when no server-side search callback provided
  const filteredOptions = useMemo(() => {
    if (onSearchChange) {
      // Server-side search - don't filter locally
      return options;
    }
    // Client-side filtering
    const searchLower = debouncedSearch.toLowerCase();
    return options.filter((option) =>
      (option.searchText ?? option.label).toLowerCase().includes(searchLower)
    );
  }, [options, debouncedSearch, onSearchChange]);

  // Handle infinite scroll
  const loadMoreItems = useCallback(async () => {
    if (isLoadingMoreLocal || !hasMore || !onLoadMore) return;
    setIsLoadingMoreLocal(true);
    try {
      await onLoadMore();
    } finally {
      setIsLoadingMoreLocal(false);
    }
  }, [isLoadingMoreLocal, hasMore, onLoadMore]);

  const showLoading = isLoading || isLoadingMore || isLoadingMoreLocal;
  const loadingMore = isLoadingMore || isLoadingMoreLocal;
  const showDoneFooter = Boolean(doneLoadingMessage) && !hasMore && !showLoading;
  const showLoadMoreFooter = Boolean(onLoadMore) && (hasMore || loadingMore || showDoneFooter);

  useEffect(() => {
    const currentLoaderRef = loaderRef.current;
    if (!showLoadMoreFooter || !currentLoaderRef) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMoreItems();
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(currentLoaderRef);

    return () => {
      observer.unobserve(currentLoaderRef);
    };
  }, [loadMoreItems, showLoadMoreFooter]);

  useEffect(() => {
    if (!selectOpen) {
      setLocalSearch('');
      setDebouncedSearch('');
      return;
    }
    const timeoutId = setTimeout(() => searchInputRef.current?.focus(), 0);
    return () => clearTimeout(timeoutId);
  }, [selectOpen]);

  const handleSelectOpenChange = (isOpen: boolean) => {
    setSelectOpen(isOpen);
    onOpenChange?.(isOpen);
  };

  const handleValueChange = (newValue: string) => {
    onChange?.(newValue);
    onChangeControl(newValue);
  };

  const handleBlur = () => {
    onBlur();
  };

  const resolveLabel = useCallback(
    (optionValue: string) => options.find((option) => option.value === optionValue)?.label,
    [options]
  );

  const defaultRenderValue = useCallback(
    (currentValue: string | string[] | undefined) => {
      if (Array.isArray(currentValue)) {
        return currentValue.map((v) => resolveLabel(v) ?? v).join(', ');
      }
      return currentValue ? (resolveLabel(currentValue) ?? currentValue) : undefined;
    },
    [resolveLabel]
  );

  /**
   * The trigger truncates long values with an ellipsis; without this tooltip, two options with
   * different names but the same visible prefix are indistinguishable once selected.
   */
  const triggerTooltip = Array.isArray(value)
    ? value.map((v) => resolveLabel(v) ?? v).join(', ')
    : value
      ? (resolveLabel(value) ?? value)
      : undefined;

  return (
    <FormField
      name={useControllerProps.name}
      slotError={hideError ? undefined : error?.message}
      status={status || (error ? 'error' : undefined)}
      required={required}
      {...formFieldProps}
    >
      <SelectRoot
        disabled={disabled}
        value={value ?? ''}
        onValueChange={handleValueChange}
        open={selectOpen}
        onOpenChange={handleSelectOpenChange}
      >
        <Tooltip slotContent={triggerTooltip} disabled={!triggerTooltip}>
          <SelectTrigger
            renderValue={renderValue ?? defaultRenderValue}
            className="w-full border-1 nv-input"
            onBlur={handleBlur}
            placeholder={
              isLoading
                ? 'Loading...'
                : triggerPlaceholder === ''
                  ? INVISIBLE_TRIGGER_PLACEHOLDER
                  : triggerPlaceholder
            }
            slotEnd={isLoading && <TextInputSpinner />}
            required={required}
            status={status || (error ? 'error' : undefined)}
            {...selectProps}
          />
        </Tooltip>
        {/* No `min-w-full`: it resolves against the anchored popover's containing block, not the
            trigger, so it stretches near-viewport-wide and (per the CSS min/max conflict rule)
            wins over `max-w-*` outright. */}
        <SelectContent className="w-max max-w-96">
          <SelectListbox>
            <Block className="p-2 w-full sticky top-0 bg-surface z-10">
              <TextInput
                ref={searchInputRef}
                name={`${useControllerProps.name}-search`}
                className="overflow-hidden"
                slotStart={<Filter />}
                placeholder={searchPlaceholder}
                value={localSearch}
                onChange={(e: ChangeEvent<HTMLInputElement>) => {
                  setLocalSearch(e.target.value);
                }}
                attributes={{
                  Input: {
                    ['data-testid' as never]: `${useControllerProps.name}-search`,
                  },
                }}
              />
            </Block>
            {/* eslint-disable-next-line no-restricted-syntax */}
            <Stack className="overflow-auto w-full" style={{ maxHeight }}>
              {isLoading && filteredOptions.length === 0 ? (
                <Flex align="center" justify="center" className="py-4">
                  <Spinner aria-label="Loading options" size="small" />
                </Flex>
              ) : filteredOptions.length > 0 ? (
                <>
                  {isLoading && (
                    <Flex align="center" justify="center" className="py-2">
                      <Spinner aria-label="Refreshing options" size="small" />
                    </Flex>
                  )}
                  {(() => {
                    const groupKeys = groupLabels ? Object.keys(groupLabels) : [];
                    if (groupKeys.length === 0) {
                      return filteredOptions.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.render ?? option.label}
                        </SelectItem>
                      ));
                    }
                    const buckets = new Map<string | undefined, SelectItemOption[]>();
                    for (const option of filteredOptions) {
                      const key = option.group;
                      const existing = buckets.get(key);
                      if (existing) existing.push(option);
                      else buckets.set(key, [option]);
                    }
                    const orderedKeys = [
                      ...groupKeys.filter((k) => buckets.has(k)),
                      ...Array.from(buckets.keys()).filter(
                        (k): k is string | undefined => !k || !groupKeys.includes(k)
                      ),
                    ];
                    return orderedKeys.flatMap((key) => {
                      const items = buckets.get(key) ?? [];
                      if (items.length === 0) return [];
                      const heading = key ? groupLabels?.[key] : undefined;
                      return [
                        heading ? (
                          <Block
                            key={`__heading-${key}`}
                            className="px-2 pt-2 pb-1 text-xs uppercase tracking-wide text-subtle"
                          >
                            {heading}
                          </Block>
                        ) : null,
                        ...items.map((option) => (
                          <SelectItem key={`${key ?? ''}:${option.value}`} value={option.value}>
                            {option.render ?? option.label}
                          </SelectItem>
                        )),
                      ].filter(Boolean);
                    });
                  })()}
                  {/* Infinite scroll sentinel; collapsed to 1px when idle with more pages (no empty min-height) */}
                  {showLoadMoreFooter ? (
                    <Flex
                      ref={loaderRef}
                      align="center"
                      justify="center"
                      className={
                        loadingMore || showDoneFooter
                          ? 'min-h-8 py-2'
                          : 'h-px min-h-px w-full shrink-0 py-0'
                      }
                    >
                      {loadingMore ? <Spinner aria-label="Loading more" size="small" /> : null}
                      {showDoneFooter ? (
                        <Text kind="body/regular/sm" className="text-subtle">
                          {doneLoadingMessage}
                        </Text>
                      ) : null}
                    </Flex>
                  ) : null}
                </>
              ) : (
                <Flex align="center" justify="center" className="py-4">
                  <Text kind="body/regular/sm" className="text-subtle">
                    {emptyMessage}
                  </Text>
                </Flex>
              )}
            </Stack>
            {listFooter ? (
              <Block className="shrink-0 border-t-1 border-t-base p-0 w-full">
                {listFooter({
                  close: () => setSelectOpen(false),
                })}
              </Block>
            ) : null}
          </SelectListbox>
        </SelectContent>
      </SelectRoot>
    </FormField>
  );
};
