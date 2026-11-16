// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentDropdownItem } from '@nemo/common/src/components/AgentSelect/AgentDropdownItem';
import type { AgentRow, AgentSelectProps } from '@nemo/common/src/components/AgentSelect/types';
import {
  buildAgentRows,
  groupAgentsByFormat,
  matchesAgentSearch,
} from '@nemo/common/src/components/AgentSelect/utils';
import { ModelDropdownSearch } from '@nemo/common/src/components/ModelSelectV2/ModelDropdownSearch';
import { VirtualDropdownList } from '@nemo/common/src/components/VirtualDropdownList';
import { ChevronDown, HourglassMid } from '@nemo/icons/icons';
import type { Agent } from '@nemo/sdk/generated/agents/schema';
import {
  Button,
  DropdownContent,
  DropdownHeading,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  Tag,
  Text,
} from '@nvidia/foundations-react-core';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type FC } from 'react';
import { useDebounce } from 'use-debounce';

const DEFAULT_SEARCH_DEBOUNCE_MS = 300;
const HEADING_HEIGHT = 32;
const ITEM_HEIGHT = 36;

const estimateSize = (row: AgentRow) => (row.kind === 'heading' ? HEADING_HEIGHT : ITEM_HEIGHT);

/**
 * The one picker for agents, single or multiple (`multiple`). Rows show each agent's details on
 * hover, can be grouped (`groupBy`), and the list is virtualized with search-as-you-type and
 * scroll paging. Use `WorkspaceAgentSelect` to have it fetch its own agents.
 */
export const AgentSelect: FC<AgentSelectProps> = (props) => {
  const {
    agents,
    groupBy,
    onSearchChange,
    searchDebounceMs = DEFAULT_SEARCH_DEBOUNCE_MS,
    onLoadMore,
    hasMore,
    isLoadingMore,
    doneLoadingMessage,
    emptyMessage = 'No agents found',
    loading = false,
    getDisabledReason,
    disabled = false,
    placeholder = props.multiple ? 'Select agents' : 'Select an agent',
    fullWidth = false,
    dropdownSide = 'bottom',
    onOpenChange,
    'aria-label': ariaLabel,
  } = props;

  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [debouncedSearch] = useDebounce(search, searchDebounceMs);

  // Read through a ref so an inline `onSearchChange` only fires when the debounced text changes.
  const onSearchChangeRef = useRef(onSearchChange);
  useLayoutEffect(() => {
    onSearchChangeRef.current = onSearchChange;
  });
  useEffect(() => {
    onSearchChangeRef.current?.(debouncedSearch);
  }, [debouncedSearch]);

  // Every agent seen so far, so a multi-selection made across searches still hands back entities.
  const seen = useRef(new Map<string, Agent>());
  useEffect(() => {
    agents?.forEach((agent) => agent.name && seen.current.set(agent.name, agent));
  }, [agents]);

  const selected: readonly string[] = props.multiple
    ? props.value
    : props.value
      ? [props.value]
      : [];
  const atMax =
    props.multiple && props.maxSelected !== undefined && selected.length >= props.maxSelected;

  const rows = useMemo(() => {
    const visible =
      onSearchChange || !debouncedSearch
        ? (agents ?? [])
        : (agents ?? []).filter((agent) => matchesAgentSearch(agent, debouncedSearch));
    return buildAgentRows(visible, groupBy);
  }, [agents, debouncedSearch, groupBy, onSearchChange]);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      setOpen(nextOpen);
      if (!nextOpen) setSearch('');
      onOpenChange?.(nextOpen);
    },
    [onOpenChange]
  );

  // Read through a ref so `toggle` stays stable and the memoized rows skip re-rendering.
  const latest = useRef({ props, handleOpenChange });
  useLayoutEffect(() => {
    latest.current = { props, handleOpenChange };
  });

  const toggle = useCallback((agent: Agent) => {
    const { props: current, handleOpenChange: close } = latest.current;
    const name = agent.name ?? '';
    if (!current.multiple) {
      current.onValueChange(name, agent);
      close(false);
      return;
    }
    const next = current.value.includes(name)
      ? current.value.filter((n) => n !== name)
      : [...current.value, name];
    current.onValueChange(
      next,
      next.flatMap((n) => {
        const entity = n === name ? agent : seen.current.get(n);
        return entity ? [entity] : [];
      })
    );
  }, []);

  const renderRow = (row: AgentRow) => {
    if (row.kind === 'heading') return <DropdownHeading>{row.label}</DropdownHeading>;
    const name = row.agent.name ?? '';
    const isSelected = selected.includes(name);
    const disabledReason =
      getDisabledReason?.(row.agent) ??
      (atMax && !isSelected && props.multiple
        ? `Up to ${props.maxSelected} agents can be selected.`
        : undefined);
    return (
      <AgentDropdownItem
        agent={row.agent}
        isSelected={isSelected}
        disabledReason={disabledReason}
        showFormat={groupBy !== groupAgentsByFormat}
        onToggle={toggle}
      />
    );
  };

  const [first, ...rest] = selected;

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
          aria-label={ariaLabel ?? placeholder}
          data-testid="agent-select-trigger"
          className="overflow-hidden data-[state=open]:border-(--border-color-feedback-success) data-[state=open]:bg-(--background-color-interaction-base)"
        >
          <Flex
            align="center"
            gap="density-sm"
            className={`min-w-0 ${fullWidth ? 'w-full justify-between' : ''}`}
          >
            <Flex align="center" gap="density-sm" className="min-w-0 flex-1">
              {loading && !first && <HourglassMid size="16" className="animate-spin shrink-0" />}
              <Text className="truncate">{first ?? placeholder}</Text>
              {rest.length > 0 && (
                <Tag color="gray" kind="outline" density="compact" readOnly>
                  {`+${rest.length}`}
                </Tag>
              )}
            </Flex>
            <ChevronDown size="16" className="shrink-0" />
          </Flex>
        </Button>
      </DropdownTrigger>
      <DropdownContent
        align="start"
        side={dropdownSide}
        data-testid="agent-select-content"
        className="min-w-[360px] min-h-[200px]"
        style={{ width: 360 }} // eslint-disable-line no-restricted-syntax -- KUI DropdownContent needs explicit width
      >
        {open && (
          <>
            <ModelDropdownSearch
              value={search}
              onChange={setSearch}
              name="agent-filter"
              testId="agent-select-filter"
            />
            {props.multiple && selected.length > 0 && (
              <Flex align="center" justify="between" className="w-full px-3 pb-1">
                <Text className="text-secondary" kind="body/regular/sm">
                  {props.maxSelected
                    ? `${selected.length} of ${props.maxSelected} selected`
                    : `${selected.length} selected`}
                </Text>
                <Button
                  kind="tertiary"
                  size="tiny"
                  data-testid="agent-select-clear"
                  onClick={() => props.onValueChange([], [])}
                >
                  Clear
                </Button>
              </Flex>
            )}
            <VirtualDropdownList
              rows={rows}
              renderRow={renderRow}
              estimateSize={estimateSize}
              loading={loading}
              onLoadMore={onLoadMore}
              hasMore={hasMore}
              isLoadingMore={isLoadingMore}
              doneLoadingMessage={doneLoadingMessage}
              emptyMessage={emptyMessage}
              loadingMessage="Loading agents..."
            />
          </>
        )}
      </DropdownContent>
    </DropdownRoot>
  );
};
