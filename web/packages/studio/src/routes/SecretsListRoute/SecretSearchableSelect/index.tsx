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
  ControlledSearchableSelect,
  SelectItemOption,
} from '@nemo/common/src/components/form/ControlledSearchableSelect';
import {
  getSecretsListSecretsQueryKey,
  secretsListSecrets,
} from '@nemo/sdk/generated/platform/secrets';
import { MenuItem } from '@nvidia/foundations-react-core';
import { useInfiniteQuery } from '@tanstack/react-query';
import { type ReactNode, useCallback, useMemo } from 'react';
import { type FieldValues, type UseControllerProps } from 'react-hook-form';

const SECRETS_PAGE_SIZE = 20;

/** Value used for the optional “no secret” row in the select. */
export const NO_SECRET_SELECT_VALUE = '';

export type SecretSearchableSelectFormFieldProps = {
  slotLabel?: ReactNode;
  slotInfo?: ReactNode;
  slotError?: string;
};

export type SecretSearchableSelectProps<T extends FieldValues> = {
  workspace: string;
  /** When false, the secrets query does not run (e.g. parent panel/modal closed). */
  queryEnabled?: boolean;
  /** Ensures this value appears in the option list (e.g. current secret when editing). */
  ensureOptionValue?: string;
  useControllerProps: UseControllerProps<T>;
  onRequestNewSecret: () => void;
  formFieldProps: SecretSearchableSelectFormFieldProps;
  /** Use `''` for no visible trigger text (see `ControlledSearchableSelect` `triggerPlaceholder`). */
  triggerPlaceholder?: string;
};

export function SecretSearchableSelect<T extends FieldValues>({
  workspace,
  queryEnabled = true,
  ensureOptionValue,
  useControllerProps,
  onRequestNewSecret,
  formFieldProps,
  triggerPlaceholder = 'Select a secret (optional)',
}: SecretSearchableSelectProps<T>) {
  const {
    data: secretsPages,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: isLoadingSecrets,
  } = useInfiniteQuery({
    queryKey: [...getSecretsListSecretsQueryKey(workspace), 'infinite'] as const,
    queryFn: ({ signal, pageParam }) =>
      secretsListSecrets(workspace, { page: pageParam, page_size: SECRETS_PAGE_SIZE }, signal),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      const p = lastPage.pagination;
      return p && p.page < p.total_pages ? p.page + 1 : undefined;
    },
    enabled: queryEnabled && !!workspace,
  });

  const secretOptions: SelectItemOption[] = useMemo(() => {
    const list = secretsPages?.pages.flatMap((page) => page.data) ?? [];
    const options: SelectItemOption[] = [
      { value: NO_SECRET_SELECT_VALUE, label: 'None' },
      ...list.map((secret) => ({ value: secret.name, label: secret.name })),
    ];
    if (ensureOptionValue && !options.some((o) => o.value === ensureOptionValue)) {
      options.push({ value: ensureOptionValue, label: ensureOptionValue });
    }
    return options;
  }, [secretsPages?.pages, ensureOptionValue]);

  const handleLoadMoreSecrets = useCallback(async () => {
    if (hasNextPage && !isFetchingNextPage) await fetchNextPage();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  return (
    <ControlledSearchableSelect
      useControllerProps={useControllerProps as unknown as UseControllerProps<FieldValues>}
      options={secretOptions}
      onLoadMore={handleLoadMoreSecrets}
      hasMore={hasNextPage ?? false}
      isLoading={isLoadingSecrets}
      isLoadingMore={isFetchingNextPage}
      searchPlaceholder="Search secrets..."
      emptyMessage="No secrets found"
      triggerPlaceholder={triggerPlaceholder}
      listFooter={({ close }) => (
        <MenuItem
          onClick={() => {
            close();
            onRequestNewSecret();
          }}
        >
          New Secret
        </MenuItem>
      )}
      formFieldProps={formFieldProps}
    />
  );
}
