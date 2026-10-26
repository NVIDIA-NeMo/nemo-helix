// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

/** Where a name's uniqueness check stands. A debounced value that has fallen behind what is on
 *  screen reads as still checking, never as a verdict for a name the user has moved on from. */
export type NameCheckStatus = 'checking' | 'conflict' | 'failed' | 'available' | undefined;

export const nameCheckStatus = (
  preview: string,
  debounced: string,
  query: { data?: { data?: unknown[] }; isFetching: boolean; isError: boolean }
): NameCheckStatus => {
  if (!preview) return undefined;
  if (debounced !== preview || query.isFetching) return 'checking';
  if (query.isError) return 'failed';
  return (query.data?.data?.length ?? 0) > 0 ? 'conflict' : 'available';
};

const withArticle = (noun: string): string => `${/^[aeiou]/i.test(noun) ? 'An' : 'A'} ${noun}`;

/** The error for a name already in use. Naming the workspace says where the name must be
 *  unique, so the user knows another workspace would not collide. */
export const nameTakenMessage = (entity: string, name: string, workspace?: string): string =>
  workspace
    ? `${withArticle(entity)} named ${name} already exists in the ${workspace} workspace`
    : `${withArticle(entity)} named ${name} already exists`;

/** slotHelp/slotError for a name field, first match wins per the contract's precedence table.
 *  Pass `workspace` when the name is unique per workspace, so the copy says so. */
export const nameFieldSlots = ({
  entity,
  preview,
  status,
  schemaError,
  describe,
  workspace,
}: {
  entity: string;
  preview: string;
  status: NameCheckStatus;
  schemaError?: string;
  describe: ReactNode;
  workspace?: string;
}): { slotHelp?: ReactNode; slotError?: string; status?: 'error' } => {
  if (status === 'checking') return { slotHelp: 'Checking name...' };
  if (status === 'conflict')
    return { slotError: nameTakenMessage(entity, preview, workspace), status: 'error' };
  if (schemaError) return { slotError: schemaError, status: 'error' };
  if (status === 'failed')
    return { slotHelp: "Couldn't check name availability. You can still submit." };
  if (!preview) return { slotHelp: describe };
  return {
    slotHelp: (
      <>
        Your {entity} will be created as <span className="text-primary">{preview}</span>
        {workspace && (
          <>
            {' '}
            in the <span className="text-primary">{workspace}</span> workspace
          </>
        )}
      </>
    ),
  };
};
