// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FC, PropsWithChildren } from 'react';

export interface SelectableCardProps {
  selected: boolean;
  onSelect: () => void;
  /**
   * Inert rather than hidden, so the full set of choices stays visible. Suppresses the
   * hover affordance and selection alongside `aria-disabled`.
   */
  disabled?: boolean;
  /** Sizing and inner spacing, which differ per card. */
  className?: string;
}

/**
 * A card that behaves like a radio option: a `<button>` carrying the selected, hover and
 * focus treatment, with its contents left to the caller.
 *
 * Shared because the start options and the recipe tiles sit on the same screen, so the two
 * drifting apart shows up directly as one set of cards not matching the other.
 */
export const SelectableCard: FC<PropsWithChildren<SelectableCardProps>> = ({
  selected,
  onSelect,
  disabled = false,
  className = '',
  children,
}) => {
  const stateClasses = disabled
    ? 'cursor-not-allowed border-base opacity-50'
    : selected
      ? 'cursor-pointer border-[#76b900]'
      : 'cursor-pointer border-base hover:-translate-y-0.5 hover:border-[#76b900] hover:bg-surface-hover hover:shadow-md';

  return (
    <button
      type="button"
      onClick={disabled ? undefined : onSelect}
      aria-pressed={disabled ? undefined : selected}
      aria-disabled={disabled}
      className={`flex w-full flex-col items-start rounded-md border bg-surface-raised text-left transition focus-visible:border-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#76b900] ${className} ${stateClasses}`}
    >
      {children}
    </button>
  );
};
