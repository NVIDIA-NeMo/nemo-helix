// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InfoCircle } from '@nemo/icons/icons';
import { Text, Tooltip } from '@nvidia/foundations-react-core';
import { tooltipClassName } from '@studio/styles/common';
import type { FC, KeyboardEvent, SyntheticEvent } from 'react';

const keepParentClosed = (event: SyntheticEvent) => {
  event.preventDefault();
  event.stopPropagation();
};

const keepParentClosedOnActivate = (event: KeyboardEvent) => {
  if (event.key === 'Enter' || event.key === ' ') {
    keepParentClosed(event);
  }
};

/** An info icon with explanatory text on hover; clicks never reach an enclosing accordion toggle. */
export const InfoTip: FC<{ label: string; text: string }> = ({ label, text }) => (
  <Tooltip
    slotContent={
      <div className={tooltipClassName}>
        <Text kind="body/regular/sm" className="block max-w-[420px]">
          {text}
        </Text>
      </div>
    }
  >
    <span
      role="button"
      tabIndex={0}
      aria-label={`About ${label}`}
      className="inline-flex cursor-help text-secondary"
      onClick={keepParentClosed}
      onKeyDown={keepParentClosedOnActivate}
    >
      <InfoCircle />
    </span>
  </Tooltip>
);
