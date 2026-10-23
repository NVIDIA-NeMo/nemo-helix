// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelpCircle } from '@nemo/icons/icons';
import { Button, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { tooltipClassName } from '@studio/styles/common';
import { type FC } from 'react';

const Tip: FC<{ children: React.ReactNode }> = ({ children }) => (
  <Text kind="body/regular/sm">{children}</Text>
);

const TemplateSyntaxTooltipContent: FC = () => (
  <div className={tooltipClassName}>
    <Stack gap="density-xs">
      <Text kind="label/bold/sm">Template syntax</Text>
      <Tip>
        Values are Jinja2. <code>{'{{ column }}'}</code> inserts that column&apos;s value for the
        row; text outside the braces is kept as-is, so{' '}
        <code>{'Ticket {{ id }}: {{ summary }}'}</code> is one field.
      </Tip>
      <Tip>Text with no braces is a constant — every row gets the same value.</Tip>
      <Tip>
        <strong>Filters</strong> transform a value: <code>{'{{ topic | upper }}'}</code>,{' '}
        <code>{'{{ text | trim }}'}</code>, <code>{'{{ score | int }}'}</code>.
      </Tip>
      <Tip>
        <strong>Fallbacks</strong> cover missing values:{' '}
        <code>{"{{ notes | default('none') }}"}</code> applies only when the column is undefined — a
        defined but empty cell stays empty. Use <code>{"{{ notes | default('none', true) }}"}</code>{' '}
        to replace empty values too.
      </Tip>
      <Text kind="label/bold/sm" className="pt-density-xs">
        Key syntax
      </Text>
      <Tip>
        A dot in a key nests an object: <code>reference.expected</code> becomes{' '}
        <code>{'{ "reference": { "expected": … } }'}</code>.
      </Tip>
      <Tip>
        A number in a key builds an array: <code>messages.0.content</code> and{' '}
        <code>messages.1.content</code> become a two-item <code>messages</code> list.
      </Tip>
      <Tip>
        Leaving a field blank drops it from the output entirely rather than writing an empty string.
      </Tip>
    </Stack>
  </div>
);

/** Help affordance explaining Jinja2 template and key syntax for the field mapping. */
export const TemplateSyntaxTooltip: FC = () => (
  <Tooltip slotContent={<TemplateSyntaxTooltipContent />} side="right">
    <Button
      aria-label="Template syntax help"
      kind="tertiary"
      color="neutral"
      size="small"
      className="cursor-help text-fg-subdued"
    >
      <HelpCircle width={16} height={16} />
    </Button>
  </Tooltip>
);
