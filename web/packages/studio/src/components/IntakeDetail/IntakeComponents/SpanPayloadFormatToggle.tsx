// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Tooltip } from '@nvidia/foundations-react-core';
import type {
  SpanPayloadFormat,
  SpanPayloadFormatState,
} from '@studio/components/IntakeDetail/IntakeComponents/spanPayloadFormat';
import { type FC, type MouseEvent } from 'react';

const FORMAT_OPTIONS: readonly { format: SpanPayloadFormat; label: string; name: string }[] = [
  { format: 'raw', label: 'raw', name: 'raw text' },
  { format: 'md', label: 'md', name: 'markdown' },
  { format: 'json', label: 'json', name: 'JSON' },
  { format: 'chat', label: 'chat', name: 'a chat' },
];

interface SpanPayloadFormatToggleProps {
  state: SpanPayloadFormatState;
  /** Names the payload in labels and tooltips, e.g. `input` → "View input as JSON". */
  payloadLabel: string;
}

export const SpanPayloadFormatToggle: FC<SpanPayloadFormatToggleProps> = ({
  state,
  payloadLabel,
}) => {
  if (state.isEmpty) {
    return null;
  }

  // The trigger row is a <summary>; keep clicks from toggling/collapsing it.
  const withoutToggle = (handler: () => void) => (event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    handler();
  };

  return (
    <Flex align="center" gap="density-xs" className="shrink-0">
      {FORMAT_OPTIONS.map(({ format, label, name }) => {
        const unavailableReason =
          format === 'json' && !state.isJson
            ? `This ${payloadLabel} is not valid JSON`
            : format === 'chat' && !state.isChat
              ? `This ${payloadLabel} is not an OpenAI chat payload`
              : null;
        const unavailable = unavailableReason !== null;
        const active = state.format === format;
        const button = (
          <Button
            type="button"
            size="tiny"
            kind="tertiary"
            className="font-mono"
            color={active ? 'brand' : 'neutral'}
            aria-label={`View ${payloadLabel} as ${name}`}
            aria-pressed={active}
            disabled={unavailable}
            onClick={withoutToggle(() => state.select(format))}
          >
            {label}
          </Button>
        );
        return (
          <Tooltip key={format} side="top" slotContent={unavailableReason ?? `View as ${name}`}>
            {/* A disabled button fires no hover or focus events, so its
                tooltip needs a focusable wrapper. */}
            {unavailable ? <span tabIndex={0}>{button}</span> : button}
          </Tooltip>
        );
      })}
    </Flex>
  );
};
