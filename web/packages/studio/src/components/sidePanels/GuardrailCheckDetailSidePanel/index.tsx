// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronLeft, ChevronRight } from '@nemo/icons/icons';
import type { RailsConfig } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, SidePanel, Text } from '@nvidia/foundations-react-core';
import type { GuardrailCheckEntity } from '@studio/api/guardrail-checks/types';
import { ConversationPane } from '@studio/components/sidePanels/GuardrailCheckDetailSidePanel/ConversationPane';
import { ResultsPane } from '@studio/components/sidePanels/GuardrailCheckDetailSidePanel/ResultsPane';
import type { FC } from 'react';

export interface GuardrailCheckDetailSidePanelProps {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly check: GuardrailCheckEntity;
  /** The parent config's rails, used to list declared guardrail coverage. */
  readonly configData: RailsConfig | undefined;
  /** The check's stable number in the full test list, as the Tests sub-tab numbers its cards. */
  readonly checkIndex: number;
  /**
   * Position among the rows the results table is showing, or null when the
   * check is not one of them — navigation is hidden in that case.
   */
  readonly visibleIndex: number | null;
  /** How many rows the results table is showing. */
  readonly visibleCount: number;
  readonly onNavigate: (visibleIndex: number) => void;
}

/**
 * Detail view for one guardrail test result: the conversation on the left, the
 * verdict and per-rail breakdown on the right.
 *
 * Non-modal, so the results table stays interactive behind it and prev/next can
 * walk the visible rows — `visibleIndex`, not `checkIndex` — without reopening.
 */
export const GuardrailCheckDetailSidePanel: FC<GuardrailCheckDetailSidePanelProps> = ({
  open,
  onClose,
  check,
  configData,
  checkIndex,
  visibleIndex,
  visibleCount,
  onNavigate,
}) => {
  return (
    <SidePanel
      // Clamped to the viewport so the two panes stay side by side on a laptop
      // screen instead of overflowing off the edge.
      className="w-[min(960px,90vw)]"
      bordered
      modal={false}
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
      slotHeading={
        <Flex align="center" className="relative w-full">
          <Text kind="label/bold/lg">Test Results</Text>
          {/* Absolutely centered rather than laid out in flow, so the controls do
              not shift as the "N of M" label changes width. */}
          {visibleIndex !== null && (
            <Flex align="center" gap="density-xs" className="-translate-x-1/2 absolute left-1/2">
              <Button
                kind="tertiary"
                aria-label="Previous test"
                disabled={visibleIndex <= 0}
                onClick={() => onNavigate(visibleIndex - 1)}
              >
                <ChevronLeft size="16" />
              </Button>
              <Text kind="body/regular/md" className="tabular-nums whitespace-nowrap">
                {visibleIndex + 1} of {visibleCount}
              </Text>
              <Button
                kind="tertiary"
                aria-label="Next test"
                disabled={visibleIndex >= visibleCount - 1}
                onClick={() => onNavigate(visibleIndex + 1)}
              >
                <ChevronRight size="16" />
              </Button>
            </Flex>
          )}
        </Flex>
      }
      attributes={{
        SidePanelMain: { className: 'p-0 overflow-hidden' },
      }}
    >
      <Flex className="h-full min-h-0 divide-x divide-base overflow-hidden">
        <ConversationPane check={check} className="w-1/2 min-w-0 shrink-0" />
        <ResultsPane
          check={check}
          configData={configData}
          checkIndex={checkIndex}
          className="w-1/2 min-w-0 shrink-0"
        />
      </Flex>
    </SidePanel>
  );
};
