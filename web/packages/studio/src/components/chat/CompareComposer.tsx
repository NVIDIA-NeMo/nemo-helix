// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ArrowUp, Retry, ShapeSquare } from '@nemo/icons/icons';
import { Button } from '@nvidia/foundations-react-core';
import { SeedQuestions } from '@studio/components/chat/SeedQuestions';
import type { ComposerSeed } from '@studio/routes/ModelCompareRoute/types';
import * as React from 'react';
import {
  type FC,
  type MutableRefObject,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

interface CompareComposerProps {
  /** Any panel currently streaming? Switches the Send button into a Stop button. */
  isAnyRunning: boolean;
  /** Number of panels with a model selected — used for the placeholder + disable state. */
  readyPanelCount: number;
  /** Total panel count — used to phrase the placeholder when models are missing. */
  totalPanelCount: number;
  onSubmit: (text: string) => void;
  onStop: () => void;
  /** Clears all panel histories. */
  onResetAll: () => void;
  /** Suggested prompts rendered above the input INSIDE the same composer card.
   *  Clicking a chip fills the draft but does NOT auto-submit — preserves the
   *  "send happens on the green button" mental model. */
  seedQuestions?: string[];
  /** Rendered right-aligned at the trailing end of the seed-questions row. */
  slotSeedEnd?: ReactNode;
  /** Kept in sync with internal draft so callers can read it imperatively. */
  draftRef?: MutableRefObject<string>;
  /** When triggerCount changes, resets the draft to text (panel→broadcast transfer). */
  seed?: ComposerSeed;
}

/**
 * Page-level composer shown only in Compare mode. Mirrors the per-panel
 * AssistantComposer's single-card layout: seeds sit in an attached sub-row
 * above the input, separated by a thin internal divider, all inside one
 * rounded border.
 */
export const CompareComposer: FC<CompareComposerProps> = ({
  isAnyRunning,
  readyPanelCount,
  totalPanelCount,
  onSubmit,
  onStop,
  onResetAll,
  seedQuestions,
  slotSeedEnd,
  draftRef,
  seed,
}) => {
  const [draft, setDraft] = useState('');

  // Keep caller's ref in sync so they can read the current draft imperatively.
  if (draftRef) draftRef.current = draft;

  // Pre-fill from panel toggle transfer (panel→broadcast).
  const seenSeedTriggerRef = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!seed?.text || seed.triggerCount === seenSeedTriggerRef.current) return;
    seenSeedTriggerRef.current = seed.triggerCount;
    setDraft(seed.text);
  }, [seed?.triggerCount, seed?.text]);

  const canSend = !isAnyRunning && readyPanelCount > 0 && draft.trim().length > 0;

  const handleSubmit = useCallback(() => {
    if (!canSend) return;
    onSubmit(draft.trim());
    setDraft('');
  }, [canSend, draft, onSubmit]);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const placeholder =
    readyPanelCount === 0
      ? totalPanelCount > 0
        ? 'Pick a model in each panel to broadcast a prompt…'
        : 'Add panels and pick models to broadcast…'
      : `Broadcast to ${readyPanelCount} of ${totalPanelCount} panel${
          totalPanelCount === 1 ? '' : 's'
        }…`;

  const showSeeds = !!seedQuestions && seedQuestions.length > 0;

  return (
    <div className="flex flex-col gap-2">
      {(showSeeds || slotSeedEnd) && (
        <SeedQuestions
          questions={showSeeds ? seedQuestions : []}
          onSelect={(text) => setDraft(text)}
          disabled={readyPanelCount === 0}
          slotEnd={slotSeedEnd}
        />
      )}
      <div className="flex items-center gap-2 rounded-md border border-base bg-surface-base px-3 py-1.5 focus-within:border-emphasis">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          rows={1}
          disabled={readyPanelCount === 0}
          aria-label="Compare prompt"
          className="placeholder:text-fg-subdued min-h-[24px] max-h-32 flex-1 resize-none border-0 bg-transparent text-sm leading-6 outline-none disabled:cursor-not-allowed disabled:text-fg-disabled"
        />
        <Button
          kind="tertiary"
          size="small"
          onClick={onResetAll}
          title="Clear all panels"
          aria-label="Clear all panels"
        >
          <Retry />
        </Button>
        {isAnyRunning ? (
          <Button
            color="danger"
            size="small"
            className="size-8 rounded-full p-0"
            onClick={onStop}
            title="Stop all panels"
            aria-label="Stop all panels"
          >
            <ShapeSquare size="14" />
          </Button>
        ) : (
          <Button
            color="brand"
            size="small"
            className="size-8 rounded-full p-0"
            onClick={handleSubmit}
            disabled={!canSend}
            title="Broadcast to all panels"
            aria-label="Broadcast to all panels"
          >
            <ArrowUp size="16" />
          </Button>
        )}
      </div>
    </div>
  );
};
