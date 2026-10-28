// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ComposerMode } from '@nemo/common/src/components/AssistantChat';
import type { BroadcastSignal } from '@nemo/common/src/components/AssistantChat/types';
import { ChatMulti, ChatSingle } from '@nemo/icons/icons';
import { Tooltip } from '@nvidia/foundations-react-core';
import { CompareComposer } from '@studio/components/chat/CompareComposer';
import { DEFAULT_SEED_QUESTIONS } from '@studio/components/chat/defaultSeedQuestions';
import { ModelCompareChat } from '@studio/components/ModelCompareChat';
import type { ComposerSeed, SharedModelEntry } from '@studio/routes/ModelCompareRoute/types';
import { type FC, useCallback, useRef, useState } from 'react';

export interface ComparePlaygroundProps {
  workspace: string;
  models: SharedModelEntry[];
  onRemoveModel: (id: number) => void;
  onSetModel: (id: number, modelURN: string | null, adapter?: string | null) => void;
  /** Omit to hide the Add model button. */
  onAddModel?: () => void;
}

/** Chat with several models side by side, through one shared composer or one per panel. */
export const ComparePlayground: FC<ComparePlaygroundProps> = ({
  workspace,
  models,
  onRemoveModel,
  onSetModel,
  onAddModel,
}) => {
  const [perPanelInput, setPerPanelInput] = useState(false);

  // Seed transfer: broadcast→panels and panel→broadcast.
  const compareComposerDraftRef = useRef('');
  const [panelSeed, setPanelSeed] = useState<ComposerSeed | null>(null);
  const [composerSeed, setComposerSeed] = useState<ComposerSeed | null>(null);

  // Compare-mode plumbing: broadcast carries the prompt to every panel via
  // sequence-keyed effect; stopCount flips to stop them all.
  const [chatResetCount, setChatResetCount] = useState(0);
  const [broadcast, setBroadcast] = useState<BroadcastSignal | null>(null);
  const [stopCount, setStopCount] = useState(0);
  const [runningById, setRunningById] = useState<Map<number, boolean>>(() => new Map());
  // Read through the open panels only, so a removed panel's last report doesn't linger.
  const isAnyRunning = models.some((m) => runningById.get(m.id));
  // Per-panel empty/non-empty state. Drives seed-chip visibility: the broadcast
  // composer only shows chips when every open panel is in its empty initial
  // state. Panels default to empty until they report otherwise.
  const [emptyById, setEmptyById] = useState<Map<number, boolean>>(() => new Map());
  const allPanelsEmpty = models.every((m) => emptyById.get(m.id) ?? true);

  const handleRunningChange = useCallback((id: number, running: boolean) => {
    setRunningById((prev) => {
      if (prev.get(id) === running) return prev;
      const next = new Map(prev);
      next.set(id, running);
      return next;
    });
  }, []);

  const handleEmptyChange = useCallback((id: number, empty: boolean) => {
    setEmptyById((prev) => {
      if (prev.get(id) === empty) return prev;
      const next = new Map(prev);
      next.set(id, empty);
      return next;
    });
  }, []);

  const resetAll = useCallback(() => {
    setStopCount((n) => n + 1); // cancel in-flight completions before remount
    setBroadcast(null);
    setRunningById(new Map());
    setChatResetCount((n) => n + 1);
  }, []);

  const handleBroadcast = useCallback((text: string) => {
    setBroadcast((prev) => ({ seq: (prev?.seq ?? 0) + 1, text }));
  }, []);

  const handleStopAll = useCallback(() => {
    setStopCount((n) => n + 1);
  }, []);

  const readyPanelCount = models.filter((m) => !!m.modelURN).length;

  return (
    <ModelCompareChat
      workspace={workspace}
      models={models}
      onRemoveModel={onRemoveModel}
      onSetModel={onSetModel}
      chatResetCount={chatResetCount}
      composerMode={perPanelInput ? ComposerMode.PER_PANEL : ComposerMode.BROADCAST_ALL}
      slotComposerEnd={
        perPanelInput ? (
          <Tooltip slotContent="Shared input">
            <button
              onClick={(e) => {
                const panel = (e.currentTarget as HTMLElement).closest('[data-model-panel]');
                const textarea = panel?.querySelector<HTMLTextAreaElement>(
                  'textarea[aria-label="Task prompt"]'
                );
                const draft = textarea?.value ?? '';
                if (draft) {
                  setComposerSeed((prev) => ({
                    triggerCount: (prev?.triggerCount ?? 0) + 1,
                    text: draft,
                  }));
                }
                setPerPanelInput(false);
              }}
              className="flex cursor-pointer items-center justify-center rounded p-1.5 text-fg-subdued transition-colors hover:bg-surface-sunken hover:text-fg-base"
              aria-label="Shared input"
            >
              <ChatSingle size="15" />
            </button>
          </Tooltip>
        ) : undefined
      }
      composerSeed={panelSeed ?? undefined}
      broadcast={!perPanelInput ? (broadcast ?? undefined) : undefined}
      stopCount={stopCount}
      onRunningChange={handleRunningChange}
      onEmptyChange={handleEmptyChange}
      onAddModel={onAddModel}
      seedQuestions={DEFAULT_SEED_QUESTIONS}
      slotComposer={
        !perPanelInput ? (
          <CompareComposer
            isAnyRunning={isAnyRunning}
            readyPanelCount={readyPanelCount}
            totalPanelCount={models.length}
            onSubmit={handleBroadcast}
            onStop={handleStopAll}
            onResetAll={resetAll}
            seedQuestions={allPanelsEmpty ? DEFAULT_SEED_QUESTIONS : []}
            draftRef={compareComposerDraftRef}
            seed={composerSeed ?? undefined}
            slotSeedEnd={
              <Tooltip slotContent="Per-panel input">
                <button
                  onClick={() => {
                    const draft = compareComposerDraftRef.current;
                    if (draft) {
                      setPanelSeed((prev) => ({
                        triggerCount: (prev?.triggerCount ?? 0) + 1,
                        text: draft,
                      }));
                    }
                    setComposerSeed(null);
                    setPerPanelInput(true);
                  }}
                  className="flex cursor-pointer items-center justify-center rounded border border-base bg-surface-raised p-1.5 text-fg-subdued transition-colors hover:bg-surface-sunken hover:text-fg-base"
                  aria-label="Per-panel input"
                >
                  <ChatMulti size="15" />
                </button>
              </Tooltip>
            }
          />
        ) : undefined
      }
    />
  );
};
