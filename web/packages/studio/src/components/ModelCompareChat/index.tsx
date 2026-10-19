// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tooltip } from '@nvidia/foundations-react-core';
import { ModelChatPanel } from '@studio/components/ModelChatPanel';
import {
  PANEL_ROLE_COLORS,
  PANEL_ROLE_LABELS,
  type PanelChatControls,
  type PanelState,
  type SharedModelEntry,
} from '@studio/routes/ModelCompareRoute/types';
import { Plus } from 'lucide-react';
import { useCallback, useState, type FC, type ReactNode } from 'react';

interface ModelCompareChatProps extends PanelChatControls {
  /** Route workspace — used only as a fallback for panels without an assigned model. */
  workspace: string;
  models: SharedModelEntry[];
  onRemoveModel: (id: number) => void;
  onSetModel: (id: number, modelURN: string | null, adapter?: string | null) => void;
  /** Incremented to remount all chat panels (clears messages) without losing model selections. */
  chatResetCount?: number;
  /** Called when the user clicks the Add Model button. Omit to hide the button (gutter stays). */
  onAddModel?: () => void;
  /** Broadcast composer, rendered below the panels and aligned to the panels' width. */
  slotComposer?: ReactNode;
}

export const ModelCompareChat: FC<ModelCompareChatProps> = ({
  workspace,
  models,
  onRemoveModel,
  onSetModel,
  chatResetCount,
  composerMode,
  broadcast,
  stopCount,
  onRunningChange,
  onEmptyChange,
  slotComposerEnd,
  composerSeed,
  seedQuestions,
  onAddModel,
  slotComposer,
}) => {
  // Per-panel UI state that's view-local (doesn't cross over to Prompts)
  const [collapsedIds, setCollapsedIds] = useState<Set<number>>(new Set());

  const togglePanel = useCallback((id: number) => {
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const isSinglePanel = models.length === 1;

  // Compose PanelState per-render from shared entry + per-view ephemeral state
  // + position-derived role.
  const panels: PanelState[] = models.map((m, idx) => {
    const roleColor = PANEL_ROLE_COLORS[Math.min(idx, PANEL_ROLE_COLORS.length - 1)];
    return {
      id: m.id,
      collapsed: collapsedIds.has(m.id),
      modelURN: m.modelURN,
      adapter: m.adapter,
      roleColor,
      roleLabel: PANEL_ROLE_LABELS[roleColor],
      isSinglePanel,
      locked: !!m.locked,
    };
  });

  // The right gutter is always present so the panels' scroll area keeps a fixed
  // right edge: at max panels (no add button) it holds an invisible placeholder
  // of the same width, and panels overflow-scroll underneath it.
  const gutter = (
    <div className="flex shrink-0 items-start pr-6 pt-2">
      {onAddModel ? (
        <Tooltip slotContent="Add model">
          <button
            onClick={onAddModel}
            className="flex cursor-pointer items-center justify-center rounded border border-base bg-surface-raised p-1.5 text-fg-subdued transition-colors hover:bg-surface-sunken hover:text-fg-base"
            aria-label="Add model"
          >
            <Plus size={16} />
          </button>
        </Tooltip>
      ) : (
        <span
          aria-hidden
          className="invisible flex items-center justify-center rounded border border-base p-1.5"
        >
          <Plus size={16} />
        </span>
      )}
    </div>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        <div className="flex min-h-0 flex-1 gap-3 overflow-x-auto pl-6 pr-2 pt-2 pb-2">
          {panels.map((panel) => (
            <ModelChatPanel
              key={`${panel.id}-${chatResetCount ?? 0}`}
              panel={panel}
              fallbackWorkspace={workspace}
              onToggle={togglePanel}
              onRemove={onRemoveModel}
              onModelChange={onSetModel}
              hideRemove={panel.locked || models.length <= 1}
              composerMode={composerMode}
              slotComposerEnd={slotComposerEnd}
              composerSeed={composerSeed}
              broadcast={broadcast}
              stopCount={stopCount}
              onRunningChange={onRunningChange}
              onEmptyChange={onEmptyChange}
              seedQuestions={seedQuestions}
            />
          ))}
        </div>
        {gutter}
      </div>
      {slotComposer && (
        <div className="flex">
          <div className="min-w-0 flex-1 pl-6 pr-2 pb-3">{slotComposer}</div>
          {/* Invisible gutter copy keeps the composer's right edge aligned to the panels. */}
          <div className="invisible flex shrink-0 items-start pr-6" aria-hidden>
            <span className="flex items-center justify-center rounded border border-base p-1.5">
              <Plus size={16} />
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
