// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  AssistantChat,
  ComposerMode,
  type AssistantChatProps,
} from '@nemo/common/src/components/AssistantChat';
import type { AssistantMessageCompletion } from '@nemo/common/src/components/AssistantChat/types';
import type { ResourceRef } from '@nemo/common/src/types';
import { handleGenericError } from '@nemo/common/src/utils/logger';
import type { ModelChatStatus } from '@nemo/common/src/utils/models';
import { DEFAULT_SEED_QUESTIONS } from '@studio/components/chat/defaultSeedQuestions';
import { SeedQuestions } from '@studio/components/chat/SeedQuestions';
import { StatsBadge, type ChatMetrics } from '@studio/components/chat/StatsBadge';
import { DeployModelCta } from '@studio/components/ModelChat/DeployModelCta';
import type { ComposerSeed } from '@studio/routes/ModelCompareRoute/types';
import { type ReactNode, useEffect, useRef, useState, type FC } from 'react';

interface ModelChatProps extends Pick<
  AssistantChatProps,
  | 'model'
  | 'workspace'
  | 'baseURL'
  | 'promptData'
  | 'tools'
  | 'assistantName'
  | 'placeholder'
  | 'disabled'
  | 'className'
  | 'initialMessages'
  | 'emptyState'
  | 'onError'
  | 'composerMode'
  | 'broadcast'
  | 'stopCount'
  | 'onRunningChange'
> {
  /**
   * When provided, ModelChat derives default `disabled` state and a
   * status-aware empty state ("Chat Unavailable" / "Model Deployment in
   * Progress") from this status. Explicit `disabled` and `emptyState` take
   * precedence.
   */
  modelChatStatus?: ModelChatStatus;
  /** When set, renders the suggestion-chip strip above the composer when there
   *  are no messages yet. Clicking a chip seeds the composer (using the
   *  AssistantChat composer set-input API via a small DOM bridge). */
  seedQuestions?: string[];
  /** When false, hides the per-response StatsBadge. Default true. */
  showMetrics?: boolean;
  /** Rendered right-aligned at the trailing end of the seed-questions row. */
  slotComposerEnd?: ReactNode;
  /** When triggerCount changes, pre-fills the panel's composer textarea with text. */
  composerSeed?: ComposerSeed;
  /** Fires when this panel's thread transitions between empty and non-empty. */
  onEmptyChange?: (isEmpty: boolean) => void;
  /**
   * `<workspace>/<name>` of the model entity behind this chat. When set and the
   * status is `disabled`, the empty state offers a "Deploy this model" action.
   * For an adapter's chat this is the *base model* — deploying it is what makes
   * the adapter servable.
   */
  deployModelRef?: ResourceRef;
  /** Overrides the deploy button's copy. Defaults to "Deploy this model". */
  deployModelLabel?: string;
}

function statusEmptyState(
  status: Exclude<ModelChatStatus, 'enabled'>,
  deployModelRef?: ResourceRef,
  deployModelLabel?: string
): NonNullable<AssistantChatProps['emptyState']> {
  if (status === 'pending') {
    return {
      slotHeading: 'Model Deployment in Progress',
      slotSubheading: 'Check back in a few minutes to chat with this model.',
    };
  }
  return {
    slotHeading: 'Chat Unavailable',
    slotSubheading: 'This model does not have an active deployment.',
    // Only `disabled` gets the CTA: `pending` already has a deployment coming up.
    slotAction: deployModelRef ? (
      <DeployModelCta modelRef={deployModelRef} label={deployModelLabel} />
    ) : undefined,
  };
}

export const ModelChat: FC<ModelChatProps> = ({
  model,
  modelChatStatus,
  disabled,
  assistantName,
  emptyState,
  onError,
  promptData,
  seedQuestions = DEFAULT_SEED_QUESTIONS,
  showMetrics = true,
  slotComposerEnd,
  composerSeed,
  onEmptyChange,
  workspace,
  deployModelRef,
  deployModelLabel,
  ...rest
}) => {
  const resolvedDisabled = disabled ?? (modelChatStatus ? modelChatStatus !== 'enabled' : false);
  const statusDerivedEmptyState =
    disabled === undefined && modelChatStatus && modelChatStatus !== 'enabled'
      ? statusEmptyState(modelChatStatus, deployModelRef, deployModelLabel)
      : undefined;
  // In broadcast-all mode the page-level composer is the affordance, so the
  // per-panel subhead "Prompt your model to get started." is redundant.
  const compareEmptyState =
    rest.composerMode === ComposerMode.BROADCAST_ALL
      ? { slotHeading: 'Ready', slotSubheading: '' }
      : undefined;
  const resolvedEmptyState = emptyState ?? statusDerivedEmptyState ?? compareEmptyState;

  // Per-message metrics: store the latest completion so a single StatsBadge
  // can render under the chat surface.
  const [latestMetrics, setLatestMetrics] = useState<ChatMetrics | null>(null);

  // Clear stale metrics when inference identity changes (different model or workspace).
  const prevModelRef = useRef(model);
  const prevWorkspaceRef = useRef(workspace);
  if (model !== prevModelRef.current || workspace !== prevWorkspaceRef.current) {
    prevModelRef.current = model;
    prevWorkspaceRef.current = workspace;
    if (latestMetrics !== null) setLatestMetrics(null);
  }

  const handleMessageComplete = (info: AssistantMessageCompletion) => {
    setLatestMetrics({
      totalMs: info.totalMs,
      completionTokens: info.completionTokens,
      tokensPerSec: info.tokensPerSec,
    });
  };

  const containerRef = useRef<HTMLDivElement>(null);

  // Scoped textarea setter — finds the panel's own composer, not any global one.
  // Kept DOM-based until AssistantChat exposes a proper setInput API.
  const setComposerText = (text: string) => {
    if (typeof document === 'undefined' || !containerRef.current) return;
    const textarea = containerRef.current.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="Task prompt"]'
    );
    if (!textarea) return;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
    setter?.call(textarea, text);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.focus();
  };

  const seedComposer = (text: string) => setComposerText(text);

  // Pre-fill from parent (mode toggle transfer: broadcast→panels).
  // Intentionally omits composerSeed.text: only fire when triggerCount changes.
  useEffect(() => {
    if (composerSeed?.text) setComposerText(composerSeed.text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [composerSeed?.triggerCount]);

  const isBroadcastAll = rest.composerMode === ComposerMode.BROADCAST_ALL;

  // Track empty/non-empty so per-panel seeds only show in an empty thread.
  const [isEmpty, setIsEmpty] = useState(true);
  const handleEmptyChange = (empty: boolean) => {
    setIsEmpty(empty);
    onEmptyChange?.(empty);
    // Reset clears the thread → drop the stale metrics badge too.
    if (empty) setLatestMetrics(null);
  };

  // Seeds render inside the composer card. In broadcast-all mode the page-level
  // CompareComposer owns seeds, so suppress them here. Only show in an empty thread.
  const showChatSeeds = !!seedQuestions && seedQuestions.length > 0 && !isBroadcastAll && isEmpty;

  // In per-panel mode: metrics sit above seeds in slotComposerStart.
  // In broadcast-all mode: slotComposerStart is hidden, so metrics fall below.
  const metricsInComposer = showMetrics && latestMetrics && !isBroadcastAll;
  const metricsBelow = showMetrics && latestMetrics && isBroadcastAll;

  const chatSeedSlot =
    showChatSeeds || metricsInComposer || (slotComposerEnd && !isBroadcastAll) ? (
      <SeedQuestions
        questions={showChatSeeds ? seedQuestions : []}
        onSelect={seedComposer}
        disabled={resolvedDisabled}
        slotStart={
          metricsInComposer && latestMetrics ? <StatsBadge metrics={latestMetrics} /> : undefined
        }
        slotEnd={slotComposerEnd}
      />
    ) : undefined;

  return (
    <div ref={containerRef} className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1">
        <AssistantChat
          model={model}
          workspace={workspace}
          assistantName={assistantName ?? model}
          disabled={resolvedDisabled}
          emptyState={resolvedEmptyState}
          onError={onError ?? handleGenericError}
          promptData={promptData}
          onMessageComplete={handleMessageComplete}
          onEmptyChange={handleEmptyChange}
          {...rest}
          slotComposerStart={chatSeedSlot}
        />
      </div>
      {metricsBelow && (
        <div className="shrink-0 px-3 pt-1">
          <StatsBadge metrics={latestMetrics} />
        </div>
      )}
    </div>
  );
};
