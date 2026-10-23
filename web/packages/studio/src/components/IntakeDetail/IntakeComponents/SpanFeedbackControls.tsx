// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Notes, ThumbDown, ThumbUp } from '@nemo/icons/icons';
import { FeedbackAnnotationInputValue } from '@nemo/sdk/generated/platform/schema';
import { Button, Divider, Flex, Tooltip } from '@nvidia/foundations-react-core';
import { useSpanAnnotationActions } from '@studio/components/IntakeDetail/IntakeComponents/useSpanAnnotationActions';
import { type FC, type MouseEvent, useEffect } from 'react';

interface SpanFeedbackControlsProps {
  workspace: string;
  spanId: string;
  sessionId: string;
  /** Current feedback sentiment for the span, to highlight the active thumb. */
  activeFeedback?: FeedbackAnnotationInputValue;
  /** Whether the span has any note annotations, to highlight the note icon. */
  hasNotes?: boolean;
  /** Open the span and focus its annotations note field (handled by the parent). */
  onAddNote: () => void;
}

/**
 * Compact feedback controls for a span header: thumbs up/down plus an "add note"
 * button. The header may be a `<summary>` (the accordion row trigger), so each
 * click stops propagation and the default toggle. Feedback mutations are shared
 * with the full annotations panel via {@link useSpanAnnotationActions}; adding a
 * note is delegated upward so the parent can reveal the annotations panel.
 */
export const SpanFeedbackControls: FC<SpanFeedbackControlsProps> = ({
  workspace,
  spanId,
  sessionId,
  activeFeedback,
  hasNotes,
  onAddNote,
}) => {
  const { submitFeedback, isMutating, error, clearError } = useSpanAnnotationActions(
    workspace,
    spanId,
    sessionId
  );
  const toast = useToast();

  // Surface a failed feedback mutation; clear it so the same error doesn't re-toast.
  useEffect(() => {
    if (error) {
      toast.error(error);
      clearError();
    }
  }, [error, toast, clearError]);

  const positive = activeFeedback === FeedbackAnnotationInputValue.positive;
  const negative = activeFeedback === FeedbackAnnotationInputValue.negative;

  // The trigger row is a <summary>; keep clicks from toggling/collapsing it.
  const withoutToggle = (handler: () => void) => (event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    handler();
  };

  return (
    <Flex align="center" gap="density-xs" className="shrink-0">
      {/* Separate the span metadata (tokens/cost/duration) from the annotation
          controls. */}
      <Divider orientation="vertical" className="mr-density-xs h-4 self-center" />
      <Tooltip slotContent="Positive feedback" side="top">
        <Button
          type="button"
          size="tiny"
          kind="tertiary"
          color={positive ? 'brand' : 'neutral'}
          aria-label="Positive feedback"
          aria-pressed={positive}
          disabled={isMutating}
          onClick={withoutToggle(() => void submitFeedback(FeedbackAnnotationInputValue.positive))}
        >
          <ThumbUp
            size="14"
            aria-hidden
            className={positive ? 'text-[color:var(--text-color-accent-green)]' : undefined}
          />
        </Button>
      </Tooltip>
      <Tooltip slotContent="Negative feedback" side="top">
        <Button
          type="button"
          size="tiny"
          kind="tertiary"
          color={negative ? 'danger' : 'neutral'}
          aria-label="Negative feedback"
          aria-pressed={negative}
          disabled={isMutating}
          onClick={withoutToggle(() => void submitFeedback(FeedbackAnnotationInputValue.negative))}
        >
          <ThumbDown
            size="14"
            aria-hidden
            className={negative ? 'text-[color:var(--text-color-accent-red)]' : undefined}
          />
        </Button>
      </Tooltip>
      <Tooltip slotContent={hasNotes ? 'Notes added' : 'Add note'} side="top">
        <Button
          type="button"
          size="tiny"
          kind="tertiary"
          aria-label={hasNotes ? 'Add note (this span has notes)' : 'Add note'}
          onClick={withoutToggle(onAddNote)}
        >
          {/* Tinted to match the Guardrail kind accent (var --text-color-accent-yellow). */}
          <Notes
            size="14"
            aria-hidden
            className={hasNotes ? 'text-[color:var(--text-color-accent-yellow)]' : undefined}
          />
        </Button>
      </Tooltip>
    </Flex>
  );
};
