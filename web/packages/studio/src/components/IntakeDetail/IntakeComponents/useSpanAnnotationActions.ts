// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import {
  getListAnnotationsQueryKey,
  listAnnotations,
  useCreateAnnotation,
  useDeleteAnnotation,
} from '@nemo/sdk/generated/platform/annotations';
import {
  AnnotationKind,
  FeedbackAnnotationInputKind,
  type FeedbackAnnotationInputValue,
  NoteAnnotationInputKind,
} from '@nemo/sdk/generated/platform/schema';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useState } from 'react';

interface SpanAnnotationActions {
  /** Create (or replace) the span's feedback sentiment. */
  submitFeedback: (value: FeedbackAnnotationInputValue) => Promise<boolean>;
  /** Add a free-text note to the span. */
  submitNote: (text: string) => Promise<boolean>;
  /** Remove an annotation by id. */
  deleteAnnotation: (annotationId: string) => Promise<boolean>;
  /** True while any annotation mutation is in flight. */
  isMutating: boolean;
  /** Last mutation error message, if any. */
  error?: string;
  clearError: () => void;
}

const errorMessage = (error: unknown, fallback: string): string =>
  error instanceof Error ? getErrorMessage(error, fallback) : fallback;

/**
 * Annotation create/delete mutations for a single span, shared by the full
 * annotations panel and the compact feedback controls in the span header. It
 * owns no list query — both consumers read annotations from their own queries
 * and these mutations invalidate the whole annotations list to refresh them.
 */
export const useSpanAnnotationActions = (
  workspace: string,
  spanId: string,
  sessionId: string
): SpanAnnotationActions => {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | undefined>();
  const createAnnotation = useCreateAnnotation();
  const deleteMutation = useDeleteAnnotation();

  const refresh = useCallback(
    () => queryClient.invalidateQueries({ queryKey: getListAnnotationsQueryKey(workspace) }),
    [queryClient, workspace]
  );

  const submitFeedback = useCallback(
    async (value: FeedbackAnnotationInputValue): Promise<boolean> => {
      setError(undefined);
      try {
        // Feedback is single-valued per span, but the API always inserts a new
        // row with a fresh id. Remove any existing feedback first so
        // re-submitting replaces the sentiment instead of stacking duplicate
        // rows (which would also inflate the annotation count).
        const existing = await listAnnotations(workspace, {
          page: 1,
          page_size: 100,
          filter: { span_id: spanId, kind: AnnotationKind.feedback },
        });
        await Promise.all(
          existing.data
            .filter((annotation) => annotation.kind === AnnotationKind.feedback)
            .map((annotation) =>
              deleteMutation.mutateAsync({ workspace, annotationId: annotation.annotation_id })
            )
        );
        await createAnnotation.mutateAsync({
          workspace,
          data: {
            kind: FeedbackAnnotationInputKind.feedback,
            value,
            session_id: sessionId,
            span_id: spanId,
          },
        });
        await refresh();
        return true;
      } catch (caught) {
        setError(errorMessage(caught, 'Failed to save feedback.'));
        return false;
      }
    },
    [createAnnotation, deleteMutation, workspace, sessionId, spanId, refresh]
  );

  const submitNote = useCallback(
    async (text: string): Promise<boolean> => {
      setError(undefined);
      try {
        await createAnnotation.mutateAsync({
          workspace,
          data: {
            kind: NoteAnnotationInputKind.note,
            text,
            session_id: sessionId,
            span_id: spanId,
          },
        });
        await refresh();
        return true;
      } catch (caught) {
        setError(errorMessage(caught, 'Failed to save note.'));
        return false;
      }
    },
    [createAnnotation, workspace, sessionId, spanId, refresh]
  );

  const deleteAnnotation = useCallback(
    async (annotationId: string): Promise<boolean> => {
      setError(undefined);
      try {
        await deleteMutation.mutateAsync({ workspace, annotationId });
        await refresh();
        return true;
      } catch (caught) {
        setError(errorMessage(caught, 'Failed to delete annotation.'));
        return false;
      }
    },
    [deleteMutation, workspace, refresh]
  );

  return {
    submitFeedback,
    submitNote,
    deleteAnnotation,
    isMutating: createAnnotation.isPending || deleteMutation.isPending,
    error,
    clearError: useCallback(() => setError(undefined), []),
  };
};
