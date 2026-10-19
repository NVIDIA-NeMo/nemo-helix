// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { DEFAULT_PAGE_SIZE } from '@nemo/common/src/constants/api';
import { useListAnnotations } from '@nemo/sdk/generated/platform/annotations';
import {
  AnnotationSortField,
  FeedbackAnnotationInputValue,
  type Annotation,
} from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  CodeSnippet,
  Flex,
  FormField,
  Stack,
  Text,
  TextArea,
} from '@nvidia/foundations-react-core';
import { ThumbButton } from '@studio/components/buttons/ThumbButton';
import { useSpanAnnotationActions } from '@studio/components/IntakeDetail/IntakeComponents/useSpanAnnotationActions';
import { MessageSquarePlus, Trash2 } from 'lucide-react';
import { type FC, useEffect, useMemo, useRef } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { z } from 'zod';

// A note is optional — no minimum length is enforced.
const noteSchema = z.object({
  text: z.string().trim(),
});

type NoteFormValues = z.infer<typeof noteSchema>;

const getAnnotationErrorMessage = (error: unknown, fallback: string): string =>
  error instanceof Error ? getErrorMessage(error, fallback) : fallback;

const formatAnnotationTitle = (annotation: Annotation): string => {
  switch (annotation.kind) {
    case 'feedback':
      return annotation.value === 'positive' ? 'Positive feedback' : 'Negative feedback';
    case 'note':
      return 'Note';
    case 'label':
      return annotation.name ? `Label: ${annotation.name}` : 'Label';
    case 'metadata':
      return 'Metadata';
  }
};

const renderAnnotationBody = (annotation: Annotation) => {
  switch (annotation.kind) {
    case 'feedback':
      return (
        <Text kind="body/regular/sm" className="capitalize">
          {annotation.value}
        </Text>
      );
    case 'note':
      return (
        <Text kind="body/regular/sm" className="whitespace-pre-wrap break-words">
          {annotation.text}
        </Text>
      );
    case 'label':
      return (
        <Text kind="body/regular/sm" className="break-words">
          {String(annotation.value)}
        </Text>
      );
    case 'metadata':
      return (
        <CodeSnippet
          value={JSON.stringify(annotation.metadata, null, 2)}
          language="json"
          kind="block"
          collapsible
          defaultOpen={false}
          attributes={{
            CodeSnippetCode: {
              className:
                'max-h-[220px] [&_code]:whitespace-pre-wrap [&_code]:break-words [&_pre]:whitespace-pre-wrap',
            },
          }}
        />
      );
  }
};

export interface AnnotationsPanelProps {
  workspace: string;
  spanId: string;
  sessionId: string;
  /** Bumped to scroll the note field into view and focus it. */
  focusNonce?: number;
}

export const AnnotationsPanel: FC<AnnotationsPanelProps> = ({
  workspace,
  spanId,
  sessionId,
  focusNonce,
}) => {
  const {
    register,
    handleSubmit,
    reset,
    watch,
    formState: { errors, isValid },
  } = useForm<NoteFormValues>({
    resolver: zodResolver(noteSchema),
    defaultValues: { text: '' },
    mode: 'onChange',
  });

  const listParams = useMemo(
    () => ({
      page: 1,
      page_size: DEFAULT_PAGE_SIZE,
      sort: AnnotationSortField['-created_at'],
      filter: {
        span_id: spanId,
      },
    }),
    [spanId]
  );

  const {
    data: annotationsResponse,
    error: listError,
    isLoading,
  } = useListAnnotations(workspace, listParams);
  const {
    submitFeedback,
    submitNote,
    deleteAnnotation,
    isMutating,
    error: mutationError,
  } = useSpanAnnotationActions(workspace, spanId, sessionId);

  const noteText = watch('text');
  const annotations = annotationsResponse?.data ?? [];
  const activeFeedback = annotations.find((annotation) => annotation.kind === 'feedback');

  const handleNoteSubmit: SubmitHandler<NoteFormValues> = async ({ text }) => {
    if (await submitNote(text)) {
      reset();
    }
  };

  // Focus the note field when an external "add note" action targets this span.
  const noteFieldId = `intake-note-field-${spanId}`;
  const prevFocus = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (focusNonce === undefined || focusNonce === prevFocus.current) return;
    prevFocus.current = focusNonce;
    // Defer a frame so a just-opened accordion section has laid out first.
    const frame = requestAnimationFrame(() => {
      const field = document.getElementById(noteFieldId);
      field?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      (field as HTMLTextAreaElement | null)?.focus();
    });
    return () => cancelAnimationFrame(frame);
  }, [focusNonce, noteFieldId]);

  return (
    <Stack gap="density-xl" className="min-w-0">
      <Stack gap="density-md" className="min-w-0">
        {isLoading ? (
          <Text kind="body/regular/sm" className="text-secondary">
            Loading annotations...
          </Text>
        ) : annotations.length === 0 ? (
          <Text kind="body/regular/sm" className="text-secondary">
            No annotations yet.
          </Text>
        ) : (
          annotations.map((annotation) => (
            <Flex
              key={annotation.annotation_id}
              role="article"
              aria-label={`${formatAnnotationTitle(annotation)} annotation`}
              align="start"
              gap="density-md"
              className="min-w-0"
            >
              {/* Fixed-width date column so every annotation body starts on one
                  line; date and delete stay pinned to the top while a note wraps. */}
              <Text
                kind="body/regular/xs"
                className="w-[13rem] shrink-0 whitespace-nowrap pt-density-xxs text-secondary"
              >
                {formatAbsoluteTimestamp(annotation.created_at)}
              </Text>
              <div className="min-w-0 flex-1">{renderAnnotationBody(annotation)}</div>
              <Button
                size="small"
                kind="tertiary"
                aria-label="Delete annotation"
                disabled={isMutating}
                onClick={() => void deleteAnnotation(annotation.annotation_id)}
                className="shrink-0"
              >
                <Trash2 />
              </Button>
            </Flex>
          ))
        )}
      </Stack>

      <Flex gap="density-lg" align="start" wrap="wrap" className="min-w-0">
        <form
          className="min-w-0 flex-1"
          onSubmit={(event) => void handleSubmit(handleNoteSubmit)(event)}
        >
          <Stack gap="density-sm">
            <FormField
              name="annotation-note"
              slotLabel="Notes"
              slotError={errors.text?.message}
              status={errors.text && 'error'}
            >
              <TextArea
                value={noteText}
                placeholder="Add a note about this span."
                disabled={isMutating}
                status={errors.text && 'error'}
                resizeable="manual"
                attributes={{
                  TextAreaElement: {
                    rows: 3,
                    id: noteFieldId,
                  },
                }}
                {...register('text')}
              />
            </FormField>
            <Flex justify="end">
              <Button type="submit" disabled={isMutating || !isValid}>
                <MessageSquarePlus />
                Add Note
              </Button>
            </Flex>
          </Stack>
        </form>

        {/* Fixed-width feedback column (matches the design's 201px frame) so the
            FormField — whose intrinsic width is 100% — can't collapse to the
            "Feedback" label width and wrap the thumb buttons. `shrink-0` keeps it
            from being squeezed by the note form, which fills the rest of the row.
            The FormField label aligns the thumbs with the textarea row. */}
        <div className="w-[14rem] shrink-0">
          <FormField name="annotation-feedback" slotLabel="Feedback">
            <Flex gap="density-sm" align="center">
              <ThumbButton
                direction="up"
                selected={activeFeedback?.value === 'positive'}
                disabled={isMutating}
                onClick={() => void submitFeedback(FeedbackAnnotationInputValue.positive)}
              >
                Positive
              </ThumbButton>
              <ThumbButton
                direction="down"
                selected={activeFeedback?.value === 'negative'}
                disabled={isMutating}
                onClick={() => void submitFeedback(FeedbackAnnotationInputValue.negative)}
              >
                Negative
              </ThumbButton>
            </Flex>
          </FormField>
        </div>
      </Flex>

      {(mutationError || listError) && (
        <Text kind="body/regular/sm" className="text-danger">
          {mutationError ?? getAnnotationErrorMessage(listError, 'Failed to load annotations.')}
        </Text>
      )}
    </Stack>
  );
};
