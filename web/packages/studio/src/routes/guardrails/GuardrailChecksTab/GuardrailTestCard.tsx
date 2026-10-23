// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Add } from '@nemo/icons/icons';
import { Button, Flex, Panel, SegmentedControl, Stack, Text } from '@nvidia/foundations-react-core';
import { useUpdateGuardrailCheck } from '@studio/api/guardrail-checks/hooks';
import type {
  GuardrailCheckEntity,
  GuardrailCheckMessage,
} from '@studio/api/guardrail-checks/types';
import {
  type GuardrailCheckFormValues,
  type GuardrailMessageFormRow,
  GuardrailMessageRow,
} from '@studio/routes/guardrails/GuardrailChecksTab/GuardrailMessageRow';
import { type FC, type FocusEvent, useCallback, useEffect, useRef, useState } from 'react';
import { useFieldArray, useForm, useWatch } from 'react-hook-form';

interface GuardrailTestCardProps {
  check: GuardrailCheckEntity;
  index: number;
  workspace: string;
  /** Registers this card's flusher with the editor; called with `null` on unmount. */
  registerFlush: (name: string, flush: (() => Promise<GuardrailCheckEntity>) | null) => void;
}

type ViewMode = 'chat' | 'json';

const emptyMessageRow = (): GuardrailMessageFormRow => ({
  role: 'user',
  content: '',
});

/** Whether the user has typed since a save was dispatched. */
const isSameRows = (a: GuardrailMessageFormRow[], b: GuardrailMessageFormRow[]): boolean =>
  a.length === b.length &&
  a.every((row, i) => row.role === b[i]?.role && row.content === b[i]?.content);

/** Server messages -> RHF rows. Non-string content is coerced to '' (the editor is text-only). */
const toFormRows = (messages: GuardrailCheckMessage[]): GuardrailMessageFormRow[] => {
  const rows = messages.map((m) => ({
    role: m.role,
    content: typeof m.content === 'string' ? m.content : '',
  }));
  return rows.length > 0 ? rows : [emptyMessageRow()];
};

/** RHF rows -> server messages. */
const toCheckMessages = (rows: GuardrailMessageFormRow[]): GuardrailCheckMessage[] =>
  rows.map(({ role, content }) => ({ role, content }) as GuardrailCheckMessage);

export const GuardrailTestCard: FC<GuardrailTestCardProps> = ({
  check,
  index,
  workspace,
  registerFlush,
}) => {
  const toast = useToast();
  const [viewMode, setViewMode] = useState<ViewMode>('chat');

  const form = useForm<GuardrailCheckFormValues>({
    defaultValues: { messages: toFormRows(check.data.messages) },
  });
  // Read during render so RHF's formState proxy subscribes to dirty transitions.
  const { isDirty } = form.formState;
  const { fields, append, insert, move, remove } = useFieldArray({
    control: form.control,
    name: 'messages',
  });

  // Sync the RHF draft if the server entity is refreshed externally.
  const lastCheckId = useRef(check.id);
  useEffect(() => {
    if (check.id !== lastCheckId.current) {
      lastCheckId.current = check.id;
      form.reset({ messages: toFormRows(check.data.messages) });
    }
  }, [check.id, check.data.messages, form]);

  const updateMutation = useUpdateGuardrailCheck({
    onError: (error) => {
      toast.error(getErrorMessage(error, 'Failed to save check'));
    },
  });

  // "Run Tests" awaits this so a run never executes against an already-edited-past snapshot.
  const pendingSaveRef = useRef<Promise<GuardrailCheckEntity> | null>(null);

  const persist = useCallback(
    (rows: GuardrailMessageFormRow[]): Promise<GuardrailCheckEntity> => {
      const saved = updateMutation
        .mutateAsync({
          workspace,
          name: check.name,
          patch: {
            data: { ...check.data, messages: toCheckMessages(rows) },
            expected_db_version: check.db_version,
            parent: check.parent,
          },
        })
        .then((entity) => {
          // Clear dirty only if nothing was typed since dispatch.
          if (isSameRows(form.getValues('messages'), rows)) {
            form.reset({ messages: rows });
          }
          return entity;
        })
        // onError already toasts; don't abort the run. Stays dirty, so the next flush retries.
        .catch(() => check)
        .finally(() => {
          if (pendingSaveRef.current === saved) pendingSaveRef.current = null;
        });

      pendingSaveRef.current = saved;
      return saved;
    },
    [check, form, updateMutation, workspace]
  );

  /** What a run executes against: the in-flight save, a fresh save if dirty, else `check`. */
  const flush = useCallback((): Promise<GuardrailCheckEntity> => {
    if (pendingSaveRef.current) return pendingSaveRef.current;
    if (!isDirty) return Promise.resolve(check);
    return persist(form.getValues('messages'));
  }, [check, form, isDirty, persist]);

  useEffect(() => {
    registerFlush(check.name, flush);
    return () => registerFlush(check.name, null);
  }, [check.name, flush, registerFlush]);

  const handleMove = (from: number, to: number) => {
    const rows = form.getValues('messages');
    const next = [...rows];
    [next[from], next[to]] = [next[to], next[from]];
    move(from, to);
    persist(next);
  };

  const handleDuplicate = (i: number) => {
    const rows = form.getValues('messages');
    const source = rows[i];
    if (!source) return;
    const copy = { ...source };
    insert(i + 1, copy);
    persist([...rows.slice(0, i + 1), copy, ...rows.slice(i + 1)]);
  };

  const handleRemove = (i: number) => {
    const rows = form.getValues('messages');
    remove(i);
    persist(rows.filter((_, idx) => idx !== i));
  };

  const handleAddMessage = () => {
    const row = emptyMessageRow();
    append(row);
    persist([...form.getValues('messages'), row]);
  };

  // Persist content edits only when focus leaves the message list entirely; moves/duplicates/
  // removes persist explicitly. Skip clean cards — every write bumps db_version.
  const handleContainerBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    if (!isDirty) return;
    persist(form.getValues('messages'));
  };

  const watchedMessages = useWatch({ control: form.control, name: 'messages' });

  return (
    <Panel elevation="high">
      <Stack gap="density-md">
        {/* Card header */}
        <Flex align="center" justify="between">
          <Text kind="label/bold/md">Test {index + 1}</Text>
          <SegmentedControl
            size="small"
            value={viewMode}
            onValueChange={(value) => setViewMode(value as ViewMode)}
            items={[
              { value: 'chat', children: 'Chat' },
              { value: 'json', children: 'JSON' },
            ]}
          />
        </Flex>

        {/* Chat view */}
        {viewMode === 'chat' ? (
          <div onBlur={handleContainerBlur}>
            <Stack gap="density-md">
              {fields.map((field, i) => (
                <GuardrailMessageRow
                  key={field.id}
                  control={form.control}
                  name={`messages.${i}`}
                  dataTestId={`guardrail-check-message-${i}`}
                  onMoveUp={i > 0 ? () => handleMove(i, i - 1) : undefined}
                  onMoveDown={i < fields.length - 1 ? () => handleMove(i, i + 1) : undefined}
                  onDuplicate={() => handleDuplicate(i)}
                  onRemove={() => handleRemove(i)}
                  allowRemove={fields.length > 1}
                />
              ))}
              <Button
                kind="tertiary"
                color="neutral"
                size="small"
                onClick={handleAddMessage}
                className="w-fit"
              >
                <Add size="14" />
                Add Message
              </Button>
            </Stack>
          </div>
        ) : (
          /* JSON view (read-only) */
          <pre className="overflow-auto rounded bg-surface p-density-md text-xs font-mono text-text-primary">
            {JSON.stringify(toCheckMessages(watchedMessages ?? []), null, 2)}
          </pre>
        )}
      </Stack>
    </Panel>
  );
};
