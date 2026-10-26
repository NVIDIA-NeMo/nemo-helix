// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import type { NvidiaGuiIconProps } from '@nemo/icons';
import { ArrowDown, ArrowUp, CopyGeneric, Trash } from '@nemo/icons/icons';
import { Button, Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import type { GuardrailCheckMessage } from '@studio/api/guardrail-checks/types';
import cn from 'classnames';
import { type FC, useRef } from 'react';
import { type Control, type Path, useController } from 'react-hook-form';

/** RHF row shape for a single guardrail-check message. No collapse state — the body is always shown. */
export interface GuardrailMessageFormRow {
  role: GuardrailCheckMessage['role'];
  content: string;
}

export interface GuardrailCheckFormValues {
  messages: GuardrailMessageFormRow[];
}

/** Roles offered in the row's role dropdown (subset of the chat-completion roles). */
const ROLE_ITEMS: { value: string; children: string }[] = [
  { value: 'user', children: 'user' },
  { value: 'assistant', children: 'assistant' },
  { value: 'system', children: 'system' },
];

/** Plain-text role label (mock shows e.g. "user ⌄", not a colored badge). */
const renderRoleValue = (value: string) => <Text kind="label/bold/sm">{value}</Text>;

/** A single icon action in the row header. */
const RowIconButton: FC<{
  label: string;
  icon: FC<Omit<NvidiaGuiIconProps, 'iconName'>>;
  onClick: () => void;
  disabled?: boolean;
}> = ({ label, icon: Icon, onClick, disabled }) => (
  <Tooltip slotContent={label} side="top">
    <Button
      type="button"
      size="tiny"
      kind="tertiary"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
    >
      <Icon className="size-3.5" aria-hidden />
    </Button>
  </Tooltip>
);

export interface GuardrailMessageRowProps {
  control: Control<GuardrailCheckFormValues>;
  /** Path to this message row in the form, e.g. `messages.0`. */
  name: string;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onDuplicate: () => void;
  onRemove: () => void;
  /** When false, the delete action is disabled (never remove the last message). */
  allowRemove: boolean;
  dataTestId?: string;
}

/**
 * One guardrail-check message: a role dropdown + an always-visible, manually resizable
 * text body, with reorder / duplicate / delete actions. Purpose-built for the guardrails
 * Tests tab (replaces the shared ChatCompletionInput).
 *
 * The row is a single bordered card with a native textarea — no nested input shell — so the
 * body has exactly one border (which highlights on focus) rather than a doubled one.
 */
export const GuardrailMessageRow: FC<GuardrailMessageRowProps> = ({
  control,
  name,
  onMoveUp,
  onMoveDown,
  onDuplicate,
  onRemove,
  allowRemove,
  dataTestId,
}) => {
  const rolePath = `${name}.role` as Path<GuardrailCheckFormValues>;
  const contentPath = `${name}.content` as Path<GuardrailCheckFormValues>;
  const { field: content } = useController({ control, name: contentPath });
  const bodyRef = useRef<HTMLTextAreaElement>(null);

  // After a role is chosen the Select closes and restores focus to its trigger at an unknown tick.
  // Re-focus the message body across a short window so it wins whenever that restoration fires,
  // keeping the user in the typing flow.
  const focusBody = () => {
    let ticks = 0;
    const id = setInterval(() => {
      bodyRef.current?.focus();
      if (++ticks >= 8) clearInterval(id);
    }, 25);
  };

  return (
    <Stack
      gap="density-sm"
      className={cn(
        'rounded-md border border-interaction-base bg-surface-base px-density-md py-density-sm',
        'transition-[border-color] focus-within:border-[var(--border-color-brand)]'
      )}
      {...(dataTestId ? { 'data-testid': dataTestId } : {})}
    >
      <Flex align="center" justify="between" gap="density-sm">
        {/* Role dropdown: plain text + chevron, sized to the selected role (not the widest). */}
        <div className="w-max shrink-0">
          <ControlledSelect
            useControllerProps={{ control, name: rolePath }}
            items={ROLE_ITEMS}
            renderValue={renderRoleValue}
            onChange={focusBody}
            hideError
            attributes={{
              SelectTrigger: {
                className: cn(
                  'w-max border-none bg-transparent shadow-none items-center gap-density-xs',
                  'h-6 min-h-0 max-h-6 p-0 [&_button]:!px-0 [&_*]:!min-w-max',
                  'data-[state=open]:shadow-none'
                ),
              },
              // Menu width is decoupled from the (content-hugging) trigger so item labels aren't clipped.
              SelectContent: { className: 'min-w-[8rem]' },
            }}
          />
        </div>

        <Flex align="center" gap="density-xs" className="shrink-0 text-text-secondary">
          {onMoveUp ? <RowIconButton label="Move up" icon={ArrowUp} onClick={onMoveUp} /> : null}
          {onMoveDown ? (
            <RowIconButton label="Move down" icon={ArrowDown} onClick={onMoveDown} />
          ) : null}
          <RowIconButton label="Duplicate message" icon={CopyGeneric} onClick={onDuplicate} />
          <RowIconButton
            label="Delete message"
            icon={Trash}
            onClick={onRemove}
            disabled={!allowRemove}
          />
        </Flex>
      </Flex>

      <textarea
        name={content.name}
        ref={(el) => {
          content.ref(el);
          bodyRef.current = el;
        }}
        value={typeof content.value === 'string' ? content.value : ''}
        onChange={content.onChange}
        onBlur={content.onBlur}
        placeholder="Type your message..."
        aria-label="Message content"
        data-testid="guardrail-check-message-content"
        className={cn(
          'min-h-[3.5rem] w-full resize-y border-none bg-transparent p-0 outline-none',
          'text-sm leading-normal text-text-primary placeholder:text-muted focus:outline-none'
        )}
      />
    </Stack>
  );
};
