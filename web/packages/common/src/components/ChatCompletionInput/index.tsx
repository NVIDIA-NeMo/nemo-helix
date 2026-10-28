/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import {
  chatCompletionRoleBadgeColor,
  DEFAULT_CHAT_COMPLETION_ROLE_ITEMS,
  type ChatCompletionRoleSelectItem,
} from '@nemo/common/src/components/ChatCompletionInput/constants';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledVariableTextArea } from '@nemo/common/src/components/form/ControlledVariableTextArea';
import type {
  VariableDef,
  VariableTextAreaHandle,
} from '@nemo/common/src/components/form/VariableTextArea';
import { ArrowDown, ArrowUp, ChevronDown, CopyGeneric, Trash } from '@nemo/icons/icons';
import {
  Badge,
  Button,
  Flex,
  InputShell,
  Stack,
  Text,
  Tooltip,
} from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { type ReactNode, useCallback, useMemo, useRef } from 'react';
import {
  type Control,
  type FieldValues,
  type Path,
  useController,
  useFormState,
  useWatch,
} from 'react-hook-form';

export type { ChatCompletionMessageRowValues } from '@nemo/common/src/components/ChatCompletionInput/schema';
export {
  chatCompletionMessageRowSchema,
  chatCompletionMessageRowRoles,
  defaultChatCompletionMessageRow,
} from '@nemo/common/src/components/ChatCompletionInput/schema';
export {
  CHAT_COMPLETION_ROLE_BADGE_COLOR,
  chatCompletionRoleBadgeColor,
  DEFAULT_CHAT_COMPLETION_ROLE_ITEMS,
  type ChatCompletionRoleBadgeColor,
} from '@nemo/common/src/components/ChatCompletionInput/constants';

export interface ChatCompletionInputProps<TFieldValues extends FieldValues = FieldValues> {
  control: Control<TFieldValues>;
  /**
   * Path to this message row in the form (no trailing segment).
   * Example with `useFieldArray({ name: 'messages' })`: `messages.${index}`.
   */
  name: string;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  /**
   * When set to `1`, move up/down controls are not shown (nothing to reorder).
   * Pass `fields.length` from `useFieldArray` when wiring reorder callbacks.
   */
  fieldArrayLength?: number;
  onDuplicate?: () => void;
  onRemove?: () => void;
  /** When false, the delete action is disabled. Default true. */
  allowRemove?: boolean;
  /** Extra actions or future UI (templates, tools, …) below the message body. */
  footer?:
    | ReactNode
    | ((api: { insertVariable: (name: string) => void; focus: () => void }) => ReactNode);
  className?: string;
  /** Role options for the dropdown; defaults to {@link DEFAULT_CHAT_COMPLETION_ROLE_ITEMS}. */
  roleItems?: ChatCompletionRoleSelectItem[];
  contentPlaceholder?: string;
  variables?: VariableDef[];
  /** Merged with react-hook-form `disabled` form state. */
  disabled?: boolean;
  /** Optional anchor for tests (`data-testid` on the message row root). */
  dataTestId?: string;
}

/**
 * One chat completion message row: role (RHF), expand/collapse (RHF `expanded`), content (RHF),
 * reorder / duplicate / delete (callbacks), autoresizing body, optional footer slot.
 *
 * Expected shape under `name`: `{ role, content, expanded }` — see {@link defaultChatCompletionMessageRow}.
 */
export function ChatCompletionInput<TFieldValues extends FieldValues = FieldValues>({
  control,
  name,
  onMoveUp,
  onMoveDown,
  fieldArrayLength,
  onDuplicate,
  onRemove,
  allowRemove = true,
  footer,
  className,
  roleItems = DEFAULT_CHAT_COMPLETION_ROLE_ITEMS,
  contentPlaceholder = 'Type your message...',
  variables,
  disabled: disabledProp,
  dataTestId,
}: ChatCompletionInputProps<TFieldValues>) {
  const { disabled: formDisabled } = useFormState({ control });
  const isDisabled = Boolean(disabledProp) || Boolean(formDisabled);

  const variableRef = useRef<VariableTextAreaHandle>(null);

  const insertVariable = useCallback((name: string) => {
    variableRef.current?.insertVariable(name);
  }, []);
  const focusEditor = useCallback(() => variableRef.current?.focus(), []);

  const rolePath = `${name}.role` as Path<TFieldValues>;
  const contentPath = `${name}.content` as Path<TFieldValues>;
  const expandedPath = `${name}.expanded` as Path<TFieldValues>;

  const roleSelectItems = useMemo(
    () =>
      roleItems.map((item) => ({
        value: item.value,
        children: (
          <Badge
            kind="solid"
            color={chatCompletionRoleBadgeColor(item.value)}
            className="font-normal"
          >
            {item.children}
          </Badge>
        ),
      })),
    [roleItems]
  );

  const renderRoleValue = useCallback(
    (value: string) => {
      const item = roleItems.find((r) => r.value === value);
      const label = item?.children ?? value;
      return (
        <Badge kind="solid" color={chatCompletionRoleBadgeColor(value)} className="font-normal">
          {label}
        </Badge>
      );
    },
    [roleItems]
  );

  const contentValue = useWatch({ control, name: contentPath });

  const { field: expandedField } = useController({ control, name: expandedPath });

  const expanded = expandedField.value !== false;

  const setExpanded = useCallback(
    (next: boolean) => {
      expandedField.onChange(next);
    },
    [expandedField]
  );

  const toggleExpanded = useCallback(() => {
    setExpanded(!expanded);
  }, [expanded, setExpanded]);

  const showReorder =
    fieldArrayLength === undefined ? Boolean(onMoveUp || onMoveDown) : fieldArrayLength > 1;
  const moveUpAction = showReorder ? onMoveUp : undefined;
  const moveDownAction = showReorder ? onMoveDown : undefined;

  const hasReorderActions = Boolean(moveUpAction || moveDownAction);
  const hasDuplicateOrRemove = Boolean(onDuplicate || onRemove);
  const showActionStrip = hasReorderActions || hasDuplicateOrRemove;

  const rawContent = typeof contentValue === 'string' ? contentValue : '';
  const collapsedPreviewText =
    rawContent.trim() !== '' ? rawContent.replace(/\s+/g, ' ').trim() : '';

  return (
    <InputShell
      kind="flat"
      className={cn(
        'group transition-[border-color] duration-150 h-auto pt-2 pb-3',
        !isDisabled &&
          'has-[textarea:focus]:border-[var(--border-color-brand)] has-[.cm-content:focus]:border-[var(--border-color-brand)]',
        className
      )}
      disableFocusRedirect
      {...(dataTestId ? { 'data-testid': dataTestId } : {})}
    >
      <Stack gap="2" className="flex-1 min-w-0">
        <Flex align="center" gap="1">
          <div className="w-[140px] shrink-0">
            <ControlledSelect
              attributes={{
                SelectTrigger: {
                  className: cn(
                    'border-none flex h-8 min-h-0 max-h-8 items-center p-0 shadow-none bg-transparent',
                    // The inner trigger button carries foundations' own 12px inline-start
                    // padding; zero it so the role badge left-aligns with the message body.
                    '[&_button]:!pl-0',
                    'data-[state=open]:shadow-none'
                  ),
                },
              }}
              useControllerProps={{ control, name: rolePath, disabled: isDisabled }}
              items={roleSelectItems}
              renderValue={renderRoleValue}
              disabled={isDisabled}
            />
          </div>

          {expanded ? (
            <div className="flex-1" />
          ) : (
            <div className="flex min-h-0 min-w-0 flex-1 items-center self-center">
              <Text
                kind="body/regular/md"
                className="text-muted m-0 w-full min-w-0 truncate pl-1 pr-2 leading-normal"
                title={rawContent.trim() !== '' ? rawContent : undefined}
              >
                {collapsedPreviewText}
              </Text>
            </div>
          )}

          <Flex align="center" gap="3" className="shrink-0">
            {showActionStrip ? (
              <Flex
                align="center"
                gap="0.5"
                className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
              >
                {hasReorderActions ? (
                  <Flex align="center" gap="density-xs" className="shrink-0">
                    {moveUpAction ? (
                      <Tooltip slotContent="Move up" side="top">
                        <Button
                          type="button"
                          size="tiny"
                          kind="tertiary"
                          aria-label="Move up"
                          disabled={isDisabled}
                          onClick={moveUpAction}
                        >
                          <ArrowUp className="size-3" aria-hidden />
                        </Button>
                      </Tooltip>
                    ) : null}
                    {moveDownAction ? (
                      <Tooltip slotContent="Move down" side="top">
                        <Button
                          type="button"
                          size="tiny"
                          kind="tertiary"
                          aria-label="Move down"
                          disabled={isDisabled}
                          onClick={moveDownAction}
                        >
                          <ArrowDown className="size-3" aria-hidden />
                        </Button>
                      </Tooltip>
                    ) : null}
                  </Flex>
                ) : null}
                {hasDuplicateOrRemove ? (
                  <Flex align="center" gap="density-xs" className="shrink-0">
                    {onDuplicate ? (
                      <Tooltip slotContent="Duplicate message" side="top">
                        <Button
                          type="button"
                          size="tiny"
                          kind="tertiary"
                          aria-label="Duplicate message"
                          disabled={isDisabled}
                          onClick={onDuplicate}
                        >
                          <CopyGeneric className="size-3" aria-hidden />
                        </Button>
                      </Tooltip>
                    ) : null}
                    {onRemove ? (
                      <Tooltip slotContent="Delete message" side="top">
                        <Button
                          type="button"
                          size="tiny"
                          kind="tertiary"
                          aria-label="Delete message"
                          disabled={isDisabled || !allowRemove}
                          onClick={onRemove}
                        >
                          <Trash className="size-3" aria-hidden />
                        </Button>
                      </Tooltip>
                    ) : null}
                  </Flex>
                ) : null}
              </Flex>
            ) : null}

            <Tooltip slotContent={expanded ? 'Collapse message' : 'Expand message'} side="top">
              <Button
                type="button"
                kind="tertiary"
                size="tiny"
                aria-expanded={expanded}
                aria-label={expanded ? 'Collapse message' : 'Expand message'}
                disabled={isDisabled}
                onClick={toggleExpanded}
                className="size-3 shrink-0 p-0"
              >
                <ChevronDown
                  className={cn('size-3.5 transition-transform', !expanded && '-rotate-90')}
                  aria-hidden
                />
              </Button>
            </Tooltip>
          </Flex>
        </Flex>

        {expanded ? (
          variables ? (
            <ControlledVariableTextArea
              ref={variableRef}
              variables={variables}
              useControllerProps={{ control, name: contentPath, disabled: isDisabled }}
              formFieldProps={{ name: String(contentPath) }}
              placeholder={contentPlaceholder}
              readOnly={isDisabled}
              disabled={isDisabled}
              className="border-none shadow-none rounded-none bg-transparent p-0 [&_.cm-editor]:border-none [&_.cm-line]:px-0"
              attributes={{
                TextAreaElement: {
                  'data-testid': 'chat-completion-message-content',
                  'aria-label': String(contentPath),
                },
              }}
            />
          ) : (
            <ControlledTextArea
              useControllerProps={{ control, name: contentPath, disabled: isDisabled }}
              formFieldProps={{ name: String(contentPath) }}
              placeholder={contentPlaceholder}
              resizeable="auto"
              readOnly={isDisabled}
              disabled={isDisabled}
              className="border-none shadow-none rounded-none bg-transparent p-0 [&_.nv-text-area-element]:border-none [&_.nv-text-area-element]:bg-transparent [&_.nv-text-area-element]:shadow-none"
              attributes={{
                TextAreaElement: {
                  name: String(contentPath),
                  ['data-testid' as never]: 'chat-completion-message-content',
                },
              }}
            />
          )
        ) : null}

        {(() => {
          const resolvedFooter =
            typeof footer === 'function' ? footer({ insertVariable, focus: focusEditor }) : footer;
          return resolvedFooter ? <div className="min-w-0">{resolvedFooter}</div> : null;
        })()}
      </Stack>
    </InputShell>
  );
}
