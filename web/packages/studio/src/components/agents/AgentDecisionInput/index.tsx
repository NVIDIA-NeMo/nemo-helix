// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ArrowDown, ArrowUp, Paperplane } from '@nemo/icons/icons';
import { Button, Flex, Text, TextInput } from '@nvidia/foundations-react-core';
import { type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from 'react';

export type AgentDecisionInputStatus = 'pending' | 'submitting';

export interface AgentDecisionChoice {
  id: string;
  label: string;
  description?: string;
  input?: {
    ariaLabel?: string;
    placeholder?: string;
  };
}

export interface AgentDecisionRequest {
  id: string;
  title: string;
  description?: string;
  details?: unknown;
}

export interface AgentDecisionSubmission {
  text?: string;
}

interface AgentDecisionInputProps {
  request: AgentDecisionRequest;
  choices: readonly AgentDecisionChoice[];
  status?: AgentDecisionInputStatus;
  defaultChoiceId?: string;
  skipLabel?: string;
  onSubmit: (
    choice: AgentDecisionChoice,
    submission?: AgentDecisionSubmission
  ) => Promise<void> | void;
  onSkip?: () => Promise<void> | void;
  renderDetails?: (details: unknown) => ReactNode;
}

const formatDetails = (details: unknown): string => {
  if (details === undefined || details === null) return '';
  if (typeof details === 'string') return details;
  if (typeof details === 'object' && 'command' in details && typeof details.command === 'string') {
    return details.command;
  }

  try {
    return JSON.stringify(details, null, 2);
  } catch {
    return String(details);
  }
};

const getInitialChoiceId = (
  choices: readonly AgentDecisionChoice[],
  defaultChoiceId?: string
): string => {
  if (defaultChoiceId && choices.some((choice) => choice.id === defaultChoiceId)) {
    return defaultChoiceId;
  }
  return choices[0]?.id ?? '';
};

const getChoiceRowClassName = (selected: boolean): string =>
  [
    'min-h-10 w-full justify-start rounded-lg px-density-sm py-density-xs text-left',
    'disabled:cursor-not-allowed disabled:opacity-50',
    'focus-visible:[outline:1px_solid_var(--border-color-interaction-base)] focus-visible:[outline-offset:-1px] focus-visible:[box-shadow:none]',
    selected
      ? 'bg-surface-overlay text-fg-primary'
      : 'bg-transparent text-fg-secondary hover:bg-surface-raised',
  ].join(' ');

const subduedInputFocusClassName = [
  '[--border-color:var(--border-color-interaction-base)] [--outline-offset:-1px]',
  'focus-within:[outline:1px_solid_var(--border-color-interaction-base)] focus-within:[box-shadow:none]',
].join(' ');

const subduedButtonFocusClassName =
  'focus-visible:[outline:1px_solid_var(--border-color-interaction-base)] focus-visible:[outline-offset:-1px] focus-visible:[box-shadow:none]';

export const AgentDecisionInput = ({
  request,
  choices,
  status = 'pending',
  defaultChoiceId,
  skipLabel = 'Skip',
  onSubmit,
  onSkip,
  renderDetails,
}: AgentDecisionInputProps) => {
  const [selectedChoiceId, setSelectedChoiceId] = useState(() =>
    getInitialChoiceId(choices, defaultChoiceId)
  );
  const [choiceInputValue, setChoiceInputValue] = useState('');
  const rootRef = useRef<HTMLDivElement>(null);
  const choiceInputRef = useRef<HTMLInputElement>(null);
  const detailsText = formatDetails(request.details);
  const isSubmitting = status === 'submitting';
  const selectedChoice = choices.find((choice) => choice.id === selectedChoiceId) ?? choices[0];
  const selectedChoiceIndex = choices.findIndex((choice) => choice.id === selectedChoice?.id);
  const selectedChoiceInput = selectedChoice?.input;
  const trimmedChoiceInputValue = choiceInputValue.trim();

  useEffect(() => {
    setSelectedChoiceId(getInitialChoiceId(choices, defaultChoiceId));
    setChoiceInputValue('');
  }, [choices, defaultChoiceId, request.id]);

  useEffect(() => {
    rootRef.current?.focus();
  }, [request.id]);

  useEffect(() => {
    if (!selectedChoiceInput) return;
    choiceInputRef.current?.focus();
  }, [selectedChoiceInput]);

  const submitChoice = async (choice: AgentDecisionChoice | undefined) => {
    if (!choice || isSubmitting) return;

    if (choice.input) {
      if (!trimmedChoiceInputValue) {
        setSelectedChoiceId(choice.id);
        choiceInputRef.current?.focus();
        return;
      }
      await onSubmit(choice, { text: trimmedChoiceInputValue });
      return;
    }

    await onSubmit(choice);
  };

  const submitSelectedChoice = async () => {
    await submitChoice(selectedChoice);
  };

  const selectChoiceByOffset = (offset: number) => {
    if (!choices.length) return;
    const currentIndex = selectedChoiceIndex === -1 ? 0 : selectedChoiceIndex;
    const nextIndex = (currentIndex + offset + choices.length) % choices.length;
    setSelectedChoiceId(choices[nextIndex].id);
  };

  const selectChoiceByOffsetAndFocusRoot = (offset: number) => {
    selectChoiceByOffset(offset);
    requestAnimationFrame(() => rootRef.current?.focus());
  };

  const selectOrSubmitChoice = (choice: AgentDecisionChoice) => {
    setSelectedChoiceId(choice.id);

    if (choice.input) {
      requestAnimationFrame(() => choiceInputRef.current?.focus());
      return;
    }

    void submitChoice(choice);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLInputElement) return;
    if (isSubmitting) return;

    const numericChoiceIndex = Number(event.key);
    if (Number.isInteger(numericChoiceIndex) && numericChoiceIndex >= 1) {
      const choice = choices[numericChoiceIndex - 1];
      if (!choice) return;
      event.preventDefault();
      selectOrSubmitChoice(choice);
      return;
    }

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      selectChoiceByOffset(1);
      return;
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault();
      selectChoiceByOffset(-1);
      return;
    }

    if (event.key === 'Enter') {
      event.preventDefault();
      void submitSelectedChoice();
    }
  };

  const handleChoiceInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (isSubmitting) return;

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      selectChoiceByOffsetAndFocusRoot(1);
      return;
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault();
      selectChoiceByOffsetAndFocusRoot(-1);
      return;
    }

    if (event.key === 'Enter') {
      event.preventDefault();
      void submitSelectedChoice();
    }
  };

  const renderChoiceContent = (choice: AgentDecisionChoice, selected: boolean) => {
    const showInput = selected && choice.input;

    if (showInput) {
      return (
        <TextInput
          aria-label={choice.input?.ariaLabel ?? choice.label}
          value={choiceInputValue}
          disabled={isSubmitting}
          placeholder={choice.input?.placeholder}
          size="small"
          className={`min-w-0 flex-1 ${subduedInputFocusClassName}`}
          onValueChange={(nextValue) => setChoiceInputValue(nextValue ?? '')}
          attributes={{
            Input: {
              ref: choiceInputRef,
              onKeyDown: handleChoiceInputKeyDown,
            },
          }}
        />
      );
    }

    return (
      <Flex direction="col" className="min-w-0 flex-1">
        <Text kind="body/semibold/sm" className="block truncate">
          {choice.label}
        </Text>
        {choice.description ? (
          <Text kind="body/regular/xs" className="block truncate text-fg-secondary">
            {choice.description}
          </Text>
        ) : null}
      </Flex>
    );
  };

  const handleInputChoiceRowKeyDown = (
    event: KeyboardEvent<HTMLDivElement>,
    choice: AgentDecisionChoice
  ) => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.key !== 'Enter' && event.key !== ' ') return;

    event.preventDefault();
    selectOrSubmitChoice(choice);
  };

  return (
    <Flex
      ref={rootRef}
      direction="col"
      tabIndex={0}
      role="group"
      aria-label={request.title}
      className="w-full rounded-xl border border-base bg-surface-base p-density-md outline-none"
      data-testid="agent-decision-input"
      onKeyDown={handleKeyDown}
    >
      <Flex direction="col" gap="density-md" className="min-h-16 w-full">
        <Flex direction="col" gap="density-xs" className="min-w-0">
          <Text kind="body/semibold/md" className="block">
            {request.title}
          </Text>
          {request.description ? (
            <Text kind="body/regular/sm" className="block text-fg-secondary">
              {request.description}
            </Text>
          ) : null}
        </Flex>
        {renderDetails ? renderDetails(request.details) : null}
        {!renderDetails && detailsText ? (
          <pre className="max-h-24 overflow-auto px-density-xs text-sm text-fg-secondary whitespace-pre-wrap">
            {detailsText}
          </pre>
        ) : null}
        <Flex direction="col" gap="density-sm">
          <Flex direction="col" gap="density-xs" role="listbox" aria-label="Decision options">
            {choices.map((choice, index) => {
              const selected = choice.id === selectedChoice?.id;
              const choiceContent = (
                <Flex align="center" gap="density-xs" className="w-full min-w-0">
                  <Text kind="label/regular/sm" className="w-8 shrink-0 text-fg-secondary">
                    {index + 1}.
                  </Text>
                  {renderChoiceContent(choice, selected)}
                  <Flex
                    align="center"
                    gap="density-xs"
                    className="shrink-0 text-fg-disabled"
                    aria-hidden
                  >
                    {selected ? (
                      <>
                        <ArrowUp size="16" />
                        <ArrowDown size="16" />
                      </>
                    ) : null}
                  </Flex>
                </Flex>
              );

              if (choice.input) {
                return (
                  <Flex
                    key={choice.id}
                    role="option"
                    aria-label={`${index + 1}. ${choice.label}`}
                    aria-selected={selected}
                    aria-disabled={isSubmitting}
                    tabIndex={-1}
                    className={getChoiceRowClassName(selected)}
                    onClick={() => selectOrSubmitChoice(choice)}
                    onKeyDown={(event: KeyboardEvent<HTMLDivElement>) =>
                      handleInputChoiceRowKeyDown(event, choice)
                    }
                  >
                    {choiceContent}
                  </Flex>
                );
              }

              return (
                <Button
                  key={choice.id}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  disabled={isSubmitting}
                  kind="tertiary"
                  className={getChoiceRowClassName(selected)}
                  onClick={() => selectOrSubmitChoice(choice)}
                >
                  {choiceContent}
                </Button>
              );
            })}
          </Flex>
          <Flex align="center" justify="end" gap="density-sm">
            {onSkip ? (
              <Button
                type="button"
                kind="tertiary"
                color="neutral"
                disabled={isSubmitting}
                className={subduedButtonFocusClassName}
                onClick={() => void onSkip()}
              >
                <Text kind="label/regular/sm">{skipLabel}</Text>
              </Button>
            ) : null}
            <Button
              aria-label="Send"
              type="button"
              color="brand"
              size="small"
              disabled={isSubmitting || (!!selectedChoiceInput && !trimmedChoiceInputValue)}
              className={subduedButtonFocusClassName}
              onClick={() => void submitSelectedChoice()}
            >
              <Paperplane size="16" />
            </Button>
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
};
