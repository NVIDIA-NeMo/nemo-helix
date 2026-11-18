// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThreadMessageLike } from '@assistant-ui/react';
import {
  CANCELLED_STATUS,
  COMPLETE_STATUS,
} from '@nemo/common/src/components/AssistantChat/constants';
import type {
  AgentDecisionChoice,
  AgentDecisionInputStatus,
  AgentDecisionRequest,
  AgentDecisionSubmission,
} from '@studio/components/agents/AgentDecisionInput';
import {
  createAssistantSession,
  getAssistantHistorySessionsQueryKey,
  resolveAssistantInput,
  resolveAssistantPermission,
  streamAssistantMessage,
} from '@studio/routes/agents/AssistantChatRoute/api';
import {
  createEmptyAssistantChatArtifacts,
  updateAssistantChatArtifactsFromEvent,
  updateAssistantChatArtifactsFromInputSelection,
  updateAssistantChatArtifactsFromSelections,
} from '@studio/routes/agents/AssistantChatRoute/artifacts';
import {
  ASSISTANT_RUN_INTERRUPTED_ERROR,
  describeAssistantRunError,
} from '@studio/routes/agents/AssistantChatRoute/describeAssistantRunError';
import { getAssistantPartsFromAssistantEvent } from '@studio/routes/agents/AssistantChatRoute/stream';
import {
  getStudioUiNavigationSuggestion,
  type StudioUiNavigationSuggestion,
} from '@studio/routes/agents/AssistantChatRoute/studioUiNavigationSuggestions';
import type {
  AssistantChatArtifacts,
  AssistantInputDecision,
  AssistantInputRequest,
  AssistantPermissionRequest,
} from '@studio/routes/agents/AssistantChatRoute/types';
import {
  useCustomAssistantChatRuntime,
  type CustomAssistantRunContext,
  type CustomAssistantRunResult,
} from '@studio/routes/agents/AssistantChatRoute/useCustomAssistantChatRuntime';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';

const ASK_USER_QUESTION_TOOL_NAME = 'AskUserQuestion';
const CUSTOM_INSTRUCTION_LABEL = 'No, and tell the Agent what to do';
const CUSTOM_INSTRUCTION_PLACEHOLDER = 'Tell the Agent what to do';

const createCustomInstructionChoice = (id: string): AgentDecisionChoice => ({
  id,
  label: CUSTOM_INSTRUCTION_LABEL,
  input: {
    ariaLabel: CUSTOM_INSTRUCTION_PLACEHOLDER,
    placeholder: CUSTOM_INSTRUCTION_PLACEHOLDER,
  },
});

const PERMISSION_CHOICES: readonly AgentDecisionChoice[] = [
  { id: 'yes', label: 'Yes' },
  { id: 'no', label: 'No' },
  createCustomInstructionChoice('alternative'),
];

interface AskUserQuestionOption {
  label: string;
  description?: string;
}

interface AskUserQuestion {
  question: string;
  header?: string;
  options: readonly AskUserQuestionOption[];
}

interface PermissionDecisionState {
  kind: 'permission';
}

interface AskUserQuestionDecisionState {
  kind: 'ask-user-question';
  questions: readonly AskUserQuestion[];
  questionIndex: number;
  answers: Record<string, string>;
}

type ActiveDecisionState = PermissionDecisionState | AskUserQuestionDecisionState;

type QueuedRequest =
  | { kind: 'permission'; request: AssistantPermissionRequest }
  | { kind: 'input'; request: AssistantInputRequest };

export type StudioNavigationDecision = 'continue' | 'navigate' | 'cancel';

export interface StudioNavigationRequest {
  id: string;
  prompt: string;
  suggestion: StudioUiNavigationSuggestion;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const isDefined = <T>(value: T | undefined): value is T => value !== undefined;

const trimString = (value: unknown): string | undefined => {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
};

const parseAskUserQuestionOption = (value: unknown): AskUserQuestionOption | undefined => {
  if (!isRecord(value)) return undefined;
  const label = trimString(value.label);
  if (!label) return undefined;
  return { label, description: trimString(value.description) };
};

const parseAskUserQuestion = (value: unknown): AskUserQuestion | undefined => {
  if (!isRecord(value)) return undefined;
  const question = trimString(value.question);
  if (!question) return undefined;
  const options = Array.isArray(value.options)
    ? value.options.map(parseAskUserQuestionOption).filter(isDefined)
    : [];
  return { question, header: trimString(value.header), options };
};

const parseAskUserQuestions = (input: Record<string, unknown>): readonly AskUserQuestion[] => {
  if (!Array.isArray(input.questions)) return [];
  return input.questions.map(parseAskUserQuestion).filter(isDefined);
};

const createAskUserQuestionChoices = (
  question: AskUserQuestion
): readonly AgentDecisionChoice[] => {
  if (!question.options.length) {
    return [createCustomInstructionChoice('answer-text')];
  }
  return [
    ...question.options.map((option, index) => ({
      id: `answer-${index}`,
      label: option.label,
      description: option.description,
    })),
    createCustomInstructionChoice('answer-custom'),
  ];
};

const createAskUserQuestionRequest = (
  requestId: string,
  state: AskUserQuestionDecisionState
): AgentDecisionRequest => {
  const question = state.questions[state.questionIndex];
  const title = question.header ?? 'Agent question';
  return {
    id: `${requestId}:${state.questionIndex}`,
    title:
      state.questions.length > 1
        ? `${title} (${state.questionIndex + 1} of ${state.questions.length})`
        : title,
    description: question.question,
  };
};

const quoteAnswerValue = (value: string): string =>
  `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;

const formatAskUserQuestionReason = (
  questions: readonly AskUserQuestion[],
  answers: Record<string, string>
): string => {
  const answerText = questions
    .map((question) => {
      const answer = answers[question.question];
      if (!answer) return undefined;
      return `${quoteAnswerValue(question.question)}=${quoteAnswerValue(answer)}`;
    })
    .filter(isDefined)
    .join(', ');
  if (questions.length === 1) {
    return `Your question has been answered: ${answerText}. You can now continue with this answer in mind.`;
  }
  return `Your questions have been answered: ${answerText}. You can now continue with these answers in mind.`;
};

const formatAskUserQuestionDisplayText = (
  questions: readonly AskUserQuestion[],
  answers: Record<string, string>
): string =>
  questions
    .map((question) => {
      const answer = answers[question.question];
      if (!answer) return undefined;
      return `${question.question}\n${answer}`;
    })
    .filter(isDefined)
    .join('\n\n');

const createWorkspaceArtifacts = (
  artifacts: AssistantChatArtifacts | undefined,
  workspace: string | undefined
): AssistantChatArtifacts => {
  const nextArtifacts = artifacts ?? createEmptyAssistantChatArtifacts();
  return nextArtifacts.workspace || !workspace ? nextArtifacts : { ...nextArtifacts, workspace };
};

interface UseAssistantChatRuntimeOptions {
  initialArtifacts?: AssistantChatArtifacts;
  initialMessages?: readonly ThreadMessageLike[];
  initialSessionId?: string;
  onError?: (error: Error) => void;
  onSessionIdChange?: (sessionId: string | null) => void;
  studioPathname?: string;
  workspace?: string;
}

interface LoadAssistantSessionOptions {
  artifacts?: AssistantChatArtifacts;
  messages: readonly ThreadMessageLike[];
  sessionId: string;
}

// ---------------------------------------------------------------------------
// Blocking-request reducer
// Owns all permission / input request state including the FIFO queue.
// ---------------------------------------------------------------------------

interface BlockingState {
  activePermission: AssistantPermissionRequest | null;
  activeDecision: ActiveDecisionState | null;
  decisionStatus: AgentDecisionInputStatus;
  activeInput: AssistantInputRequest | null;
  inputStatus: AgentDecisionInputStatus;
  queue: readonly QueuedRequest[];
}

type BlockingAction =
  | { type: 'enqueue_permission'; request: AssistantPermissionRequest }
  | { type: 'enqueue_input'; request: AssistantInputRequest }
  | { type: 'advance_question'; state: AskUserQuestionDecisionState }
  | { type: 'set_decision_status'; status: AgentDecisionInputStatus }
  | { type: 'set_input_status'; status: AgentDecisionInputStatus }
  | { type: 'clear_permission'; requestId?: string; dequeueNext?: boolean }
  | { type: 'clear_input'; requestId?: string; dequeueNext?: boolean }
  | { type: 'expire_permission'; requestId: string }
  | { type: 'expire_input'; requestId: string }
  | { type: 'reset' };

const INITIAL_BLOCKING_STATE: BlockingState = {
  activePermission: null,
  activeDecision: null,
  decisionStatus: 'pending',
  activeInput: null,
  inputStatus: 'pending',
  queue: [],
};

const withPermissionActivated = (
  state: BlockingState,
  request: AssistantPermissionRequest
): BlockingState => {
  if (request.toolName === ASK_USER_QUESTION_TOOL_NAME) {
    const questions = parseAskUserQuestions(request.input);
    if (questions.length) {
      return {
        ...state,
        activePermission: request,
        activeDecision: { kind: 'ask-user-question', questions, questionIndex: 0, answers: {} },
        decisionStatus: 'pending',
      };
    }
  }
  return {
    ...state,
    activePermission: request,
    activeDecision: { kind: 'permission' },
    decisionStatus: 'pending',
  };
};

const withNextDequeued = (state: BlockingState): BlockingState => {
  const [next, ...rest] = state.queue;
  if (!next) return state;
  const base = { ...state, queue: rest };
  if (next.kind === 'permission') return withPermissionActivated(base, next.request);
  return { ...base, activeInput: next.request, inputStatus: 'pending' };
};

function blockingReducer(state: BlockingState, action: BlockingAction): BlockingState {
  switch (action.type) {
    case 'enqueue_permission':
      if (state.activePermission || state.activeInput) {
        return {
          ...state,
          queue: [...state.queue, { kind: 'permission', request: action.request }],
        };
      }
      return withPermissionActivated(state, action.request);

    case 'enqueue_input':
      if (state.activePermission || state.activeInput) {
        return { ...state, queue: [...state.queue, { kind: 'input', request: action.request }] };
      }
      return { ...state, activeInput: action.request, inputStatus: 'pending' };

    case 'advance_question':
      return { ...state, activeDecision: action.state, decisionStatus: 'pending' };

    case 'set_decision_status':
      return { ...state, decisionStatus: action.status };

    case 'set_input_status':
      return { ...state, inputStatus: action.status };

    case 'clear_permission': {
      if (action.requestId && state.activePermission?.requestId !== action.requestId) return state;
      const cleared: BlockingState = {
        ...state,
        activePermission: null,
        activeDecision: null,
        decisionStatus: 'pending',
      };
      return action.dequeueNext ? withNextDequeued(cleared) : cleared;
    }

    case 'clear_input': {
      if (action.requestId && state.activeInput?.requestId !== action.requestId) return state;
      const cleared: BlockingState = { ...state, activeInput: null, inputStatus: 'pending' };
      return action.dequeueNext ? withNextDequeued(cleared) : cleared;
    }

    case 'expire_permission': {
      const withoutQueued = {
        ...state,
        queue: state.queue.filter(
          (q) => !(q.kind === 'permission' && q.request.requestId === action.requestId)
        ),
      };
      if (state.activePermission?.requestId !== action.requestId) return withoutQueued;
      const cleared: BlockingState = {
        ...withoutQueued,
        activePermission: null,
        activeDecision: null,
        decisionStatus: 'pending',
      };
      return withNextDequeued(cleared);
    }

    case 'expire_input': {
      const withoutQueued = {
        ...state,
        queue: state.queue.filter(
          (q) => !(q.kind === 'input' && q.request.requestId === action.requestId)
        ),
      };
      if (state.activeInput?.requestId !== action.requestId) return withoutQueued;
      const cleared: BlockingState = {
        ...withoutQueued,
        activeInput: null,
        inputStatus: 'pending',
      };
      return withNextDequeued(cleared);
    }

    case 'reset':
      return INITIAL_BLOCKING_STATE;
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export const useAssistantChatRuntime = (options?: UseAssistantChatRuntimeOptions) => {
  const queryClient = useQueryClient();
  const workspace = options?.workspace;
  const [sessionId, setSessionId] = useState<string | null>(options?.initialSessionId ?? null);
  const [artifacts, setArtifacts] = useState<AssistantChatArtifacts>(
    createWorkspaceArtifacts(options?.initialArtifacts, workspace)
  );
  const [studioNavigationRequest, setStudioNavigationRequest] =
    useState<StudioNavigationRequest | null>(null);
  const [studioNavigationStatus, setStudioNavigationStatus] =
    useState<AgentDecisionInputStatus>('pending');
  const [blocking, dispatchBlocking] = useReducer(blockingReducer, INITIAL_BLOCKING_STATE);
  const { activeDecision, activeInput, activePermission, decisionStatus, inputStatus } = blocking;
  const sessionIdRef = useRef(sessionId);
  const activeInputRef = useRef(activeInput);
  sessionIdRef.current = sessionId;
  activeInputRef.current = activeInput;

  const [navigationResolver, setNavigationResolver] = useState<
    ((decision: StudioNavigationDecision) => void) | null
  >(null);
  const onError = options?.onError;
  const onSessionIdChange = options?.onSessionIdChange;
  const studioPathname = options?.studioPathname;
  const studioPathnameRef = useRef(studioPathname);
  studioPathnameRef.current = studioPathname;

  // Derive from JSON so callers that create inline objects don't trigger
  // the effect on every render — identity is stable as long as content is unchanged.
  const initialArtifactsJson = options?.initialArtifacts
    ? JSON.stringify(options.initialArtifacts)
    : null;
  const initialArtifacts = useMemo<AssistantChatArtifacts | undefined>(
    () =>
      initialArtifactsJson
        ? (JSON.parse(initialArtifactsJson) as AssistantChatArtifacts)
        : undefined,
    [initialArtifactsJson]
  );

  useEffect(() => {
    setArtifacts(createWorkspaceArtifacts(initialArtifacts, workspace));
  }, [initialArtifacts, workspace]);

  const decisionRequest = useMemo((): AgentDecisionRequest | null => {
    if (!activePermission) return null;
    if (activeDecision?.kind === 'ask-user-question') {
      return createAskUserQuestionRequest(activePermission.requestId, activeDecision);
    }
    const actionDescription =
      typeof activePermission.input.description === 'string'
        ? activePermission.input.description
        : undefined;
    return {
      id: activePermission.requestId,
      title: 'Approval required',
      description: `Agent wants to use ${activePermission.toolName}${
        actionDescription ? `: ${actionDescription}` : ''
      }.`,
      details: activePermission.input,
    };
  }, [activePermission, activeDecision]);

  const decisionChoices = useMemo((): readonly AgentDecisionChoice[] => {
    if (!activePermission) return [];
    if (activeDecision?.kind === 'ask-user-question') {
      return createAskUserQuestionChoices(activeDecision.questions[activeDecision.questionIndex]);
    }
    return PERMISSION_CHOICES;
  }, [activePermission, activeDecision]);

  const ensureSessionId = useCallback(async (): Promise<string> => {
    if (sessionId) return sessionId;
    // Only one run is active at a time (UI prevents concurrent submissions),
    // so no race between concurrent callers here.
    const nextSessionId = await createAssistantSession(workspace);
    sessionIdRef.current = nextSessionId;
    setSessionId(nextSessionId);
    onSessionIdChange?.(nextSessionId);
    return nextSessionId;
  }, [sessionId, onSessionIdChange, workspace]);

  // Accepts an optional decision to cancel a pending navigation promise before
  // clearing, avoiding a spurious 'submitting' status on programmatic cancels.
  const clearStudioNavigationRequest = useCallback((cancelDecision?: StudioNavigationDecision) => {
    setNavigationResolver((current: ((decision: StudioNavigationDecision) => void) | null) => {
      if (cancelDecision) current?.(cancelDecision);
      return null;
    });
    setStudioNavigationRequest(null);
    setStudioNavigationStatus('pending');
  }, []);

  const resolveStudioNavigationRequest = useCallback(
    (decision: StudioNavigationDecision) => {
      if (!navigationResolver) return;
      setStudioNavigationStatus('submitting');
      navigationResolver(decision);
    },
    [navigationResolver]
  );

  // Stable: dispatch is guaranteed stable by React, clearStudioNavigationRequest has [] deps.
  const handlePermissionRequest = useCallback((request: AssistantPermissionRequest) => {
    dispatchBlocking({ type: 'enqueue_permission', request });
  }, []);

  const handleInputRequest = useCallback(
    (request: AssistantInputRequest) => {
      clearStudioNavigationRequest();
      dispatchBlocking({ type: 'enqueue_input', request });
    },
    [clearStudioNavigationRequest]
  );

  const requestStudioNavigationDecision = useCallback(
    async ({
      prompt,
      signal,
      prepareForUserInput,
      isCurrentRun,
    }: {
      prompt: string;
      signal: AbortSignal;
      prepareForUserInput: () => void;
      isCurrentRun: () => boolean;
    }): Promise<'continue' | 'cancel'> => {
      if (!workspace) return 'continue';

      const suggestion = getStudioUiNavigationSuggestion(prompt, workspace);
      if (!suggestion) return 'continue';

      activeInputRef.current = null;
      dispatchBlocking({ type: 'reset' });
      prepareForUserInput();

      let resolveDecision: (decision: StudioNavigationDecision) => void = () => undefined;
      const decisionPromise = new Promise<StudioNavigationDecision>((resolve) => {
        resolveDecision = resolve;
      });
      setNavigationResolver(() => resolveDecision);
      setStudioNavigationStatus('pending');
      setStudioNavigationRequest({ id: `${suggestion.id}:${Date.now()}`, prompt, suggestion });

      const handleAbort = () => resolveDecision('cancel');
      signal.addEventListener('abort', handleAbort, { once: true });

      try {
        const decision = await decisionPromise;
        if (!isCurrentRun()) return 'cancel';
        return decision === 'continue' ? 'continue' : 'cancel';
      } finally {
        signal.removeEventListener('abort', handleAbort);
        clearStudioNavigationRequest();
      }
    },
    [clearStudioNavigationRequest, workspace]
  );

  // Extracted into useCallback so onRun is a stable reference. An inline async
  // function recreates on every render, which propagates through runCompletion →
  // handleNewMessage → useExternalStoreRuntime, causing the runtime to be
  // recreated on every streaming chunk and every message to re-render/flicker.
  const handleRun = useCallback(
    async ({
      prompt,
      signal,
      appendAssistantParts,
      prepareForUserInput,
      isCurrentRun,
    }: CustomAssistantRunContext): Promise<CustomAssistantRunResult | void> => {
      activeInputRef.current = null;
      dispatchBlocking({ type: 'reset' });
      clearStudioNavigationRequest();
      const activeSessionId = await ensureSessionId();
      let doneReceived = false;

      try {
        await streamAssistantMessage({
          sessionId: activeSessionId,
          message: prompt,
          studioPathname: studioPathnameRef.current,
          workspace,
          signal,
          handlers: {
            onAssistantEvent: (event) => {
              if (signal.aborted || !isCurrentRun()) return;
              setArtifacts((current) => updateAssistantChatArtifactsFromEvent(current, event));
              appendAssistantParts(getAssistantPartsFromAssistantEvent(event));
            },
            onPermissionRequest: (request) => {
              if (signal.aborted || !isCurrentRun()) return;
              prepareForUserInput();
              handlePermissionRequest(request);
            },
            onInputRequest: (request) => {
              if (signal.aborted || !isCurrentRun()) return;
              prepareForUserInput();
              handleInputRequest(request);
            },
            onPermissionExpired: (requestId) => {
              if (signal.aborted || !isCurrentRun()) return;
              dispatchBlocking({ type: 'expire_permission', requestId });
            },
            onInputExpired: (requestId) => {
              if (signal.aborted || !isCurrentRun()) return;
              if (activeInputRef.current?.requestId === requestId) {
                activeInputRef.current = null;
              }
              dispatchBlocking({ type: 'expire_input', requestId });
            },
            onDone: () => {
              doneReceived = true;
            },
            onError: (error) => {
              throw error;
            },
          },
        });
        void queryClient.invalidateQueries({
          queryKey: getAssistantHistorySessionsQueryKey(workspace ?? 'default'),
        });
        if (!doneReceived && !signal.aborted) {
          throw new Error(ASSISTANT_RUN_INTERRUPTED_ERROR);
        }
      } finally {
        if (isCurrentRun()) {
          activeInputRef.current = null;
          dispatchBlocking({ type: 'reset' });
        }
      }

      return { status: doneReceived ? COMPLETE_STATUS : CANCELLED_STATUS };
    },
    [
      clearStudioNavigationRequest,
      ensureSessionId,
      handleInputRequest,
      handlePermissionRequest,
      queryClient,
      setArtifacts,
      workspace,
    ]
  );

  const {
    appendUserMessage,
    handleReset: resetThread,
    isRunning,
    replaceMessages,
    runtime,
    submitPrompt,
  } = useCustomAssistantChatRuntime({
    initialMessages: options?.initialMessages,
    describeError: describeAssistantRunError,
    onBeforeRun: requestStudioNavigationDecision,
    onError,
    onRun: handleRun,
  });

  const resolveInputRequest = useCallback(
    async ({
      decision,
      displayText,
    }: {
      decision: AssistantInputDecision;
      displayText?: string;
    }) => {
      if (!sessionId || !activeInput) return;

      const resolvedSessionId = sessionId;
      const resolvedRequestId = activeInput.requestId;
      const trimmedDisplayText = displayText?.trim();
      dispatchBlocking({ type: 'set_input_status', status: 'submitting' });

      try {
        await resolveAssistantInput({
          sessionId,
          requestId: activeInput.requestId,
          decision,
        });
        if (
          sessionIdRef.current !== resolvedSessionId ||
          activeInputRef.current?.requestId !== resolvedRequestId
        ) {
          return;
        }
        if (!decision.skipped && decision.value) {
          const inputValue = decision.value;
          setArtifacts((current) =>
            updateAssistantChatArtifactsFromInputSelection(current, activeInput, inputValue)
          );
        }
        if (!decision.skipped && trimmedDisplayText) appendUserMessage(trimmedDisplayText);
        dispatchBlocking({
          type: 'clear_input',
          requestId: activeInput.requestId,
          dequeueNext: true,
        });
      } catch (error: unknown) {
        dispatchBlocking({ type: 'set_input_status', status: 'pending' });
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to resolve NeMo Assistant input';
        onError?.(new Error(errorMessage));
      }
    },
    [sessionId, activeInput, appendUserMessage, onError, setArtifacts]
  );

  const submitActiveDecision = useCallback(
    async ({
      approved,
      displayText,
      reason,
    }: {
      approved: boolean;
      displayText?: string;
      reason?: string;
    }): Promise<boolean> => {
      if (!sessionId || !activePermission) return false;

      const trimmedReason = reason?.trim();
      const trimmedDisplayText = displayText?.trim();
      dispatchBlocking({ type: 'set_decision_status', status: 'submitting' });

      try {
        await resolveAssistantPermission({
          sessionId,
          requestId: activePermission.requestId,
          decision: {
            approved,
            reason: approved ? undefined : trimmedReason || 'Denied by user',
          },
        });
        if (!approved && trimmedDisplayText) appendUserMessage(trimmedDisplayText);
        dispatchBlocking({
          type: 'clear_permission',
          requestId: activePermission.requestId,
          dequeueNext: true,
        });
        return true;
      } catch (error: unknown) {
        dispatchBlocking({ type: 'set_decision_status', status: 'pending' });
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to resolve NeMo Assistant permission';
        onError?.(new Error(errorMessage));
        return false;
      }
    },
    [sessionId, activePermission, appendUserMessage, onError]
  );

  const resolveAskUserQuestionDecision = useCallback(
    async (choice: AgentDecisionChoice, submission?: AgentDecisionSubmission) => {
      if (!activePermission || activeDecision?.kind !== 'ask-user-question') return;

      const activeQuestion = activeDecision.questions[activeDecision.questionIndex];
      const answer = submission?.text?.trim() || choice.label.trim();
      if (!answer) return;

      const answers = { ...activeDecision.answers, [activeQuestion.question]: answer };

      if (activeDecision.questionIndex < activeDecision.questions.length - 1) {
        dispatchBlocking({
          type: 'advance_question',
          state: { ...activeDecision, answers, questionIndex: activeDecision.questionIndex + 1 },
        });
        return;
      }

      const submitted = await submitActiveDecision({
        approved: false,
        reason: formatAskUserQuestionReason(activeDecision.questions, answers),
        displayText: formatAskUserQuestionDisplayText(activeDecision.questions, answers),
      });
      if (submitted) {
        setArtifacts((current) =>
          updateAssistantChatArtifactsFromSelections(current, activeDecision.questions, answers)
        );
      }
    },
    [activePermission, activeDecision, setArtifacts, submitActiveDecision]
  );

  const resolveDecisionRequest = useCallback(
    async (choice: AgentDecisionChoice, submission?: AgentDecisionSubmission) => {
      if (activeDecision?.kind === 'ask-user-question') {
        await resolveAskUserQuestionDecision(choice, submission);
        return;
      }
      const approved = choice.id === 'yes';
      const trimmedReason = submission?.text?.trim();
      await submitActiveDecision({
        approved,
        reason: approved ? undefined : trimmedReason || 'Denied by user',
        displayText: approved ? undefined : trimmedReason,
      });
    },
    [activeDecision, resolveAskUserQuestionDecision, submitActiveDecision]
  );

  const skipDecisionRequest = useCallback(
    () => void submitActiveDecision({ approved: false, reason: 'Skipped by user' }),
    [submitActiveDecision]
  );

  const skipInputRequest = useCallback(
    () => void resolveInputRequest({ decision: { skipped: true } }),
    [resolveInputRequest]
  );

  const handleReset = useCallback(() => {
    sessionIdRef.current = null;
    activeInputRef.current = null;
    setSessionId(null);
    onSessionIdChange?.(null);
    setArtifacts(createWorkspaceArtifacts(undefined, workspace));
    dispatchBlocking({ type: 'reset' });
    clearStudioNavigationRequest('cancel');
    resetThread();
  }, [clearStudioNavigationRequest, onSessionIdChange, resetThread, workspace]);

  const loadSession = useCallback(
    ({
      artifacts: nextArtifacts,
      messages,
      sessionId: nextSessionId,
    }: LoadAssistantSessionOptions) => {
      sessionIdRef.current = nextSessionId;
      activeInputRef.current = null;
      setSessionId(nextSessionId);
      onSessionIdChange?.(nextSessionId);
      setArtifacts(createWorkspaceArtifacts(nextArtifacts, workspace));
      dispatchBlocking({ type: 'reset' });
      clearStudioNavigationRequest('cancel');
      replaceMessages(messages);
    },
    [clearStudioNavigationRequest, onSessionIdChange, replaceMessages, workspace]
  );

  return {
    artifacts,
    decisionChoices,
    decisionRequest,
    decisionStatus,
    handleReset,
    inputRequest: activeInput,
    inputStatus,
    isRunning,
    loadSession,
    resolveDecisionRequest,
    resolveInputRequest,
    resolveStudioNavigationRequest,
    runtime,
    sessionId,
    skipDecisionRequest,
    skipInputRequest,
    studioNavigationRequest,
    studioNavigationStatus,
    submitPrompt,
  };
};

export type AssistantChatRuntime = ReturnType<typeof useAssistantChatRuntime>;
