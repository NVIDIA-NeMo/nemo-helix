// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BASE_URL, PLATFORM_BASE_URL } from '@studio/constants/environment';
import {
  cleanAssistantArtifactText,
  createEmptyAssistantChatArtifacts,
  updateAssistantChatArtifactsFromHistoryItems,
} from '@studio/routes/agents/AssistantChatRoute/artifacts';
import { parseJsonObject, parseSseChunk } from '@studio/routes/agents/AssistantChatRoute/stream';
import type {
  AssistantResponseHistoryPart,
  AssistantChatArtifacts,
  AssistantChatFileArtifact,
  AssistantChatJobArtifact,
  AssistantChatLinkArtifact,
  AssistantChatModelSource,
  AssistantChatSelectionArtifact,
  AssistantHistorySession,
  AssistantInputDecision,
  AssistantInputRequest,
  AssistantPermissionDecision,
  AssistantPermissionRequest,
  AssistantSessionHistory,
  AssistantSessionHistoryItem,
  AssistantSkill,
  AssistantStreamHandlers,
} from '@studio/routes/agents/AssistantChatRoute/types';
import { User } from 'oidc-client-ts';

const ASSISTANT_API_BASE_PATH = '/apis/studio/v2/assistant';

export class AssistantSessionNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AssistantSessionNotFoundError';
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const assistantApiUrl = (path: string): string =>
  `${PLATFORM_BASE_URL}${ASSISTANT_API_BASE_PATH}${path}`;

const getAssistantAccessToken = (): string | undefined => {
  const authority = import.meta.env.VITE_AUTH_AUTHORITY;
  const clientId = import.meta.env.VITE_AUTH_CLIENT_ID;
  if (!authority || !clientId || typeof localStorage === 'undefined') return undefined;

  const oidcStorageKey = `oidc.user:${authority}:${clientId}`;
  const oidcStorage = localStorage.getItem(oidcStorageKey);
  if (!oidcStorage) return undefined;

  try {
    const user = User.fromStorageString(oidcStorage);
    return user?.access_token && !user.expired ? user.access_token : undefined;
  } catch {
    localStorage.removeItem(oidcStorageKey);
    return undefined;
  }
};

const assistantFetch = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const accessToken = getAssistantAccessToken();
  if (!accessToken) return fetch(input, init);

  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${accessToken}`);
  return fetch(input, { ...init, headers });
};

const getStudioBaseUrl = (): string | undefined => {
  if (typeof window === 'undefined') return undefined;

  const normalizedBaseUrl = BASE_URL.replace(/\/+$/, '');
  const basePath = normalizedBaseUrl && normalizedBaseUrl !== '/' ? normalizedBaseUrl : '';
  return `${window.location.origin}${basePath}`;
};

const getStudioPathname = (): string | undefined => {
  if (typeof window === 'undefined') return undefined;
  return window.location.pathname;
};

export const getAssistantHistorySessionsQueryKey = (workspace: string) =>
  ['assistant', 'history', 'sessions', workspace] as const;

export const ASSISTANT_SKILLS_QUERY_KEY = ['assistant', 'skills'] as const;

export const getAssistantSessionHistoryQueryKey = (sessionId: string) =>
  ['assistant', 'history', 'session', sessionId] as const;

const getResponseErrorMessage = async (response: Response, fallback: string): Promise<string> => {
  const text = await response.text();
  if (!text) return fallback;

  try {
    const body = JSON.parse(text) as unknown;
    if (isRecord(body) && typeof body.detail === 'string') return body.detail;
  } catch {
    return text;
  }

  return text;
};

export const createAssistantSession = async (workspace = 'default'): Promise<string> => {
  const params = new URLSearchParams({ workspace });
  const response = await assistantFetch(assistantApiUrl(`/sessions?${params.toString()}`), {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(
      await getResponseErrorMessage(response, 'Failed to create NeMo Assistant session')
    );
  }

  const body = (await response.json()) as unknown;
  if (!isRecord(body) || typeof body.session_id !== 'string') {
    throw new Error('NeMo Assistant session response did not include a session id');
  }

  return body.session_id;
};

const getString = (value: unknown): string => (typeof value === 'string' ? value : '');

const getNumber = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : 0;

const getStringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];

const getOptionalString = (value: unknown): string | undefined => {
  const text = getString(value).trim();
  return text || undefined;
};

const getOptionalArtifactString = (value: unknown): string | undefined => {
  const text = getOptionalString(value);
  return text ? cleanAssistantArtifactText(text) : undefined;
};

const parseModelSource = (value: unknown): AssistantChatModelSource | undefined => {
  if (value === 'copilot') return 'assistant';
  if (value === 'assistant' || value === 'selection' || value === 'spec') return value;
  return undefined;
};

const parseSelectionArtifact = (value: unknown): AssistantChatSelectionArtifact | undefined => {
  if (!isRecord(value)) return undefined;
  const label = getOptionalString(value.label);
  const artifactValue = getOptionalArtifactString(value.value);
  if (!label || !artifactValue) return undefined;
  return { label, value: artifactValue };
};

const parseFileArtifact = (value: unknown): AssistantChatFileArtifact | undefined => {
  if (!isRecord(value)) return undefined;
  const action = getOptionalString(value.action);
  const path = getOptionalString(value.path);
  if (!action || !path) return undefined;
  return { action, path };
};

const parseLinkArtifact = (value: unknown): AssistantChatLinkArtifact | undefined => {
  if (!isRecord(value)) return undefined;
  const label = getOptionalString(value.label);
  if (!label) return undefined;
  return {
    label,
    destination: getOptionalString(value.destination),
    href: getOptionalString(value.href),
  };
};

const parseJobArtifact = (value: unknown): AssistantChatJobArtifact | undefined => {
  if (!isRecord(value)) return undefined;
  const name = getOptionalArtifactString(value.name);
  if (!name) return undefined;
  return {
    name,
    job_type: getOptionalArtifactString(value.job_type),
    source: getOptionalArtifactString(value.source),
    href: getOptionalString(value.href),
  };
};

const parseArray = <T>(value: unknown, parseItem: (item: unknown) => T | undefined): T[] =>
  Array.isArray(value) ? value.map(parseItem).filter((item): item is T => item !== undefined) : [];

const parseChatArtifacts = (value: unknown): AssistantChatArtifacts => {
  if (!isRecord(value)) return createEmptyAssistantChatArtifacts();

  const modelSource = parseModelSource(value.model_source);
  const model = getOptionalArtifactString(value.model);
  const assistantModel =
    getOptionalArtifactString(value.assistant_model) ||
    getOptionalArtifactString(value.copilot_model) ||
    (modelSource === 'assistant' ? model : undefined);

  return {
    agent: getOptionalArtifactString(value.agent),
    model: modelSource === 'assistant' ? undefined : model,
    model_source: modelSource === 'assistant' ? undefined : modelSource,
    assistant_model: assistantModel,
    workspace: getOptionalArtifactString(value.workspace),
    selections: parseArray(value.selections, parseSelectionArtifact),
    files: parseArray(value.files, parseFileArtifact),
    links: parseArray(value.links, parseLinkArtifact),
    jobs: parseArray(value.jobs, parseJobArtifact),
    tools: getStringArray(value.tools).map(cleanAssistantArtifactText),
  };
};

const parseHistorySession = (value: unknown): AssistantHistorySession | undefined => {
  if (!isRecord(value)) return undefined;
  const sessionId = getString(value.session_id);
  if (!sessionId) return undefined;

  return {
    session_id: sessionId,
    mtime: getNumber(value.mtime),
    title: getOptionalString(value.title),
    first_prompt: getString(value.first_prompt),
    message_count: getNumber(value.message_count),
    token_count: getNumber(value.token_count),
    tool_call_count: getNumber(value.tool_call_count),
    tool_calls: getStringArray(value.tool_calls),
    chat_artifacts: parseChatArtifacts(value.chat_artifacts),
  };
};

const parseAssistantSkill = (value: unknown): AssistantSkill | undefined => {
  if (!isRecord(value)) return undefined;
  const name = getString(value.name);
  const claudeName = getString(value.claude_name);
  const installPath = getString(value.install_path);
  if (!name || !claudeName || !installPath) return undefined;

  return {
    name,
    claude_name: claudeName,
    description: getString(value.description),
    source: getString(value.source) || '-',
    source_path: getString(value.source_path) || undefined,
    install_path: installPath,
    installed: value.installed === true,
  };
};

const parseAssistantPart = (value: unknown): AssistantResponseHistoryPart | undefined => {
  if (!isRecord(value)) return undefined;

  if (value.type === 'text') {
    const text = getString(value.text);
    return text ? { type: 'text', text } : undefined;
  }

  if (value.type === 'thinking') {
    const thinking = getString(value.thinking);
    return thinking ? { type: 'thinking', thinking } : undefined;
  }

  if (value.type === 'tool_use') {
    return {
      type: 'tool_use',
      id: getString(value.id) || undefined,
      name: getString(value.name) || 'tool',
      input: isRecord(value.input) ? value.input : {},
    };
  }

  return undefined;
};

const parseSessionHistoryItem = (value: unknown): AssistantSessionHistoryItem | undefined => {
  if (!isRecord(value)) return undefined;

  if (value.kind === 'user') {
    const text = getString(value.text);
    return text ? { kind: 'user', text } : undefined;
  }

  if (value.kind === 'assistant' && Array.isArray(value.parts)) {
    const parts = value.parts
      .map(parseAssistantPart)
      .filter((part): part is AssistantResponseHistoryPart => part !== undefined);
    return parts.length ? { kind: 'assistant', parts } : undefined;
  }

  return undefined;
};

export const listAssistantHistorySessions = async (
  workspace = 'default'
): Promise<AssistantHistorySession[]> => {
  const params = new URLSearchParams({ workspace });
  const response = await assistantFetch(assistantApiUrl(`/history/sessions?${params.toString()}`));

  if (!response.ok) {
    throw new Error(
      await getResponseErrorMessage(response, 'Failed to load NeMo Assistant history')
    );
  }

  const body = (await response.json()) as unknown;
  if (!Array.isArray(body)) return [];

  return body
    .map(parseHistorySession)
    .filter((session): session is AssistantHistorySession => session !== undefined);
};

export const listAssistantSkills = async (): Promise<AssistantSkill[]> => {
  const response = await assistantFetch(assistantApiUrl('/skills'));

  if (!response.ok) {
    throw new Error(
      await getResponseErrorMessage(response, 'Failed to load NeMo Assistant skills')
    );
  }

  const body = (await response.json()) as unknown;
  if (!Array.isArray(body)) return [];

  return body
    .map(parseAssistantSkill)
    .filter((skill): skill is AssistantSkill => skill !== undefined);
};

export const getAssistantSessionHistory = async (
  sessionId: string,
  workspace = 'default'
): Promise<AssistantSessionHistory> => {
  const params = new URLSearchParams({ workspace });
  const response = await assistantFetch(
    assistantApiUrl(`/history/sessions/${encodeURIComponent(sessionId)}?${params.toString()}`)
  );

  if (!response.ok) {
    const message = await getResponseErrorMessage(
      response,
      'Failed to load NeMo Assistant session'
    );
    if (response.status === 404) {
      throw new AssistantSessionNotFoundError(message);
    }
    throw new Error(message);
  }

  const body = (await response.json()) as unknown;
  if (!isRecord(body)) {
    throw new Error('NeMo Assistant session history response was not an object');
  }

  const items = Array.isArray(body.items)
    ? body.items
        .map(parseSessionHistoryItem)
        .filter((item): item is AssistantSessionHistoryItem => item !== undefined)
    : [];

  return {
    session_id: getString(body.session_id) || sessionId,
    items,
    chat_artifacts: updateAssistantChatArtifactsFromHistoryItems(
      parseChatArtifacts(body.chat_artifacts),
      items
    ),
  };
};

export const deleteAssistantSessionHistory = async (
  sessionId: string,
  workspace = 'default'
): Promise<void> => {
  const params = new URLSearchParams({ workspace });
  const response = await assistantFetch(
    assistantApiUrl(`/history/sessions/${encodeURIComponent(sessionId)}?${params.toString()}`),
    { method: 'DELETE' }
  );

  if (!response.ok) {
    throw new Error(
      await getResponseErrorMessage(response, 'Failed to delete NeMo Assistant session')
    );
  }
};

const getStreamErrorMessage = (payload: unknown): string => {
  if (!isRecord(payload)) return 'NeMo Assistant stream failed';
  if (typeof payload.stderr === 'string' && payload.stderr) return payload.stderr;
  if (typeof payload.detail === 'string' && payload.detail) return payload.detail;
  if (typeof payload.message === 'string' && payload.message) return payload.message;
  return 'NeMo Assistant stream failed';
};

const parsePermissionRequest = (payload: unknown): AssistantPermissionRequest | undefined => {
  if (!isRecord(payload) || typeof payload.request_id !== 'string') return undefined;
  if (typeof payload.tool_name !== 'string' || !payload.tool_name) return undefined;
  if (!isRecord(payload.input) || Array.isArray(payload.input)) return undefined;

  return {
    requestId: payload.request_id,
    toolName: payload.tool_name,
    input: payload.input,
    toolUseId: typeof payload.tool_use_id === 'string' ? payload.tool_use_id : undefined,
  };
};

const parseInputRequest = (payload: unknown): AssistantInputRequest | undefined => {
  if (!isRecord(payload) || typeof payload.request_id !== 'string') return undefined;
  if (
    payload.kind !== 'agent' &&
    payload.kind !== 'eval_config' &&
    payload.kind !== 'dataset_file' &&
    payload.kind !== 'model'
  ) {
    return undefined;
  }
  if (!isRecord(payload.input) || Array.isArray(payload.input)) return undefined;

  return {
    requestId: payload.request_id,
    kind: payload.kind,
    input: payload.input,
  };
};

const handleSseEvent = (
  event: { event?: string; data: string },
  handlers: AssistantStreamHandlers
): boolean => {
  if (event.event === 'done') {
    handlers.onDone();
    return true;
  }

  if (event.event === 'error') {
    handlers.onError(new Error(getStreamErrorMessage(parseJsonObject(event.data))));
    return false;
  }

  if (event.event === 'permission_request') {
    const request = parsePermissionRequest(parseJsonObject(event.data));
    if (!request) {
      handlers.onError(new Error('NeMo Assistant permission request was malformed'));
      return false;
    }
    handlers.onPermissionRequest(request);
    return true;
  }

  if (event.event === 'input_request') {
    const request = parseInputRequest(parseJsonObject(event.data));
    if (!request) {
      handlers.onError(new Error('NeMo Assistant input request was malformed'));
      return false;
    }
    handlers.onInputRequest(request);
    return true;
  }

  if (event.event === 'permission_expired') {
    const payload = parseJsonObject(event.data);
    if (isRecord(payload) && typeof payload.request_id === 'string') {
      handlers.onPermissionExpired?.(payload.request_id);
    }
    return true;
  }

  if (event.event === 'input_expired') {
    const payload = parseJsonObject(event.data);
    if (isRecord(payload) && typeof payload.request_id === 'string') {
      handlers.onInputExpired?.(payload.request_id);
    }
    return true;
  }

  handlers.onAssistantEvent(parseJsonObject(event.data));
  return true;
};

export const resolveAssistantPermission = async ({
  sessionId,
  requestId,
  decision,
}: {
  sessionId: string;
  requestId: string;
  decision: AssistantPermissionDecision;
}): Promise<void> => {
  const response = await assistantFetch(
    assistantApiUrl(`/sessions/${sessionId}/permissions/${requestId}`),
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        approved: decision.approved,
        reason: decision.reason,
        updated_input: decision.updatedInput,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      await getResponseErrorMessage(response, 'Failed to resolve NeMo Assistant permission')
    );
  }
};

export const resolveAssistantInput = async ({
  decision,
  requestId,
  sessionId,
}: {
  decision: AssistantInputDecision;
  requestId: string;
  sessionId: string;
}): Promise<void> => {
  const response = await assistantFetch(
    assistantApiUrl(`/sessions/${sessionId}/inputs/${requestId}`),
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        skipped: decision.skipped,
        value: decision.value,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      await getResponseErrorMessage(response, 'Failed to resolve NeMo Assistant input')
    );
  }
};

export const streamAssistantMessage = async ({
  sessionId,
  message,
  studioBaseUrl,
  studioPathname,
  workspace,
  signal,
  handlers,
}: {
  sessionId: string;
  message: string;
  studioBaseUrl?: string;
  studioPathname?: string;
  workspace?: string;
  signal: AbortSignal;
  handlers: AssistantStreamHandlers;
}): Promise<void> => {
  const response = await assistantFetch(assistantApiUrl(`/sessions/${sessionId}/messages`), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      studio_base_url: studioBaseUrl ?? getStudioBaseUrl(),
      studio_pathname: studioPathname ?? getStudioPathname(),
      workspace,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(
      await getResponseErrorMessage(response, 'Failed to send NeMo Assistant message')
    );
  }
  if (!response.body) {
    throw new Error('NeMo Assistant response did not include a stream');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = '';
  let shouldCancelReader = true;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        shouldCancelReader = false;
        break;
      }

      buffered += decoder.decode(value, { stream: true });
      const parsed = parseSseChunk(buffered);
      buffered = parsed.rest;

      for (const event of parsed.events) {
        if (!handleSseEvent(event, handlers)) return;
      }
    }

    buffered += decoder.decode();
    if (buffered) {
      const parsed = parseSseChunk(`${buffered}\n\n`);
      for (const event of parsed.events) {
        if (!handleSseEvent(event, handlers)) return;
      }
    }
  } finally {
    if (shouldCancelReader) {
      await reader.cancel().catch(() => undefined);
    }
    reader.releaseLock();
  }
};
