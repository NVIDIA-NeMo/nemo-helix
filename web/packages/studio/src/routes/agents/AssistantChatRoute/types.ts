// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface AssistantStreamHandlers {
  onAssistantEvent: (event: unknown) => void;
  onInputRequest: (request: AssistantInputRequest) => void;
  onPermissionRequest: (request: AssistantPermissionRequest) => void;
  onInputExpired?: (requestId: string) => void;
  onPermissionExpired?: (requestId: string) => void;
  onDone: () => void;
  onError: (error: Error) => void;
}

export interface AssistantPermissionRequest {
  requestId: string;
  toolName: string;
  input: Record<string, unknown>;
  toolUseId?: string;
}

export interface AssistantPermissionDecision {
  approved: boolean;
  reason?: string;
  updatedInput?: Record<string, unknown>;
}

export type AssistantInputRequestKind = 'agent' | 'eval_config' | 'dataset_file' | 'model';

export interface AssistantInputRequest {
  requestId: string;
  kind: AssistantInputRequestKind;
  input: Record<string, unknown>;
}

export interface AssistantInputDecision {
  skipped?: boolean;
  value?: Record<string, unknown>;
}

export interface AssistantChatRouteState {
  initialPrompt?: string;
}

export interface AssistantChatSelectionArtifact {
  label: string;
  value: string;
}

export interface AssistantChatFileArtifact {
  action: string;
  path: string;
}

export interface AssistantChatLinkArtifact {
  label: string;
  destination?: string;
  href?: string;
}

export interface AssistantChatJobArtifact {
  name: string;
  job_type?: string;
  source?: string;
  href?: string;
}

export type AssistantChatModelSource = 'assistant' | 'selection' | 'spec';

export interface AssistantChatArtifacts {
  agent?: string;
  model?: string;
  model_source?: AssistantChatModelSource;
  assistant_model?: string;
  workspace?: string;
  selections: AssistantChatSelectionArtifact[];
  files: AssistantChatFileArtifact[];
  links: AssistantChatLinkArtifact[];
  jobs: AssistantChatJobArtifact[];
  tools: string[];
}

export interface AssistantHistorySession {
  session_id: string;
  mtime: number;
  title?: string;
  first_prompt: string;
  message_count: number;
  token_count: number;
  tool_call_count: number;
  tool_calls: string[];
  chat_artifacts: AssistantChatArtifacts;
}

export interface AssistantSkill {
  name: string;
  claude_name: string;
  description: string;
  source: string;
  source_path?: string | null;
  install_path: string;
  installed: boolean;
}

export interface AssistantUserHistoryItem {
  kind: 'user';
  text: string;
}

export interface AssistantResponseTextPart {
  type: 'text';
  text: string;
}

export interface AssistantResponseToolUsePart {
  type: 'tool_use';
  id?: string;
  name: string;
  input: Record<string, unknown>;
}

export interface AssistantResponseThinkingPart {
  type: 'thinking';
  thinking: string;
}

export type AssistantResponseHistoryPart =
  | AssistantResponseTextPart
  | AssistantResponseThinkingPart
  | AssistantResponseToolUsePart;

export interface AssistantResponseHistoryItem {
  kind: 'assistant';
  parts: AssistantResponseHistoryPart[];
}

export type AssistantSessionHistoryItem = AssistantUserHistoryItem | AssistantResponseHistoryItem;

export interface AssistantSessionHistory {
  session_id: string;
  items: AssistantSessionHistoryItem[];
  chat_artifacts: AssistantChatArtifacts;
}
