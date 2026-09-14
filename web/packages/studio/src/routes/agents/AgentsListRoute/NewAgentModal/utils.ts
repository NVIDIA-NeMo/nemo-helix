// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateAgentRequestConfig } from '@nemo/sdk/generated/agents/schema/CreateAgentRequestConfig';
import {
  AGENT_CONFIG_FILENAME,
  AGENT_SPEC_FILENAME,
  FABRIC_CONFIG_FORMAT,
  IGNORED_DIRECTORIES,
  IGNORED_EXTENSIONS,
  IGNORED_FILENAMES,
  MAX_AGENT_SPEC_BYTES,
  MAX_AGENT_SPEC_FILES,
  MAX_PICKED_FILES,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type {
  PickedFile,
  UploadAgentEntry,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import YAML from 'yaml';

/**
 * Convention only — the Agent entity stores no reference to it. Must match
 * `ethos_fileset_name` in the agents plugin, which is what deployments stage from.
 */
export const agentSpecFilesetName = (agentName: string): string => `${agentName}-ethos`;

export const tooManyPickedFiles = (pickedCount: number): string | undefined =>
  pickedCount > MAX_PICKED_FILES
    ? `That directory holds ${pickedCount.toLocaleString()} files, far more than an agent directory should. Point at the agent's own directory.`
    : undefined;

export const isIgnoredPath = (path: string): boolean => {
  const segments = path.split('/');
  if (segments.some((segment) => IGNORED_DIRECTORIES.has(segment))) return true;

  const filename = segments[segments.length - 1] ?? '';
  if (IGNORED_FILENAMES.has(filename)) return true;

  return IGNORED_EXTENSIONS.some((extension) => filename.endsWith(extension));
};

const pathCollator = new Intl.Collator();

/** The fileset holds the directory's contents, so the picked root is stripped from each path. */
export const collectAgentEntries = (picked: PickedFile[]): UploadAgentEntry[] => {
  const entries: UploadAgentEntry[] = [];

  for (const { file, relativePath } of picked) {
    const path = relativePath.split('/').slice(1).join('/') || file.name;
    if (!path || isIgnoredPath(path)) continue;
    entries.push({ path, file });
  }

  return entries.sort((left, right) => pathCollator.compare(left.path, right.path));
};

/** A directory picker reports the path on the File itself; a drop does not. */
export const pickedFromFileList = (files: File[]): PickedFile[] =>
  files.map((file) => ({ file, relativePath: file.webkitRelativePath || file.name }));

/**
 * Walk dropped directories into files.
 *
 * `dataTransfer.files` flattens a dropped folder to a useless zero-byte entry, so the
 * directory has to be traversed through `webkitGetAsEntry`. Paths are built during the
 * walk because a File produced this way has an empty `webkitRelativePath`.
 *
 * Traversal stops once the ceiling is passed: a mistaken drop of a large tree is the
 * same hazard as the equivalent pick, and here the reading is ours to abandon.
 */
export const pickedFromDataTransfer = async (items: DataTransferItem[]): Promise<PickedFile[]> => {
  const roots = items
    .map((item) => (item.kind === 'file' ? item.webkitGetAsEntry() : null))
    .filter((entry): entry is FileSystemEntry => entry !== null);

  const picked: PickedFile[] = [];
  const pending: FileSystemEntry[] = [...roots];

  while (pending.length > 0 && picked.length <= MAX_PICKED_FILES) {
    const entry = pending.shift();
    if (!entry) break;

    if (entry.isFile) {
      const file = await readEntryFile(entry as FileSystemFileEntry);
      if (file) picked.push({ file, relativePath: entry.fullPath.replace(/^\//, '') });
      continue;
    }

    if (entry.isDirectory) {
      if (isIgnoredPath(entry.name)) continue;
      pending.push(...(await readDirectoryEntries(entry as FileSystemDirectoryEntry)));
    }
  }

  return picked;
};

const readEntryFile = (entry: FileSystemFileEntry): Promise<File | undefined> =>
  new Promise((resolve) => entry.file(resolve, () => resolve(undefined)));

const readDirectoryEntries = async (
  directory: FileSystemDirectoryEntry
): Promise<FileSystemEntry[]> => {
  const reader = directory.createReader();
  const all: FileSystemEntry[] = [];

  // readEntries yields a batch at a time and signals completion with an empty batch.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve) =>
      reader.readEntries(resolve, () => resolve([]))
    );
    if (batch.length === 0) return all;
    all.push(...batch);
    if (all.length > MAX_PICKED_FILES) return all;
  }
};

export const totalEntryBytes = (entries: UploadAgentEntry[]): number =>
  entries.reduce((total, entry) => total + entry.file.size, 0);

export const validateAgentEntries = (entries: UploadAgentEntry[]): string | undefined => {
  if (entries.length === 0) return 'That directory has no uploadable files.';

  // Two directories dropped at once merge, and a path they share would upload twice.
  const seen = new Set<string>();
  for (const { path } of entries) {
    if (seen.has(path)) {
      return `That selection holds more than one ${path}. Select a single agent directory.`;
    }
    seen.add(path);
  }

  if (!entries.some((entry) => entry.path === AGENT_CONFIG_FILENAME)) {
    return `No ${AGENT_CONFIG_FILENAME} at the top level of that directory.`;
  }

  if (entries.length > MAX_AGENT_SPEC_FILES) {
    return `That directory holds ${entries.length} files; the limit is ${MAX_AGENT_SPEC_FILES}. Point at a directory containing only the agent's own files.`;
  }

  const bytes = totalEntryBytes(entries);
  if (bytes > MAX_AGENT_SPEC_BYTES) {
    return `That directory is ${Math.round(bytes / 1000)} KB; the limit is ${Math.round(MAX_AGENT_SPEC_BYTES / 1000)} KB. Point at a directory containing only the agent's own files.`;
  }

  return undefined;
};

// Container deployments read every staged file as text and fail on a decode error.
export const findNonUtf8Path = async (entries: UploadAgentEntry[]): Promise<string | undefined> => {
  const decoder = new TextDecoder('utf-8', { fatal: true });

  const offenders = await Promise.all(
    entries.map(async (entry) => {
      if (entry.path.split('/').pop() === AGENT_SPEC_FILENAME) return undefined;
      try {
        decoder.decode(await entry.file.arrayBuffer());
        return undefined;
      } catch {
        return entry.path;
      }
    })
  );

  return offenders.find((path) => path !== undefined);
};

export class AgentConfigParseError extends Error {}

export const parseAgentConfig = (text: string): CreateAgentRequestConfig => {
  let parsed: unknown;
  try {
    parsed = YAML.parse(text);
  } catch (error) {
    throw new AgentConfigParseError(
      `${AGENT_CONFIG_FILENAME} is not valid YAML: ${(error as Error).message}`
    );
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new AgentConfigParseError(`${AGENT_CONFIG_FILENAME} must contain a YAML mapping.`);
  }

  const config = parsed as CreateAgentRequestConfig;
  const configFormat = config.config_format;
  if (configFormat !== FABRIC_CONFIG_FORMAT) {
    throw new AgentConfigParseError(
      `${AGENT_CONFIG_FILENAME} must set config_format: ${FABRIC_CONFIG_FORMAT}${
        typeof configFormat === 'string' ? ` (found ${configFormat})` : ''
      }.`
    );
  }

  return config;
};

export const agentNameFromConfig = (config: Record<string, unknown>): string | undefined =>
  typeof config.name === 'string' && config.name.trim() ? config.name.trim() : undefined;
