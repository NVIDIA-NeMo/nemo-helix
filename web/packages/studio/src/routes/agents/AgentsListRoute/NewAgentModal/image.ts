// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentImageBundleFile } from '@nemo/sdk/generated/agents/schema/AgentImageBundleFile';
import type { UploadAgentEntry } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import {
  isIgnoredPath,
  toAgentName,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';

const PATH_COMPONENT = '[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*';
const REGISTRY = '[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?(?::[0-9]+)?';
const TAG = '[\\w][\\w.-]{0,127}';
const DIGEST = 'sha256:[a-f0-9]{64}';

// Docker's reference grammar, minus the uncommon digest algorithms.
const IMAGE_REFERENCE = new RegExp(
  `^(?:${REGISTRY}/)?${PATH_COMPONENT}(?:/${PATH_COMPONENT})*(?::${TAG})?(?:@${DIGEST})?$`
);

const INTERNAL_HOST_SUFFIXES = ['.localhost', '.local', '.internal', '.svc', '.cluster.local'];
const PRIVATE_IPV4 =
  /^(?:0\.|10\.|127\.|169\.254\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.)/;

// Mirrors the platform's refusal of its own network, so the field says so before a round trip.
const registryProblem = (reference: string): string | undefined => {
  const [first = '', ...rest] = reference.split('@')[0]?.split('/') ?? [];
  const namesRegistry = rest.length > 0 && (first.includes('.') || first.includes(':'));
  if (!namesRegistry && first !== 'localhost') {
    return undefined;
  }
  const host = first.replace(/:\d+$/, '').toLowerCase();
  const internal =
    host === 'localhost' ||
    INTERNAL_HOST_SUFFIXES.some((suffix) => host.endsWith(suffix)) ||
    !host.includes('.') ||
    PRIVATE_IPV4.test(host);
  return internal ? 'Images on private or local registries cannot be read.' : undefined;
};

export const imageReferenceProblem = (reference: string): string | undefined => {
  const trimmed = reference.trim();
  if (!trimmed) {
    return undefined;
  }
  if (/^[a-z]+:\/\//i.test(trimmed)) {
    return 'Leave out the scheme: use registry/repository:tag, not a URL.';
  }
  if (!IMAGE_REFERENCE.test(trimmed)) {
    return `"${trimmed}" is not an image reference. Use registry/repository:tag, for example docker.io/org/agent:1.0.`;
  }
  return registryProblem(trimmed);
};

const imageRepositoryName = (reference: string): string => {
  const withoutDigest = reference.trim().split('@')[0] ?? '';
  const lastSegment = withoutDigest.split('/').pop() ?? '';
  return lastSegment.split(':')[0] ?? '';
};

/** A starting point for the agent name, which the user can still edit before submitting. */
export const agentNameFromImage = (reference: string): string =>
  toAgentName(imageRepositoryName(reference));

/** The image's agent bundle as upload entries, so it is validated and stored like picked files. */
export const imageBundleEntries = (files: AgentImageBundleFile[]): UploadAgentEntry[] =>
  files
    .filter(({ path }) => !isIgnoredPath(path))
    .map(({ path, content }) => ({
      path,
      file: new File(
        [Uint8Array.from(atob(content), (char) => char.charCodeAt(0))],
        path.split('/').pop() ?? path
      ),
    }));
