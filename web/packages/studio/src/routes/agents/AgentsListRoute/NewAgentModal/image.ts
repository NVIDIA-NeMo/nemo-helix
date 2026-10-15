// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toAgentName } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';

const PATH_COMPONENT = '[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*';
const REGISTRY = '[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?(?::[0-9]+)?';
const TAG = '[\\w][\\w.-]{0,127}';
const DIGEST = 'sha256:[a-f0-9]{64}';

// Docker's reference grammar, minus the uncommon digest algorithms.
const IMAGE_REFERENCE = new RegExp(
  `^(?:${REGISTRY}/)?${PATH_COMPONENT}(?:/${PATH_COMPONENT})*(?::${TAG})?(?:@${DIGEST})?$`
);

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
  return undefined;
};

/** The repository's last path segment, without registry, tag or digest. */
const imageRepositoryName = (reference: string): string => {
  const withoutDigest = reference.trim().split('@')[0] ?? '';
  const lastSegment = withoutDigest.split('/').pop() ?? '';
  return lastSegment.split(':')[0] ?? '';
};

/** A starting point for the agent name, which the user can still edit before submitting. */
export const agentNameFromImage = (reference: string): string =>
  toAgentName(imageRepositoryName(reference));
