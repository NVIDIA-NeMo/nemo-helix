// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  isHuggingFaceUrl,
  isNgcUrl,
  parseHuggingFaceUrl,
  parseNgcUrl,
} from '@studio/util/storageConfigFromUrl';
import { useQuery } from '@tanstack/react-query';

export interface RemoteRepoMetadata {
  /** Last segment of the canonical remote identifier; suitable for use as a
   *  fileset name after running through `toValidFilesetName`. */
  slug: string;
  /** Remote description. Only populated for HuggingFace public repos in phase 1
   *  — NGC + private HF require the backend preview endpoint (see nhx-1tk). */
  description: string | null;
}

interface HfApiResponse {
  cardData?: { description?: string } | null;
}

async function fetchHuggingFaceMetadata(
  repoId: string,
  repoType: 'dataset' | 'model' | 'space',
  signal: AbortSignal
): Promise<RemoteRepoMetadata | null> {
  // Spaces don't have an obvious dataset/model description endpoint; skip.
  if (repoType === 'space') {
    return { slug: repoId.split('/').pop() ?? repoId, description: null };
  }
  const resource = repoType === 'dataset' ? 'datasets' : 'models';
  const response = await fetch(`https://huggingface.co/api/${resource}/${repoId}`, {
    signal,
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    // 404 / 401 — repo may be private or non-existent. Fall back to slug-only.
    return { slug: repoId.split('/').pop() ?? repoId, description: null };
  }
  const body = (await response.json()) as HfApiResponse;
  // Only the YAML frontmatter description is short enough to fit the backend's
  // 255-char cap. Falling back to the README body would force an ugly
  // mid-sentence truncation on auto-fill, so leave the field empty instead and
  // let the user type/paste their own summary.
  const description = body.cardData?.description?.trim() || null;
  return {
    slug: repoId.split('/').pop() ?? repoId,
    description,
  };
}

function deriveNgcMetadata(target: string): RemoteRepoMetadata {
  // NGC description fetch requires API-key headers; the secret is not
  // available client-side. Slug-only here. Backend preview (nhx-1tk) closes
  // the gap.
  return { slug: target, description: null };
}

/** Resolve a remote repo URL into a name slug (+ description when feasible
 *  client-side). Returns `undefined` until the URL is valid + recognised.
 *
 *  Returns `null` data if the URL is recognised but the fetch failed —
 *  callers should treat that as "no auto-fill available" rather than an
 *  error to surface to the user. */
export function useRemoteRepoMetadata(url: string | undefined, enabled: boolean) {
  const trimmed = url?.trim() ?? '';
  return useQuery({
    queryKey: ['remoteRepoMetadata', trimmed],
    queryFn: async ({ signal }): Promise<RemoteRepoMetadata | null> => {
      let parsed: URL;
      try {
        parsed = new URL(trimmed);
      } catch {
        return null;
      }
      if (isHuggingFaceUrl(parsed)) {
        try {
          const { repoId, repoType } = parseHuggingFaceUrl(parsed);
          return await fetchHuggingFaceMetadata(repoId, repoType, signal);
        } catch {
          return null;
        }
      }
      if (isNgcUrl(parsed)) {
        try {
          const { target } = parseNgcUrl(parsed);
          return deriveNgcMetadata(target);
        } catch {
          return null;
        }
      }
      return null;
    },
    enabled: enabled && trimmed.length > 0,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });
}
