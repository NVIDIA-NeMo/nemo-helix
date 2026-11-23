// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

/** What an agent image built off-platform must match to run on this platform. */
export interface ClusterInfo {
  /** The nemo-helix release the platform runs, absent when it runs an unpublished build. */
  releaseVersion?: string;
  /** Docker platform of the platform's nodes, e.g. `linux/amd64`, absent when not reported. */
  architecture?: string;
}

export const CLUSTER_INFO_QUERY_KEY = ['cluster-info'] as const;

/** Version and architecture endpoint exposed by the platform runner; it is not part of the OpenAPI spec. */
export const getClusterInfoUrl = (): string =>
  `${PLATFORM_BASE_URL.replace(/\/+$/, '')}/cluster-info`;

// A version a package index can serve: no `dev` segment or `+local` identifier.
const PUBLISHED_RELEASE = /^v?(\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?)$/;

const stringField = (data: unknown, key: string): string | undefined => {
  if (typeof data !== 'object' || data === null || !(key in data)) return undefined;
  const value: unknown = Reflect.get(data, key);
  return typeof value === 'string' && value ? value : undefined;
};

export const parseClusterInfo = (data: unknown): ClusterInfo => ({
  releaseVersion: PUBLISHED_RELEASE.exec(stringField(data, 'platform_version') ?? '')?.[1],
  architecture: stringField(data, 'architecture'),
});

export const fetchClusterInfo = async (): Promise<ClusterInfo> => {
  const { data } = await axios.get<unknown>(getClusterInfoUrl());
  return parseClusterInfo(data);
};

export const useClusterInfo = () =>
  useQuery({
    queryKey: CLUSTER_INFO_QUERY_KEY,
    queryFn: fetchClusterInfo,
    retry: false,
    staleTime: Infinity,
  });
