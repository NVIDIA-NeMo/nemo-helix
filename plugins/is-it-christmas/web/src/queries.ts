// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { getAnswer, getPanic, getPicture, localDate, type ApiHost, type Mood, type PictureKind } from './api';
import { PLUGIN_NAME } from './paths';

export const useIsItChristmas = (host: ApiHost) => {
  const date = localDate();
  return useQuery({
    queryKey: [PLUGIN_NAME, host.apiBaseUrl, host.workspaceId, date],
    queryFn: ({ signal }) => getAnswer(host, date, signal),
  });
};

/** The panic is the point, so never retry it or refetch it behind the viewer's back. */
export const useRustPanic = (host: ApiHost) =>
  useQuery({
    queryKey: [PLUGIN_NAME, host.apiBaseUrl, host.workspaceId, 'panic'],
    queryFn: ({ signal }) => getPanic(host, signal),
    retry: false,
    refetchOnWindowFocus: false,
  });

/** Decoration only: a missing picture must never break the page, so do not retry it. */
export const usePicture = (host: ApiHost, kind: PictureKind, mood?: Mood, enabled = true) =>
  useQuery({
    queryKey: [PLUGIN_NAME, host.apiBaseUrl, host.workspaceId, kind, mood ?? 'default'],
    queryFn: ({ signal }) => getPicture(host, kind, { mood, signal }),
    enabled,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: Infinity,
  });
