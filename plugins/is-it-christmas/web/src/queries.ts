// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { getAnswer, localDate, type ApiHost } from './api';
import { PLUGIN_NAME } from './paths';

export const useIsItChristmas = (host: ApiHost) => {
  const date = localDate();
  return useQuery({
    queryKey: [PLUGIN_NAME, host.apiBaseUrl, host.workspaceId, date],
    queryFn: ({ signal }) => getAnswer(host, date, signal),
  });
};
