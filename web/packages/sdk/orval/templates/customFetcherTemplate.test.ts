// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InternalAxiosRequestConfig } from 'axios';

type RequestInterceptor = (
  config: InternalAxiosRequestConfig
) => InternalAxiosRequestConfig | Promise<InternalAxiosRequestConfig>;

const mocks = vi.hoisted(() => ({
  axios: vi.fn(),
  requestUse: vi.fn(),
  withHelixBrowserAuthAxios: vi.fn(),
}));

vi.mock('axios', () => {
  const axios = mocks.axios as typeof mocks.axios & {
    interceptors: { request: { use: typeof mocks.requestUse } };
  };
  axios.interceptors = { request: { use: mocks.requestUse } };
  return { default: axios };
});

vi.mock('../../src/utils/platformRequest', () => ({
  withHelixBrowserAuthAxios: mocks.withHelixBrowserAuthAxios,
}));

vi.mock('../../src/utils/url', () => ({
  resolveBrowserBaseUrl: () => 'http://platform.example.test',
}));

const loadRequestInterceptor = async (): Promise<RequestInterceptor> => {
  vi.resetModules();
  mocks.requestUse.mockReset();
  await import('./customFetcherTemplate');
  const interceptor = mocks.requestUse.mock.calls[0]?.[0] as RequestInterceptor | undefined;
  if (!interceptor) throw new Error('Axios request interceptor was not registered');
  return interceptor;
};

describe('customFetcherTemplate auth interceptor', () => {
  beforeEach(() => {
    mocks.withHelixBrowserAuthAxios.mockReset();
    mocks.withHelixBrowserAuthAxios.mockImplementation(async (config) => config);
  });

  it('delegates request authentication to the platform request helper', async () => {
    const interceptor = await loadRequestInterceptor();
    const config = { headers: {} } as InternalAxiosRequestConfig;

    await interceptor(config);

    expect(mocks.withHelixBrowserAuthAxios).toHaveBeenCalledWith(config);
  });
});
