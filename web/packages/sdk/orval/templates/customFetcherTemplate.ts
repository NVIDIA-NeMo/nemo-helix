// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import axios from 'axios';
import type { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import qs from 'qs';
import { withHelixBrowserAuthAxios } from '../../src/utils/platformRequest';
import { resolveBrowserBaseUrl } from '../../src/utils/url';

interface RequestOptions extends AxiosRequestConfig {
  params?: Record<string, string | number | boolean | object | unknown>;
}

// Add X-Source and the configured browser authentication to all requests.
axios.interceptors.request.use((config) => withHelixBrowserAuthAxios(config));

const getBaseUrl = (): string | undefined => {
  // TEMPLATE_BASE_URL_CHECKS_START
  return resolveBrowserBaseUrl(undefined);
  // TEMPLATE_BASE_URL_CHECKS_END
};

const getUrl = (request: AxiosRequestConfig): string => {
  const baseUrl = getBaseUrl();
  const { url } = request;
  const fullUrl = `${baseUrl}${url}`;

  if (!baseUrl) {
    return fullUrl;
  }

  try {
    // Construct the full URL with base URL and query parameters
    return new URL(fullUrl).toString();
  } catch (error) {
    console.error('Invalid URL:', fullUrl, error);
    throw new Error(`Invalid URL format: ${fullUrl}`);
  }
};

export const customFetch = async <TData>(request: RequestOptions): Promise<TData> => {
  const requestUrl = getUrl(request);
  const response: AxiosResponse<TData> = await axios({
    ...request,
    url: requestUrl,
    paramsSerializer: {
      serialize: (params) => qs.stringify(params, { indices: false }),
    },
  });
  return response.data;
};

// https://orval.dev/reference/configuration/output#mutator
// In some case with react-query and swr you want to be able to override the return error type so you can also do it here like this
export type ErrorType<TError> = AxiosError<TError>;
