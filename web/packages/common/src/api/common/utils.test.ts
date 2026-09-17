// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getErrorMessage,
  isValidationErrorArray,
  swallowConflict,
} from '@nemo/common/src/api/common/utils';
import { AxiosError, AxiosHeaders, InternalAxiosRequestConfig } from 'axios';

describe('isValidationErrorArray', () => {
  it('returns true for valid validation error array', () => {
    const detail = [
      { msg: 'Field is required', type: 'value_error', loc: ['body', 'name'] },
      { msg: 'Invalid format', type: 'type_error', loc: ['body', 'email'] },
    ];

    expect(isValidationErrorArray(detail)).toBe(true);
  });

  it('returns false for empty array', () => {
    expect(isValidationErrorArray([])).toBe(false);
  });

  it('returns false for non-array', () => {
    expect(isValidationErrorArray('error')).toBe(false);
    expect(isValidationErrorArray(null)).toBe(false);
    expect(isValidationErrorArray(undefined)).toBe(false);
  });

  it('returns false for array with missing properties', () => {
    const detail = [{ msg: 'Error' }]; // missing type and loc
    expect(isValidationErrorArray(detail)).toBe(false);
  });
});

describe('getErrorMessage', () => {
  const createAxiosError = (
    options: {
      message?: string;
      code?: string;
      response?: {
        status: number;
        statusText: string;
        data?: { detail?: unknown };
      };
      config?: {
        method?: string;
        url?: string;
      };
      cause?: Error;
    } = {}
  ): AxiosError => {
    const headers = new AxiosHeaders();
    const config: InternalAxiosRequestConfig = {
      headers,
      method: options.config?.method,
      url: options.config?.url,
    };

    const error = new AxiosError(
      options.message ?? 'Request failed',
      options.code,
      config,
      undefined,
      options.response
        ? {
            status: options.response.status,
            statusText: options.response.statusText,
            data: options.response.data ?? {},
            headers: {},
            config,
          }
        : undefined
    );

    if (options.cause) {
      error.cause = options.cause;
    }

    return error;
  };

  describe('with validation errors', () => {
    it('joins multiple validation error messages with field paths', () => {
      const error = createAxiosError({
        response: {
          status: 422,
          statusText: 'Unprocessable Entity',
          data: {
            detail: [
              { msg: 'Field is required', type: 'value_error', loc: ['body', 'name'] },
              { msg: 'Invalid email format', type: 'type_error', loc: ['body', 'email'] },
            ],
          },
        },
      });

      expect(getErrorMessage(error)).toBe('name: Field is required; email: Invalid email format');
    });

    it('handles single validation error with field path', () => {
      const error = createAxiosError({
        response: {
          status: 422,
          statusText: 'Unprocessable Entity',
          data: {
            detail: [{ msg: 'Name is required', type: 'value_error', loc: ['body', 'name'] }],
          },
        },
      });

      expect(getErrorMessage(error)).toBe('name: Name is required');
    });

    it('includes array index in nested field path', () => {
      const error = createAxiosError({
        response: {
          status: 422,
          statusText: 'Unprocessable Entity',
          data: {
            detail: [
              {
                msg: 'Score name is required',
                type: 'value_error',
                loc: ['body', 'scores', 0, 'name'],
              },
            ],
          },
        },
      });

      expect(getErrorMessage(error)).toBe('scores.0.name: Score name is required');
    });

    it('includes field path when no body prefix is present', () => {
      const error = createAxiosError({
        response: {
          status: 422,
          statusText: 'Unprocessable Entity',
          data: {
            detail: [
              {
                msg: 'String should match pattern',
                type: 'string_pattern_mismatch',
                loc: ['model'],
              },
            ],
          },
        },
      });

      expect(getErrorMessage(error)).toBe('model: String should match pattern');
    });

    it('returns just the message when loc only contains body', () => {
      const error = createAxiosError({
        response: {
          status: 422,
          statusText: 'Unprocessable Entity',
          data: {
            detail: [{ msg: 'Invalid request body', type: 'value_error', loc: ['body'] }],
          },
        },
      });

      expect(getErrorMessage(error)).toBe('Invalid request body');
    });

    it('returns just the message when loc is empty', () => {
      const error = createAxiosError({
        response: {
          status: 422,
          statusText: 'Unprocessable Entity',
          data: {
            detail: [{ msg: 'Validation failed', type: 'value_error', loc: [] }],
          },
        },
      });

      expect(getErrorMessage(error)).toBe('Validation failed');
    });
  });

  describe('with string detail', () => {
    it('returns the detail string directly', () => {
      const error = createAxiosError({
        response: {
          status: 400,
          statusText: 'Bad Request',
          data: { detail: 'Dataset not found' },
        },
      });

      expect(getErrorMessage(error)).toBe('Dataset not found');
    });
  });

  describe('with network errors', () => {
    it('formats network error with code and message', () => {
      const error = createAxiosError({
        message: 'Network Error',
        code: 'ERR_NETWORK',
      });

      expect(getErrorMessage(error)).toBe('[ERR_NETWORK] Network Error');
    });

    it('includes request method and URL when available', () => {
      const error = createAxiosError({
        message: 'Network Error',
        code: 'ERR_NETWORK',
        config: {
          method: 'get',
          url: '/api/intake/entries',
        },
      });

      expect(getErrorMessage(error)).toBe('[ERR_NETWORK] Network Error (GET /api/intake/entries)');
    });

    it('includes cause message when available', () => {
      const error = createAxiosError({
        message: 'Network Error',
        code: 'ERR_NETWORK',
        config: {
          method: 'post',
          url: '/api/datasets',
        },
        cause: new Error('ENOTFOUND'),
      });

      expect(getErrorMessage(error)).toBe(
        '[ERR_NETWORK] Network Error (POST /api/datasets) - ENOTFOUND'
      );
    });

    it('handles timeout errors', () => {
      const error = createAxiosError({
        message: 'timeout of 30000ms exceeded',
        code: 'ECONNABORTED',
        config: {
          method: 'get',
          url: '/api/models',
        },
      });

      expect(getErrorMessage(error)).toBe(
        '[ECONNABORTED] timeout of 30000ms exceeded (GET /api/models)'
      );
    });

    it('handles network error without code', () => {
      const error = createAxiosError({
        message: 'Network Error',
      });

      expect(getErrorMessage(error)).toBe('Network Error');
    });
  });

  describe('with HTTP error responses without detail', () => {
    it('returns status and statusText', () => {
      const error = createAxiosError({
        response: {
          status: 500,
          statusText: 'Internal Server Error',
          data: {},
        },
      });

      expect(getErrorMessage(error)).toBe('500 Internal Server Error');
    });

    it('handles 404 errors', () => {
      const error = createAxiosError({
        response: {
          status: 404,
          statusText: 'Not Found',
          data: {},
        },
      });

      expect(getErrorMessage(error)).toBe('404 Not Found');
    });
  });

  describe('with regular Error', () => {
    it('returns the error message', () => {
      const error = new Error('Something went wrong');

      expect(getErrorMessage(error)).toBe('Something went wrong');
    });

    it('prefers the error message over the fallback', () => {
      const error = new Error('boom');

      expect(getErrorMessage(error, 'fallback')).toBe('boom');
    });

    it('falls back only when the error carries no message', () => {
      const error = new Error('');

      expect(getErrorMessage(error, 'fallback')).toBe('fallback');
    });
  });
});

describe('swallowConflict', () => {
  const axiosErrorWithStatus = (status: number): AxiosError => {
    const config: InternalAxiosRequestConfig = { headers: new AxiosHeaders() };
    return new AxiosError('Request failed', undefined, config, undefined, {
      status,
      statusText: '',
      data: {},
      headers: {},
      config,
    });
  };

  it('resolves to the value when the promise succeeds', async () => {
    await expect(swallowConflict(Promise.resolve('created'))).resolves.toBe('created');
  });

  it('resolves to undefined when the promise rejects with a 409', async () => {
    await expect(
      swallowConflict(Promise.reject(axiosErrorWithStatus(409)))
    ).resolves.toBeUndefined();
  });

  it('rethrows AxiosErrors that are not 409', async () => {
    const error = axiosErrorWithStatus(500);
    await expect(swallowConflict(Promise.reject(error))).rejects.toBe(error);
  });

  it('rethrows non-Axios errors', async () => {
    const error = new Error('network down');
    await expect(swallowConflict(Promise.reject(error))).rejects.toBe(error);
  });
});
