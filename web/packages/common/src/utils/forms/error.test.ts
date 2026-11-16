// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
import { logger } from '@nemo/common/src/utils/logger';
import type { FieldErrors } from 'react-hook-form';

// Mock the websiteLogger
vi.mock('@nemo/common/src/utils/logger', () => ({
  logger: {
    error: vi.fn(),
  },
}));

describe('handleFormErrorsGeneric', () => {
  const mockWebsiteLogger = logger as unknown as { error: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should handle form errors with default title', () => {
    const errors: FieldErrors = {
      email: {
        type: 'required',
        message: 'Email is required',
      },
      password: {
        type: 'minLength',
        message: 'Password must be at least 8 characters',
      },
    };

    const handler = handleFormErrorsGeneric({});
    handler(errors);

    expect(mockWebsiteLogger.error).toHaveBeenCalledWith(
      'Form Errors: \nemail: {"type":"required","message":"Email is required"}\npassword: {"type":"minLength","message":"Password must be at least 8 characters"}'
    );
  });

  it('should handle form errors with custom title', () => {
    const errors: FieldErrors = {
      username: {
        type: 'pattern',
        message: 'Username must contain only letters and numbers',
      },
    };

    const handler = handleFormErrorsGeneric({ title: 'User Registration Errors' });
    handler(errors);

    expect(mockWebsiteLogger.error).toHaveBeenCalledWith(
      'User Registration Errors: \nusername: {"type":"pattern","message":"Username must contain only letters and numbers"}'
    );
  });

  it('should handle empty errors object', () => {
    const errors: FieldErrors = {};

    const handler = handleFormErrorsGeneric({ title: 'Empty Errors' });
    handler(errors);

    expect(mockWebsiteLogger.error).toHaveBeenCalledWith('Empty Errors: \n');
  });

  it('should handle errors with nested objects', () => {
    const errors: FieldErrors = {
      profile: {
        firstName: {
          type: 'required',
          message: 'First name is required',
        },
        lastName: {
          type: 'required',
          message: 'Last name is required',
        },
      },
    };

    const handler = handleFormErrorsGeneric({ title: 'Profile Errors' });
    handler(errors);

    expect(mockWebsiteLogger.error).toHaveBeenCalledWith(
      'Profile Errors: \nprofile: {"firstName":{"type":"required","message":"First name is required"},"lastName":{"type":"required","message":"Last name is required"}}'
    );
  });

  it('should handle errors with array values', () => {
    const errors: FieldErrors = {
      tags: {
        type: 'required',
        message: 'Tags are required',
      },
    };

    const handler = handleFormErrorsGeneric({ title: 'Array Errors' });
    handler(errors);

    expect(mockWebsiteLogger.error).toHaveBeenCalledWith(
      'Array Errors: \ntags: {"type":"required","message":"Tags are required"}'
    );
  });
});
