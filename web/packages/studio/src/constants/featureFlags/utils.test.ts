/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import {
  booleanFlag,
  stringFlag,
  numberFlag,
  listFlag,
  parseFlags,
  EnvConfig,
} from '@studio/constants/featureFlags/utils';

describe('featureFlags utils', () => {
  describe('booleanFlag', () => {
    it('returns true when env var is "true"', () => {
      const flag = booleanFlag('TEST_FLAG');
      const result = flag.schema.safeParse('true');
      expect(result.success).toBe(true);
      expect(result.data).toBe(true);
    });

    it('returns false when env var is "false"', () => {
      const flag = booleanFlag('TEST_FLAG');
      const result = flag.schema.safeParse('false');
      expect(result.success).toBe(true);
      expect(result.data).toBe(false);
    });

    it('is case insensitive - "TRUE" returns true', () => {
      const flag = booleanFlag('TEST_FLAG');
      const result = flag.schema.safeParse('TRUE');
      expect(result.success).toBe(true);
      expect(result.data).toBe(true);
    });

    it('is case insensitive - "True" returns true', () => {
      const flag = booleanFlag('TEST_FLAG');
      const result = flag.schema.safeParse('True');
      expect(result.success).toBe(true);
      expect(result.data).toBe(true);
    });

    it('returns default value (false) when env var is undefined', () => {
      const flag = booleanFlag('TEST_FLAG');
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(true);
      expect(result.data).toBe(false);
    });

    it('returns custom default when specified', () => {
      const flag = booleanFlag('TEST_FLAG', true);
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(true);
      expect(result.data).toBe(true);
    });

    it('sets envVar and typeName correctly', () => {
      const flag = booleanFlag('MY_ENV_VAR');
      expect(flag.envVar).toBe('MY_ENV_VAR');
      expect(flag.typeName).toBe('boolean');
    });
  });

  describe('stringFlag', () => {
    it('returns env var value as-is when set', () => {
      const flag = stringFlag('TEST_FLAG', 'default');
      const result = flag.schema.safeParse('hello');
      expect(result.success).toBe(true);
      expect(result.data).toBe('hello');
    });

    it('returns default value when env var is undefined (with default)', () => {
      const flag = stringFlag('TEST_FLAG', 'default');
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(true);
      expect(result.data).toBe('default');
    });

    it('schema fails validation when env var is undefined (without default - required flag)', () => {
      const flag = stringFlag('TEST_FLAG');
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(false);
    });

    it('preserves empty string as valid value', () => {
      const flag = stringFlag('TEST_FLAG', 'default');
      const result = flag.schema.safeParse('');
      expect(result.success).toBe(true);
      expect(result.data).toBe('');
    });

    it('sets envVar and typeName correctly', () => {
      const flag = stringFlag('MY_ENV_VAR');
      expect(flag.envVar).toBe('MY_ENV_VAR');
      expect(flag.typeName).toBe('string');
    });
  });

  describe('numberFlag', () => {
    it('parses string to number', () => {
      const flag = numberFlag('TEST_FLAG', 0);
      const result = flag.schema.safeParse('42');
      expect(result.success).toBe(true);
      expect(result.data).toBe(42);
    });

    it('parses negative numbers', () => {
      const flag = numberFlag('TEST_FLAG', 0);
      const result = flag.schema.safeParse('-10');
      expect(result.success).toBe(true);
      expect(result.data).toBe(-10);
    });

    it('parses decimal numbers', () => {
      const flag = numberFlag('TEST_FLAG', 0);
      const result = flag.schema.safeParse('3.14');
      expect(result.success).toBe(true);
      expect(result.data).toBe(3.14);
    });

    it('returns default value when env var is undefined (with default)', () => {
      const flag = numberFlag('TEST_FLAG', 99);
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(true);
      expect(result.data).toBe(99);
    });

    it('schema fails validation when env var is undefined (without default - required flag)', () => {
      const flag = numberFlag('TEST_FLAG');
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(false);
    });

    it('schema fails validation for invalid number strings', () => {
      const flag = numberFlag('TEST_FLAG');
      const result = flag.schema.safeParse('abc');
      expect(result.success).toBe(false);
    });

    it('coerces empty string to 0', () => {
      const flag = numberFlag('TEST_FLAG');
      // z.coerce.number() coerces empty string to 0
      const result = flag.schema.safeParse('');
      expect(result.success).toBe(true);
      expect(result.data).toBe(0);
    });

    it('sets envVar and typeName correctly', () => {
      const flag = numberFlag('MY_ENV_VAR');
      expect(flag.envVar).toBe('MY_ENV_VAR');
      expect(flag.typeName).toBe('number');
    });
  });

  describe('listFlag', () => {
    it('splits a comma-separated env var into a trimmed array', () => {
      const flag = listFlag('TEST_FLAG');
      const result = flag.schema.safeParse('a, b ,c');
      expect(result.success).toBe(true);
      expect(result.data).toEqual(['a', 'b', 'c']);
    });

    it('drops empty entries', () => {
      const flag = listFlag('TEST_FLAG');
      const result = flag.schema.safeParse('a,,b,');
      expect(result.success).toBe(true);
      expect(result.data).toEqual(['a', 'b']);
    });

    it('returns default value when env var is undefined', () => {
      const flag = listFlag('TEST_FLAG', ['default']);
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(true);
      expect(result.data).toEqual(['default']);
    });

    it('defaults to an empty array when no default is given', () => {
      const flag = listFlag('TEST_FLAG');
      const result = flag.schema.safeParse(undefined);
      expect(result.success).toBe(true);
      expect(result.data).toEqual([]);
    });

    it('sets envVar and typeName correctly', () => {
      const flag = listFlag('MY_ENV_VAR');
      expect(flag.envVar).toBe('MY_ENV_VAR');
      expect(flag.typeName).toBe('list');
    });
  });

  describe('parseFlags', () => {
    it('parses all flags successfully when all env vars are set', () => {
      const testDefs = {
        myString: stringFlag('TEST_STRING', 'default'),
        myBool: booleanFlag('TEST_BOOL'),
        myNumber: numberFlag('TEST_NUMBER', 0),
      };

      const env: EnvConfig = {
        TEST_STRING: 'hello',
        TEST_BOOL: 'true',
        TEST_NUMBER: '42',
      };

      const result = parseFlags<{
        myString: string;
        myBool: boolean;
        myNumber: number;
      }>(testDefs, env);

      expect(result.myString).toBe('hello');
      expect(result.myBool).toBe(true);
      expect(result.myNumber).toBe(42);
    });

    it('uses defaults for optional flags when env vars are missing', () => {
      const testDefs = {
        optionalString: stringFlag('TEST_OPTIONAL_STRING', 'default-str'),
        optionalBool: booleanFlag('TEST_OPTIONAL_BOOL', true),
        optionalNumber: numberFlag('TEST_OPTIONAL_NUMBER', 100),
      };

      const env: EnvConfig = {};

      const result = parseFlags<{
        optionalString: string;
        optionalBool: boolean;
        optionalNumber: number;
      }>(testDefs, env);

      expect(result.optionalString).toBe('default-str');
      expect(result.optionalBool).toBe(true);
      expect(result.optionalNumber).toBe(100);
    });

    it('throws error listing missing required flag when validation fails', () => {
      const testDefs = {
        requiredString: stringFlag('TEST_REQUIRED'),
      };

      const env: EnvConfig = {};

      expect(() => parseFlags<{ requiredString: string }>(testDefs, env)).toThrowError(
        /Missing required feature flags/
      );

      expect(() => parseFlags<{ requiredString: string }>(testDefs, env)).toThrowError(
        /requiredString: TEST_REQUIRED \(string\)/
      );
    });

    it('throws combined error when multiple required flags are missing', () => {
      const testDefs = {
        requiredString: stringFlag('TEST_REQUIRED_STRING'),
        requiredNumber: numberFlag('TEST_REQUIRED_NUMBER'),
      };

      const env: EnvConfig = {};

      expect(() =>
        parseFlags<{ requiredString: string; requiredNumber: number }>(testDefs, env)
      ).toThrowError(/requiredString: TEST_REQUIRED_STRING \(string\)/);

      expect(() =>
        parseFlags<{ requiredString: string; requiredNumber: number }>(testDefs, env)
      ).toThrowError(/requiredNumber: TEST_REQUIRED_NUMBER \(number\)/);
    });

    it('error message includes flag key, env var name, and type', () => {
      const testDefs = {
        myCustomFlag: stringFlag('VITE_FF_MY_CUSTOM'),
      };

      const env: EnvConfig = {};

      try {
        parseFlags<{ myCustomFlag: string }>(testDefs, env);
        expect.fail('Expected an error to be thrown');
      } catch (error) {
        const message = (error as Error).message;
        expect(message).toContain('myCustomFlag');
        expect(message).toContain('VITE_FF_MY_CUSTOM');
        expect(message).toContain('string');
      }
    });

    it('mixes required and optional flags correctly', () => {
      const testDefs = {
        required: stringFlag('TEST_REQUIRED'),
        optional: stringFlag('TEST_OPTIONAL', 'default'),
      };

      const env: EnvConfig = {
        TEST_REQUIRED: 'provided',
      };

      const result = parseFlags<{
        required: string;
        optional: string;
      }>(testDefs, env);

      expect(result.required).toBe('provided');
      expect(result.optional).toBe('default');
    });
  });
});
