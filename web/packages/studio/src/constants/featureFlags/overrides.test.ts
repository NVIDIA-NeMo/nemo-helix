/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { logger } from '@nemo/common/src/utils/logger';
import {
  readFlagOverrides,
  readStoredOverrides,
  writeStoredOverrides,
} from '@studio/constants/featureFlags/overrides';
import {
  booleanFlag,
  numberFlag,
  parseFlags,
  previewFlag,
  stringFlag,
} from '@studio/constants/featureFlags/utils';
import { FEATURE_FLAG_OVERRIDES_KEY } from '@studio/util/localStorage';

const definitions = {
  alphaEnabled: previewFlag('VITE_FF_ALPHA_ENABLED', false),
  betaEnabled: booleanFlag('VITE_FF_BETA_ENABLED', true),
  gammaCount: numberFlag('VITE_FF_GAMMA_COUNT', 3),
  deltaName: stringFlag('VITE_FF_DELTA_NAME'),
};

const setBlob = (value: unknown) =>
  window.localStorage.setItem(
    FEATURE_FLAG_OVERRIDES_KEY,
    typeof value === 'string' ? value : JSON.stringify(value)
  );

/**
 * Simulate a browser where storage is blocked. Reading `window.localStorage` is
 * itself the throwing operation, which is why the module cannot `typeof`-guard
 * it.
 */
const withBlockedStorage = (fn: () => void) => {
  const original = Object.getOwnPropertyDescriptor(window, 'localStorage');
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    get() {
      throw new DOMException('Access denied', 'SecurityError');
    },
  });
  try {
    fn();
  } finally {
    if (original) Object.defineProperty(window, 'localStorage', original);
  }
};

/** Storage that reads normally but rejects every write, as when quota is full. */
const withFailingWrites = (fn: () => void) => {
  const original = Object.getOwnPropertyDescriptor(window, 'localStorage');
  const backing = new Map<string, string>();
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      getItem: (k: string) => backing.get(k) ?? null,
      setItem: () => {
        throw new DOMException('Quota exceeded', 'QuotaExceededError');
      },
      removeItem: () => {
        throw new DOMException('Quota exceeded', 'QuotaExceededError');
      },
      clear: () => backing.clear(),
    },
  });
  try {
    fn();
  } finally {
    if (original) Object.defineProperty(window, 'localStorage', original);
  }
};

beforeEach(() => {
  window.localStorage.clear();
  vi.spyOn(logger, 'warn').mockImplementation(() => {});
  vi.spyOn(logger, 'error').mockImplementation(() => {});
  vi.spyOn(logger, 'info').mockImplementation(() => {});
});

describe('readFlagOverrides', () => {
  it('returns an empty layer when nothing is stored', () => {
    expect(readFlagOverrides(definitions)).toEqual({});
  });

  it('maps camelCase flag keys onto their env var names', () => {
    setBlob({ alphaEnabled: 'preview', gammaCount: '7' });

    expect(readFlagOverrides(definitions)).toEqual({
      VITE_FF_ALPHA_ENABLED: 'preview',
      VITE_FF_GAMMA_COUNT: '7',
    });
  });

  it('discards the whole blob when the JSON is malformed', () => {
    setBlob('{not json');

    expect(readFlagOverrides(definitions)).toEqual({});
    expect(logger.warn).toHaveBeenCalled();
  });

  it('discards the whole blob when it is not an object', () => {
    setBlob(['alphaEnabled']);

    expect(readFlagOverrides(definitions)).toEqual({});
  });

  it('drops unknown flag keys but keeps valid siblings', () => {
    setBlob({ nopeEnabled: 'true', alphaEnabled: 'true' });

    expect(readFlagOverrides(definitions)).toEqual({ VITE_FF_ALPHA_ENABLED: 'true' });
  });

  it('drops non-string values', () => {
    setBlob({ betaEnabled: true });

    expect(readFlagOverrides(definitions)).toEqual({});
  });

  // parseFlags treats a schema failure as a fatal "missing required flag", so a
  // bad override must never reach it.
  it('drops values that fail their flag schema', () => {
    setBlob({ gammaCount: 'not-a-number', alphaEnabled: 'preview' });

    expect(readFlagOverrides(definitions)).toEqual({ VITE_FF_ALPHA_ENABLED: 'preview' });
  });

  // A bare `definitions[key]` lookup reaches Object.prototype, so these keys
  // return a truthy non-descriptor and `.schema.safeParse` throws a TypeError
  // out of the warn-and-skip loop, aborting flag parsing entirely.
  it.each(['constructor', 'toString', 'valueOf', 'hasOwnProperty'])(
    'drops the inherited key "%s" instead of throwing',
    (key) => {
      setBlob({ [key]: 'true', alphaEnabled: 'preview' });

      expect(() => readFlagOverrides(definitions)).not.toThrow();
      expect(readFlagOverrides(definitions)).toEqual({ VITE_FF_ALPHA_ENABLED: 'preview' });
    }
  );

  it('returns an empty layer when localStorage access throws', () => {
    withBlockedStorage(() => {
      expect(() => readFlagOverrides(definitions)).not.toThrow();
      expect(readFlagOverrides(definitions)).toEqual({});
    });
  });

  it('produces a layer that parseFlags consumes without throwing', () => {
    setBlob({ alphaEnabled: 'preview', betaEnabled: 'false' });

    const parsed = parseFlags<{
      alphaEnabled: unknown;
      betaEnabled: unknown;
      gammaCount: unknown;
      deltaName: unknown;
    }>(definitions, {
      VITE_FF_DELTA_NAME: 'base',
      ...readFlagOverrides(definitions),
    });

    expect(parsed.alphaEnabled).toBe('preview');
    expect(parsed.betaEnabled).toBe(false);
    expect(parsed.gammaCount).toBe(3);
  });
});

describe('readStoredOverrides', () => {
  it('returns an empty blob when nothing is stored', () => {
    expect(readStoredOverrides(definitions)).toEqual({});
  });

  // The panel shows what the app would apply on reload, not whatever happens to
  // be sitting in storage, so the same filtering applies as the parse layer.
  it('returns valid entries keyed by flag name', () => {
    setBlob({ alphaEnabled: 'preview', gammaCount: '7' });

    expect(readStoredOverrides(definitions)).toEqual({
      alphaEnabled: 'preview',
      gammaCount: '7',
    });
  });

  it('omits stale entries for flags that no longer exist', () => {
    setBlob({ removedFlag: 'true', alphaEnabled: 'true' });

    expect(readStoredOverrides(definitions)).toEqual({ alphaEnabled: 'true' });
  });

  it('omits values that no longer satisfy their flag schema', () => {
    setBlob({ gammaCount: 'not-a-number', alphaEnabled: 'true' });

    expect(readStoredOverrides(definitions)).toEqual({ alphaEnabled: 'true' });
  });

  it('returns an empty blob when localStorage access throws', () => {
    withBlockedStorage(() => {
      expect(() => readStoredOverrides(definitions)).not.toThrow();
      expect(readStoredOverrides(definitions)).toEqual({});
    });
  });
});

describe('writeStoredOverrides', () => {
  it('persists the blob and reports success', () => {
    expect(writeStoredOverrides({ alphaEnabled: 'preview' })).toBe(true);
    expect(JSON.parse(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)!)).toEqual({
      alphaEnabled: 'preview',
    });
  });

  // A cleared state should be indistinguishable from one that was never set.
  it('removes the key entirely for an empty blob', () => {
    window.localStorage.setItem(FEATURE_FLAG_OVERRIDES_KEY, '{"alphaEnabled":"true"}');

    expect(writeStoredOverrides({})).toBe(true);
    expect(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)).toBeNull();
  });

  it('round-trips through readStoredOverrides', () => {
    writeStoredOverrides({ alphaEnabled: 'preview', betaEnabled: 'false' });

    expect(readStoredOverrides(definitions)).toEqual({
      alphaEnabled: 'preview',
      betaEnabled: 'false',
    });
  });

  it('reports failure instead of throwing when the write is rejected', () => {
    withFailingWrites(() => {
      expect(() => writeStoredOverrides({ alphaEnabled: 'preview' })).not.toThrow();
      expect(writeStoredOverrides({ alphaEnabled: 'preview' })).toBe(false);
      expect(logger.error).toHaveBeenCalled();
    });
  });

  it('reports failure when localStorage access is blocked', () => {
    withBlockedStorage(() => {
      expect(writeStoredOverrides({ alphaEnabled: 'preview' })).toBe(false);
    });
  });
});
