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
  FLAG_MANIFEST_VERSION,
  FlagManifest,
  FlagsGlobal,
  installFlagsGlobal,
  readFlagOverrides,
  writeFlagManifest,
} from '@studio/constants/featureFlags/overrides';
import {
  booleanFlag,
  numberFlag,
  parseFlags,
  previewFlag,
  stringFlag,
} from '@studio/constants/featureFlags/utils';
import { FEATURE_FLAG_MANIFEST_KEY, FEATURE_FLAG_OVERRIDES_KEY } from '@studio/util/localStorage';

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

const flagsGlobal = () => (window as unknown as { __flags?: FlagsGlobal }).__flags;

const readManifest = (): FlagManifest =>
  JSON.parse(window.localStorage.getItem(FEATURE_FLAG_MANIFEST_KEY)!) as FlagManifest;

beforeEach(() => {
  window.localStorage.clear();
  delete (window as unknown as Record<string, unknown>).__flags;
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

describe('writeFlagManifest', () => {
  it('describes every flag with its base and effective value', () => {
    setBlob({ alphaEnabled: 'preview' });

    writeFlagManifest(
      definitions,
      { alphaEnabled: false, betaEnabled: true, gammaCount: 3, deltaName: 'base' },
      { alphaEnabled: 'preview', betaEnabled: true, gammaCount: 3, deltaName: 'base' },
      true
    );

    const manifest = readManifest();
    expect(manifest.version).toBe(FLAG_MANIFEST_VERSION);
    expect(manifest.overridesEnabled).toBe(true);
    expect(manifest.overridesKey).toBe(FEATURE_FLAG_OVERRIDES_KEY);

    const alpha = manifest.flags.find((f) => f.key === 'alphaEnabled');
    expect(alpha).toEqual({
      key: 'alphaEnabled',
      envVar: 'VITE_FF_ALPHA_ENABLED',
      type: 'preview',
      baseValue: false,
      effectiveValue: 'preview',
      overridden: true,
    });
    expect(manifest.flags.find((f) => f.key === 'betaEnabled')?.overridden).toBe(false);
  });

  // The extension needs to tell "disabled here" apart from "not a Studio page".
  it('writes a stub manifest when overrides are disabled', () => {
    writeFlagManifest(definitions, {}, {}, false);

    const manifest = readManifest();
    expect(manifest).toEqual({
      version: FLAG_MANIFEST_VERSION,
      overridesEnabled: false,
      overridesKey: '',
      flags: [],
    });
  });
});

describe('installFlagsGlobal', () => {
  const install = (enabled = true) =>
    installFlagsGlobal(
      definitions,
      { alphaEnabled: false, betaEnabled: true, gammaCount: 3, deltaName: 'base' },
      { alphaEnabled: false, betaEnabled: true, gammaCount: 3, deltaName: 'base' },
      enabled
    );

  it('does not attach when overrides are disabled', () => {
    install(false);
    expect(flagsGlobal()).toBeUndefined();
  });

  it('writes a valid override and refreshes the manifest', () => {
    install();
    flagsGlobal()!.set('alphaEnabled', 'preview');

    expect(JSON.parse(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)!)).toEqual({
      alphaEnabled: 'preview',
    });
    expect(readManifest().flags.find((f) => f.key === 'alphaEnabled')?.overridden).toBe(true);
  });

  it('rejects an unknown flag without writing', () => {
    install();
    flagsGlobal()!.set('nopeEnabled', 'true');

    expect(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)).toBeNull();
    expect(logger.error).toHaveBeenCalled();
  });

  it('rejects a value that fails the flag schema without writing', () => {
    install();
    flagsGlobal()!.set('gammaCount', 'banana');

    expect(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)).toBeNull();
  });

  it('merges successive set calls', () => {
    install();
    flagsGlobal()!.set('alphaEnabled', 'preview');
    flagsGlobal()!.set('betaEnabled', 'false');

    expect(JSON.parse(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)!)).toEqual({
      alphaEnabled: 'preview',
      betaEnabled: 'false',
    });
  });

  it('unset removes a single flag', () => {
    install();
    flagsGlobal()!.set('alphaEnabled', 'preview');
    flagsGlobal()!.set('betaEnabled', 'false');
    flagsGlobal()!.unset('alphaEnabled');

    expect(JSON.parse(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)!)).toEqual({
      betaEnabled: 'false',
    });
  });

  // An empty blob is removed rather than left as "{}" so a cleared state is
  // indistinguishable from never having set one.
  it('reset clears the stored blob entirely', () => {
    install();
    flagsGlobal()!.set('alphaEnabled', 'preview');
    flagsGlobal()!.reset();

    expect(window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY)).toBeNull();
  });

  it('list returns one entry per defined flag', () => {
    install();
    expect(
      flagsGlobal()!
        .list()
        .map((f) => f.key)
    ).toEqual(Object.keys(definitions));
  });
});
