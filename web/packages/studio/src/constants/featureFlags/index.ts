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

import { flagDefinitions, FeatureFlags } from '@studio/constants/featureFlags/featureFlags';
import {
  installFlagsGlobal,
  readFlagOverrides,
  writeFlagManifest,
} from '@studio/constants/featureFlags/overrides';
import { parseFlags } from '@studio/constants/featureFlags/utils';

// --- Parse cascade ---

// Pass 1: the build environment alone. This is also the only source consulted
// for `flagOverridesEnabled` — if the gate were readable from the localStorage
// override blob, anyone could set it to `true` in production and unlock every
// other flag, which would defeat the point of making it server-controlled.
const baseFlags = parseFlags<FeatureFlags>(flagDefinitions, import.meta.env);

const overridesEnabled = baseFlags.flagOverridesEnabled;

// --- Exports ---

/**
 * Parsed feature flags. Access flags as typed properties.
 *
 * Values come from the build environment, optionally overlaid with local
 * overrides from localStorage when the deployment enables them. Overrides are
 * read once here, at module load, so changing them requires a page reload.
 *
 * @example
 * import { featureFlags } from '@studio/constants/featureFlags';
 *
 * if (featureFlags.experimentalChat) {
 *   // render experimental chat UI
 * }
 */
export const featureFlags: FeatureFlags = overridesEnabled
  ? parseFlags<FeatureFlags>(flagDefinitions, {
      ...import.meta.env,
      ...readFlagOverrides(flagDefinitions),
    })
  : baseFlags;

/**
 * Publish the override tooling: `window.__flags` for devtools, and the
 * localStorage manifest the companion Chrome extension reads. Called from
 * `main.tsx` rather than at import time so tests and SSR-ish consumers can
 * import flags without the side effects.
 */
export const initFlagOverrideTooling = (): void => {
  writeFlagManifest(flagDefinitions, baseFlags, featureFlags, overridesEnabled);
  installFlagsGlobal(flagDefinitions, baseFlags, featureFlags, overridesEnabled);
};
