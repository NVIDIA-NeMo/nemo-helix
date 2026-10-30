// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type CreateFilesetRequest, FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import type { NemotronPersonasLocale } from '@studio/components/agents/PersonaTestSet/types';

/** Locales with a published Nemotron Personas dataset. Mirrors `NEMOTRON_PERSONAS_DATASET_SIZES`. */
export const NEMOTRON_PERSONAS_LOCALES = [
  'en_US',
  'en_GB',
  'en_IN',
  'en_SG',
  'de_DE',
  'fr_FR',
  'hi_Deva_IN',
  'hi_Latn_IN',
  'ja_JP',
  'ko_KR',
  'pt_BR',
] as const;

export const isNemotronPersonasLocale = (locale: string): locale is NemotronPersonasLocale =>
  (NEMOTRON_PERSONAS_LOCALES as readonly string[]).includes(locale);

/** The person sampler reads each locale from this workspace; see `data_designer_nemo.nemotron_personas`. */
export const NEMOTRON_PERSONAS_WORKSPACE = 'system';

export const nemotronPersonasFilesetName = (locale: NemotronPersonasLocale): string =>
  `nemotron-personas-dataset-${locale.toLowerCase()}`;

/**
 * The same NGC-backed fileset `nemo data-designer personas make-fileset` creates, so either path
 * leaves the platform in one state.
 */
export const nemotronPersonasFilesetRequest = (
  locale: NemotronPersonasLocale,
  apiKeySecret: string
): CreateFilesetRequest => ({
  name: nemotronPersonasFilesetName(locale),
  description: `Nemotron Personas dataset for locale: '${locale}'`,
  purpose: FilesetPurpose.dataset,
  storage: {
    type: 'ngc',
    api_key_secret: apiKeySecret,
    org: 'nvidia',
    team: 'nemotron-personas',
    target: nemotronPersonasFilesetName(locale),
    target_type: 'resource',
  },
  cache: true,
});
