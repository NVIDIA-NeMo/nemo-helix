// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Sentinel option for clearing a field so the browser's setting applies. */
export const BROWSER_DEFAULT = '__browser__';

const COMMON_LOCALES = [
  'en-US',
  'en-GB',
  'de-DE',
  'es-ES',
  'fr-FR',
  'it-IT',
  'ja-JP',
  'ko-KR',
  'pt-BR',
  'zh-CN',
  'zh-TW',
];

const languageNames = new Intl.DisplayNames(undefined, { type: 'language' });

const option = (value: string, label: string) => ({ value, children: label, filterValue: label });

export const timezoneOptions = (current?: string | null) => {
  const zones = new Set(['UTC', ...Intl.supportedValuesOf('timeZone')]);
  if (current) zones.add(current);
  return [
    option(BROWSER_DEFAULT, 'Browser default'),
    ...[...zones].sort().map((zone) => option(zone, zone.replaceAll('_', ' '))),
  ];
};

export const localeOptions = (current?: string | null) => {
  const locales = new Set([...navigator.languages, ...COMMON_LOCALES]);
  if (current) locales.add(current);
  return [
    option(BROWSER_DEFAULT, 'Browser default'),
    ...[...locales]
      .map((tag) => option(tag, `${languageNames.of(tag) ?? tag} (${tag})`))
      .sort((a, b) => a.children.localeCompare(b.children)),
  ];
};
