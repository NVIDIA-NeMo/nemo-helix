// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Shared modules kept off the plugin surface on purpose, with the reason. Modules that call the
// SDK or axios at runtime are excluded automatically and don't belong here.
const NOT_FOR_PLUGINS: Record<string, string> = {};

// Exported before this check existed. AssistantChat calls the gateway through useChatCompletion;
// the confirmation modals only use axios for an `instanceof AxiosError` check in getErrorMessage.
const SDK_BOUND_EXPORTS = [
  'components/AssistantChat',
  'components/ConfirmationModal',
  'components/DeleteConfirmationModal',
];

// Modules that predate this check and nobody has decided on yet. Only ever remove entries.
const UNREVIEWED: string[] = [
  'components/AccordionPanel',
  'components/Chat/ChatEmptyState',
  'components/Chat/MessageContent',
  'components/CodeDisplay',
  'components/CodeEditor',
  'components/ComparisonLineChart',
  'components/DataView/ColumnFilterPanel',
  'components/DataView/FilterPanel',
  'components/DataView/FilterPanelToggle',
  'components/DataView/StudioAppliedFilters',
  'components/DataView/StudioDataViewToolbar',
  'components/DataView/TableExpandableCell',
  'components/DataView/dateTimeFilter',
  'components/DataView/useRowClick',
  'components/DatasetFileSelect/DatasetFileSelectButton',
  'components/DatasetFileSelect/parseFilesetLocation',
  'components/DatasetFileSelect/utils',
  'components/DatasetRowPager',
  'components/DeltaText',
  'components/DownloadMenu',
  'components/ElapsedTime',
  'components/EntityCard',
  'components/EntityEmptyState',
  'components/ErrorMessage',
  'components/ExternalLink',
  'components/FileContentPreview',
  'components/FileList',
  'components/FilterFields/DateRangeFilterField',
  'components/GradientBackground',
  'components/IntakeAccordion',
  'components/KVPair',
  'components/KeyValueGrid',
  'components/LeftOverflow',
  'components/MarkdownContent',
  'components/ModelDetailsTooltip/utils',
  'components/Nebula',
  'components/OverflowGroup',
  'components/PreviewBox',
  'components/ScoreGauge',
  'components/ScrollTable',
  'components/SliderWithTextInput',
  'components/StatTile',
  'components/StatusDot',
  'components/ToolCard',
  'components/TrainingParameterSlider/types',
  'components/buttons/VariableButton',
  'components/charts/ChartEmptyFrame',
  'components/charts/ChartHeader',
  'components/charts/ChartLegend',
  'components/charts/ChartSkeleton',
  'components/charts/ChartSwatch',
  'components/charts/ChartTooltip',
  'components/charts/format',
  'components/charts/frame',
  'components/charts/referenceLines',
  'components/charts/tokens',
  'components/charts/types',
  'components/form/ControlledCheckbox',
  'components/form/ControlledCombobox',
  'components/form/ControlledRadioGroup',
  'components/form/ControlledSearchableSelect',
  'components/form/ControlledSegmentedControl',
  'components/form/ControlledSliderWithTextInput',
  'components/form/ControlledSwitch',
  'components/form/MappingFields',
  'components/form/MaskedTextInput',
  'components/form/TextInputSpinner',
  'components/form/VariableTextArea',
  'hooks/useCopyToClipboard',
  'hooks/useDeferredUnmount',
  'hooks/useLiveSeconds',
  'hooks/useQueryParams',
  'hooks/useRehydrateListFromDetailQuery',
  'hooks/useRowNavigation',
  'hooks/useStickToBottom',
];

const SRC_DIR = path.dirname(fileURLToPath(import.meta.url));
const SHARED_ROOTS = ['components', 'hooks', 'api'];

// common.js is bundled apart from Studio, so these would bring a second, unauthenticated axios
// client into every plugin; plugins reach the API through `host.sdk` instead.
const SDK_CLIENT =
  /^(axios|@nemo\/sdk\/src\/.+|@nemo\/sdk\/generated\/[^/]+\/(?!(schema|zod)(\/|$)).+)$/;

const isSourceModule = (name: string) =>
  /\.tsx?$/.test(name) && !/\.(test|stories|d)\.tsx?$/.test(name);

const hasIndex = (relativeDir: string) =>
  ['index.ts', 'index.tsx'].some((name) => fs.existsSync(path.join(SRC_DIR, relativeDir, name)));

const modulesUnder = (relativeDir: string): string[] =>
  fs
    .readdirSync(path.join(SRC_DIR, relativeDir), { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith('.'))
    .flatMap((entry) => {
      if (entry.isDirectory()) {
        const dir = `${relativeDir}/${entry.name}`;
        return hasIndex(dir) ? [dir] : modulesUnder(dir);
      }
      if (!isSourceModule(entry.name)) {
        return [];
      }
      return [`${relativeDir}/${entry.name.replace(/\.tsx?$/, '')}`];
    });

const resolveSourceFile = (base: string): string | null => {
  const candidates = [`${base}.ts`, `${base}.tsx`, `${base}/index.ts`, `${base}/index.tsx`];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
};

const resolveImport = (fromFile: string, specifier: string): string | null => {
  if (specifier.startsWith('@nemo/common/src/')) {
    return resolveSourceFile(path.join(SRC_DIR, specifier.slice('@nemo/common/src/'.length)));
  }
  if (specifier.startsWith('.')) {
    return resolveSourceFile(path.resolve(path.dirname(fromFile), specifier));
  }
  return null;
};

const MODULE_STATEMENT =
  /^(?:import|export)\s+(type\s+)?([^;]*?)\s*from\s+'([^']+)'|^import\s+'([^']+)'/gm;

const isTypeOnlyClause = (clause: string) => {
  const named = /^\{([\s\S]*)\}$/.exec(clause.trim());
  if (!named) {
    return false;
  }
  const elements = named[1]
    .split(',')
    .map((element) => element.trim())
    .filter(Boolean);
  return elements.length > 0 && elements.every((element) => element.startsWith('type '));
};

const runtimeImports = (file: string): string[] =>
  [...fs.readFileSync(file, 'utf-8').matchAll(MODULE_STATEMENT)].flatMap(
    ([, typeKeyword, clause, specifier, sideEffectSpecifier]) => {
      if (sideEffectSpecifier) {
        return [sideEffectSpecifier];
      }
      if (typeKeyword || isTypeOnlyClause(clause)) {
        return [];
      }
      return [specifier];
    }
  );

const sdkReachCache = new Map<string, string | null>();

// The first SDK-client import reachable from `file` at runtime, as an import chain.
const sdkReach = (file: string): string | null => {
  if (sdkReachCache.has(file)) {
    return sdkReachCache.get(file) ?? null;
  }
  sdkReachCache.set(file, null);
  let reach: string | null = null;
  for (const specifier of runtimeImports(file)) {
    if (SDK_CLIENT.test(specifier)) {
      reach = specifier;
      break;
    }
    const resolved = resolveImport(file, specifier);
    const nested = resolved ? sdkReach(resolved) : null;
    if (nested) {
      reach = `${specifier} -> ${nested}`;
      break;
    }
  }
  sdkReachCache.set(file, reach);
  return reach;
};

const unitFile = (unit: string) => resolveSourceFile(path.join(SRC_DIR, unit));

const pluginSpecifiers = (): string[] =>
  [
    ...fs
      .readFileSync(path.join(SRC_DIR, 'plugin.ts'), 'utf-8')
      .matchAll(/from '@nemo\/common\/src\/([^']+)'/g),
  ].map((match) => match[1]);

const isWithin = (unit: string, prefix: string) => unit === prefix || unit.startsWith(`${prefix}/`);

describe('plugin API surface', () => {
  const units = SHARED_ROOTS.flatMap(modulesUnder).sort();
  const specifiers = pluginSpecifiers();
  const isExposed = (unit: string) => specifiers.some((specifier) => isWithin(specifier, unit));
  const isSdkBound = (unit: string) => {
    const file = unitFile(unit);
    return !!file && !!sdkReach(file);
  };

  it('exports nothing that reaches the SDK client or axios at runtime', () => {
    const sdkBoundExports = specifiers.filter((specifier) => {
      const file = unitFile(specifier);
      return !!file && !!sdkReach(file);
    });
    expect(sdkBoundExports).toEqual(SDK_BOUND_EXPORTS);
  });

  it('decides every shared module: export it from plugin.ts or list it in NOT_FOR_PLUGINS', () => {
    const undecided = units.filter(
      (unit) =>
        !isExposed(unit) &&
        !isSdkBound(unit) &&
        !UNREVIEWED.includes(unit) &&
        !Object.keys(NOT_FOR_PLUGINS).some((prefix) => isWithin(unit, prefix))
    );
    expect(undecided).toEqual([]);
  });

  it('drops NOT_FOR_PLUGINS entries that are exported, SDK-bound or deleted', () => {
    const stale = Object.keys(NOT_FOR_PLUGINS).filter((prefix) => {
      const covered = units.filter((unit) => isWithin(unit, prefix));
      return covered.length === 0 || covered.every((unit) => isExposed(unit) || isSdkBound(unit));
    });
    expect(stale).toEqual([]);
  });

  it('drops UNREVIEWED entries once they are exported, SDK-bound or deleted', () => {
    const stale = UNREVIEWED.filter(
      (unit) => !units.includes(unit) || isExposed(unit) || isSdkBound(unit)
    );
    expect(stale).toEqual([]);
  });
});
