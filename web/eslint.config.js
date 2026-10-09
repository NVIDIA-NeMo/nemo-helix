// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import eslintConfigs from '@eslint/js';
import tanstackConfigs from '@tanstack/eslint-plugin-query';
import tsEslintPlugin from '@typescript-eslint/eslint-plugin';
import tsEslintParser from '@typescript-eslint/parser';
import importPlugin from 'eslint-plugin-import-x';
import hooksPlugin from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import eslintReact from '@eslint-react/eslint-plugin';
import unusedImports from 'eslint-plugin-unused-imports';
import globals from 'globals';
import stylistic from '@stylistic/eslint-plugin';
import noOnlyTestsPlugin from 'eslint-plugin-no-only-tests';
import testingLibrary from 'eslint-plugin-testing-library';
import vitest from '@vitest/eslint-plugin';
import tseslint from 'typescript-eslint';

const pathPrefix = '';

const baseLanguageOptions = {
  ecmaVersion: 2022,
  sourceType: 'module',
  parser: tsEslintParser,
  globals: {
    ...globals.browser,
    ...globals.node,
  },
};

const basePlugins = {
  '@typescript-eslint': tsEslintPlugin,
  'unused-imports': unusedImports,
  import: importPlugin,
  '@stylistic': stylistic,
  'no-only-tests': noOnlyTestsPlugin,
  'react-hooks': hooksPlugin,
};

const baseRules = {
  ...hooksPlugin.configs.recommended.rules,
  '@stylistic/padding-line-between-statements': [
    'error',
    { blankLine: 'always', prev: 'import', next: '*' },
    { blankLine: 'any', prev: 'import', next: 'import' },
  ],
  '@typescript-eslint/no-unused-vars': 'off',
  'import/extensions': [
    'error',
    'never',
    {
      // Enable mjs import extensions from packages (like openai)
      mjs: 'ignorePackages',
      png: 'always',
      svg: 'always',
      css: 'always',
      md: 'always',
    },
  ],
  'import/order': [
    'error',
    {
      'newlines-between': 'always', // or "never" or "always-and-inside-groups"
      groups: [['builtin', 'external'], 'internal', ['parent', 'sibling', 'index']],
      pathGroups: [
        {
          pattern: '@/**',
          group: 'internal',
        },
      ],
      alphabetize: {
        order: 'asc',
        caseInsensitive: true,
      },
    },
  ],
  'no-console': ['error', { allow: ['warn', 'error'] }],
  'no-duplicate-imports': 'error',
  'no-only-tests/no-only-tests': 'error',
  'unused-imports/no-unused-imports': 'error',
  'unused-imports/no-unused-vars': 'error',
  '@stylistic/jsx-curly-brace-presence': ['error', { props: 'never', children: 'never' }],
  'no-restricted-syntax': [
    'error',
    {
      selector:
        "JSXAttribute[value.type='JSXExpressionContainer'][value.expression.type='Literal'][value.expression.value=true]",
      message: "Omit the explicit '={true}'; use shorthand boolean JSX attribute instead.",
    },
    {
      selector: "JSXAttribute[name.name='css']",
      message: "Avoid the 'css' prop; use styled components or CSS modules.",
    },
    {
      selector: "JSXAttribute[name.name='style']",
      message: "Avoid inline 'style'; use styled components or CSS modules.",
    },
  ],
};

const ignores = [
  `${pathPrefix}packages/studio/dist`,
  `${pathPrefix}packages/studio/public/vendor`,
  `${pathPrefix}packages/studio/playwright-report`,
  `${pathPrefix}packages/studio/test-results`,
  `${pathPrefix}packages/studio/.test-reports`,
  `${pathPrefix}packages/sdk/generated/**`,
  `${pathPrefix}packages/common/plugin-types/**`,
  `${pathPrefix}packages/storybook/public/mockServiceWorker.js`,
  `${pathPrefix}demo-notebook/**`,
];

export default [
  // Top level ignores defines files that are never parsed across all configs. A config can define its own ignores.
  {
    ignores,
  },
  eslintConfigs.configs.recommended,
  {
    // New in eslint 10 recommended; pre-existing code violates them
    rules: {
      'no-useless-assignment': 'off',
      'preserve-caught-error': 'off',
    },
  },
  {
    ...reactRefresh.configs.vite,
    files: ['packages/studio/**/*.{jsx,tsx}'],
  },
  ...tseslint.configs.recommended,
  // Handles root directory
  {
    files: ['*'],
    languageOptions: baseLanguageOptions,
  },
  // Handles common
  {
    files: [`${pathPrefix}packages/common/**/*.ts`, `${pathPrefix}packages/common/**/*.tsx`],
    plugins: {
      ...basePlugins,
      '@eslint-react': eslintReact,
    },
    rules: {
      ...baseRules,
      'import/no-default-export': 'error',
    },
    languageOptions: baseLanguageOptions,
  },
  // Handles UI
  {
    files: [`${pathPrefix}packages/studio/**/*.ts`, `${pathPrefix}packages/studio/**/*.tsx`],
    plugins: {
      ...basePlugins,
      '@eslint-react': eslintReact,
      'react-hooks': hooksPlugin,
      '@tanstack/eslint-plugin-query': tanstackConfigs,
    },
    rules: {
      ...hooksPlugin.configs.recommended.rules,
      ...baseRules,
      'import/no-default-export': 'error',
      'no-console': 'error',
      'no-restricted-imports': [
        'warn',
        {
          patterns: [
            {
              group: ['../*', '../**'],
              message:
                'Use the absolute @studio/* alias instead of parent-directory relative paths.',
            },
          ],
        },
      ],
    },
    languageOptions: baseLanguageOptions,
  },
  // Enforce *.test.* filename convention across all test trees (src/, e2e-tests/, orval/, etc.)
  {
    files: [`${pathPrefix}**/*.{test,spec}.{js,jsx,ts,tsx}`],
    plugins: { vitest },
    rules: {
      'vitest/consistent-test-filename': ['error', { pattern: '.*\\.test\\.[jt]sx?$' }],
    },
  },
  // Vitest + Testing Library rules — scoped to src/ only (e2e tests use Playwright's test(), not it())
  {
    files: [`${pathPrefix}**/src/**/*.{test,spec}.{js,jsx,ts,tsx}`],
    plugins: {
      vitest,
      ...testingLibrary.configs['flat/react'].plugins,
    },
    rules: {
      ...testingLibrary.configs['flat/react'].rules,
      'vitest/consistent-test-it': ['error', { fn: 'it' }],
      'testing-library/no-debugging-utils': 'error',
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: 'vitest',
              message:
                "Do not import runtime values from 'vitest'. Test APIs are exposed as globals via `globals: true` in the shared vitest config (vi, describe, it, expect, beforeEach, afterEach). Type-only imports (e.g. `import type { Mock } from 'vitest'`) are allowed.",
              allowTypeImports: true,
            },
          ],
        },
      ],
    },
  },
  // Storybook story files: allow default export for CSF meta
  {
    files: [`${pathPrefix}**/*.stories.@(ts|tsx)`],
    rules: {
      ...baseRules,
      'import/no-default-export': 'off',
    },
  },
];
