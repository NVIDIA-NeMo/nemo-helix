// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CodeSnippetLanguage } from '@nvidia/foundations-react-core';

/**
 * Maps all supported code snippet languages to their string identifiers.
 * The keys are valid `CodeSnippetLanguage` values and the corresponding values
 * represent the language string which may be used by syntax highlighters.
 *
 * @type {Record<CodeSnippetLanguage, string>}
 *
 * @example
 * LANGUAGES['typescript'] // "typescript"
 * LANGUAGES['python'] // "python"
 */
const LANGUAGES: Record<CodeSnippetLanguage, string> = {
  typescript: 'typescript',
  javascript: 'javascript',
  tsx: 'tsx',
  jsx: 'jsx',
  json: 'json',
  css: 'css',
  html: 'html',
  bash: 'bash',
  shell: 'shell',
  python: 'python',
  rust: 'rust',
  go: 'go',
  yaml: 'yaml',
  markdown: 'markdown',
  text: 'text',
};

/**
 * A Set of all supported code snippet languages.
 *
 * This set is constructed from the keys of the LANGUAGES mapping,
 * ensuring that only valid CodeSnippetLanguage values are included.
 * It is used for fast membership checking to validate language strings.
 *
 * @type {Set<CodeSnippetLanguage>}
 *
 * @example
 * SUPPORTED_LANGUAGES.has('typescript'); // true
 * SUPPORTED_LANGUAGES.has('cpp'); // false
 */
const SUPPORTED_LANGUAGES = new Set<CodeSnippetLanguage>(
  Object.keys(LANGUAGES) as CodeSnippetLanguage[]
);

/**
 * A mapping of common language/file extension aliases to supported {@link CodeSnippetLanguage} values.
 *
 * This object enables interpreting alternative or abbreviated language identifiers (such as file
 * extensions or short names) and mapping them to their canonical code snippet language name.
 *
 * Example mappings:
 * - 'ts'   → 'typescript'
 * - 'js'   → 'javascript'
 * - 'py'   → 'python'
 * - 'sh'   → 'shell'
 * - 'yml'  → 'yaml'
 * - 'md'   → 'markdown'
 *
 * @type {Record<string, CodeSnippetLanguage>}
 */
const aliasMap: Record<string, CodeSnippetLanguage> = {
  ts: 'typescript',
  js: 'javascript',
  py: 'python',
  rs: 'rust',
  sh: 'shell',
  yml: 'yaml',
  md: 'markdown',
};

/**
 * Checks if a given string value is a supported code snippet language.
 *
 * This function validates whether the provided {@link value} is one of the languages
 * recognized by the code snippet system, as defined in {@link SUPPORTED_LANGUAGES}.
 *
 * @param {string} value - The language string to check.
 * @returns {value is CodeSnippetLanguage}
 *   True if {@link value} is a supported code snippet language; false otherwise.
 *
 * @example
 * isCodeSnippetLanguage('typescript'); // true
 * isCodeSnippetLanguage('cpp'); // false
 */
export const isCodeSnippetLanguage = (value: string): value is CodeSnippetLanguage => {
  return SUPPORTED_LANGUAGES.has(value as CodeSnippetLanguage);
};

/**
 * Infers the programming language of a code snippet based on the file extension
 * extracted from the provided file path.
 *
 * The function checks if the file extension maps to a supported code snippet language
 * or a common alias (e.g., 'js' for 'javascript', 'ts' for 'typescript').
 *
 * @param {string} filePath - The file path string from which to extract the extension.
 * @returns {CodeSnippetLanguage | undefined} - The detected language as a CodeSnippetLanguage, or undefined if not recognized/supported.
 *
 * @example
 * getLanguageFromFilePath('index.tsx'); // returns 'typescript'
 * getLanguageFromFilePath('example.py'); // returns 'python'
 * getLanguageFromFilePath('README.md'); // returns 'markdown'
 * getLanguageFromFilePath('unknown.foo'); // returns undefined
 */
export const getLanguageFromFilePath = (filePath: string): CodeSnippetLanguage | undefined => {
  if (!filePath || typeof filePath !== 'string') return undefined;
  const fileType = filePath.trim().split('.').pop()?.toLowerCase();
  if (!fileType) return undefined;
  if (isCodeSnippetLanguage(fileType)) {
    return fileType;
  }

  return aliasMap[fileType] || undefined;
};

/**
 * Attempts to infer the programming language of a code snippet based on the first word,
 * which is commonly used as a language label (e.g., "typescript ...", "python ...").
 *
 * @param {string} code - The code string, potentially prefixed with a language name.
 * @returns {CodeSnippetLanguage | undefined}
 *   The detected language as a CodeSnippetLanguage, or undefined if not recognized/supported.
 */
export const languageInCode = (code: string): CodeSnippetLanguage | undefined => {
  const language = code.trim().split(/\s+/)[0].toLowerCase();
  if (!language) return undefined;
  if (isCodeSnippetLanguage(language)) {
    return language;
  }
  return aliasMap[language] || undefined;
};
