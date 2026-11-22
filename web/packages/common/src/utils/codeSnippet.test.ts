// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getLanguageFromFilePath,
  isCodeSnippetLanguage,
  languageInCode,
} from '@nemo/common/src/utils/codeSnippet';

describe('isCodeSnippetLanguage', () => {
  it('returns true for supported languages', () => {
    expect(isCodeSnippetLanguage('typescript')).toBe(true);
    expect(isCodeSnippetLanguage('javascript')).toBe(true);
    expect(isCodeSnippetLanguage('python')).toBe(true);
    expect(isCodeSnippetLanguage('rust')).toBe(true);
    expect(isCodeSnippetLanguage('go')).toBe(true);
    expect(isCodeSnippetLanguage('json')).toBe(true);
    expect(isCodeSnippetLanguage('yaml')).toBe(true);
    expect(isCodeSnippetLanguage('markdown')).toBe(true);
    expect(isCodeSnippetLanguage('bash')).toBe(true);
    expect(isCodeSnippetLanguage('shell')).toBe(true);
    expect(isCodeSnippetLanguage('css')).toBe(true);
    expect(isCodeSnippetLanguage('html')).toBe(true);
    expect(isCodeSnippetLanguage('tsx')).toBe(true);
    expect(isCodeSnippetLanguage('jsx')).toBe(true);
  });

  it('returns false for unsupported languages', () => {
    expect(isCodeSnippetLanguage('cpp')).toBe(false);
    expect(isCodeSnippetLanguage('java')).toBe(false);
    expect(isCodeSnippetLanguage('ruby')).toBe(false);
    expect(isCodeSnippetLanguage('unknown')).toBe(false);
    expect(isCodeSnippetLanguage('')).toBe(false);
  });

  it('returns false for aliases (not canonical names)', () => {
    expect(isCodeSnippetLanguage('ts')).toBe(false);
    expect(isCodeSnippetLanguage('js')).toBe(false);
    expect(isCodeSnippetLanguage('py')).toBe(false);
    expect(isCodeSnippetLanguage('sh')).toBe(false);
  });

  it('is case-sensitive', () => {
    expect(isCodeSnippetLanguage('TypeScript')).toBe(false);
    expect(isCodeSnippetLanguage('JavaScript')).toBe(false);
    expect(isCodeSnippetLanguage('PYTHON')).toBe(false);
  });
});

describe('getLanguageFromFilePath', () => {
  it('detects languages from standard file extensions', () => {
    expect(getLanguageFromFilePath('index.ts')).toBe('typescript');
    expect(getLanguageFromFilePath('app.js')).toBe('javascript');
    expect(getLanguageFromFilePath('script.py')).toBe('python');
    expect(getLanguageFromFilePath('main.go')).toBe('go');
    expect(getLanguageFromFilePath('lib.rs')).toBe('rust');
  });

  it('maps jsonl files to json', () => {
    expect(getLanguageFromFilePath('data/train.jsonl')).toBe('json');
  });

  it('detects languages from React file extensions', () => {
    expect(getLanguageFromFilePath('Component.tsx')).toBe('tsx');
    expect(getLanguageFromFilePath('Component.jsx')).toBe('jsx');
  });

  it('detects languages from config file extensions', () => {
    expect(getLanguageFromFilePath('package.json')).toBe('json');
    expect(getLanguageFromFilePath('config.yaml')).toBe('yaml');
  });

  it('detects languages from web file extensions', () => {
    expect(getLanguageFromFilePath('styles.css')).toBe('css');
    expect(getLanguageFromFilePath('index.html')).toBe('html');
  });

  it('detects languages from script file extensions', () => {
    expect(getLanguageFromFilePath('deploy.sh')).toBe('shell');
    expect(getLanguageFromFilePath('setup.bash')).toBe('bash');
  });

  it('detects markdown files', () => {
    expect(getLanguageFromFilePath('README.md')).toBe('markdown');
    expect(getLanguageFromFilePath('docs.markdown')).toBe('markdown');
  });

  it('handles file paths with directories', () => {
    expect(getLanguageFromFilePath('/path/to/file.ts')).toBe('typescript');
    expect(getLanguageFromFilePath('src/components/App.tsx')).toBe('tsx');
    expect(getLanguageFromFilePath('./utils/helper.js')).toBe('javascript');
  });

  it('handles files with multiple dots', () => {
    expect(getLanguageFromFilePath('file.test.ts')).toBe('typescript');
    expect(getLanguageFromFilePath('config.prod.json')).toBe('json');
  });

  it('is case-insensitive for extensions', () => {
    expect(getLanguageFromFilePath('file.TS')).toBe('typescript');
    expect(getLanguageFromFilePath('file.JS')).toBe('javascript');
    expect(getLanguageFromFilePath('file.PY')).toBe('python');
  });

  it('returns undefined for unknown extensions', () => {
    expect(getLanguageFromFilePath('file.txt')).toBe(undefined);
    expect(getLanguageFromFilePath('file.pdf')).toBe(undefined);
    expect(getLanguageFromFilePath('file.unknown')).toBe(undefined);
  });

  it('returns undefined for files without extensions', () => {
    expect(getLanguageFromFilePath('Makefile')).toBe(undefined);
    expect(getLanguageFromFilePath('README')).toBe(undefined);
  });

  it('returns undefined for empty or invalid inputs', () => {
    expect(getLanguageFromFilePath('')).toBe(undefined);
    expect(getLanguageFromFilePath('   ')).toBe(undefined);
    expect(getLanguageFromFilePath('.')).toBe(undefined);
  });

  it('handles paths ending with dot', () => {
    expect(getLanguageFromFilePath('file.')).toBe(undefined);
  });
});

describe('languageInCode', () => {
  it('correctly identifies supported language at start', () => {
    expect(languageInCode('javascript console.log("Hello");')).toBe('javascript');
    expect(languageInCode('python print("Hello")')).toBe('python');
    expect(languageInCode('typescript let x: number = 5;')).toBe('typescript');
  });

  it('returns undefined for unsupported languages', () => {
    expect(languageInCode('unknown puts "Hello"')).toBe(undefined);
    expect(languageInCode('blahblah fmt.Println("Hello")')).toBe(undefined);
    expect(languageInCode('cpp cout << "Hello";')).toBe(undefined);
  });

  it('is case-insensitive', () => {
    expect(languageInCode('PYTHON print("Hello")')).toBe('python');
    expect(languageInCode('JavaScript console.log("Hello");')).toBe('javascript');
    expect(languageInCode('TypeScript let x: number = 5;')).toBe('typescript');
  });

  it('handles code without language prefix', () => {
    expect(languageInCode('console.log("Hello");')).toBe(undefined);
    expect(languageInCode('let x = 5;')).toBe(undefined);
  });

  it('handles empty strings', () => {
    expect(languageInCode('')).toBe(undefined);
  });

  it('handles strings with only whitespace', () => {
    expect(languageInCode('   ')).toBe(undefined);
  });

  it('correctly identifies language names with spaces', () => {
    expect(languageInCode('typescript Console.WriteLine("Hello")')).toBe('typescript');
    expect(languageInCode('python   print("test")')).toBe('python');
  });

  it('recognizes language aliases', () => {
    expect(languageInCode('ts const x = 5;')).toBe('typescript');
    expect(languageInCode('js console.log("test");')).toBe('javascript');
    expect(languageInCode('py print("test")')).toBe('python');
    expect(languageInCode('sh echo "test"')).toBe('shell');
    expect(languageInCode('yml key: value')).toBe('yaml');
    expect(languageInCode('md # Header')).toBe('markdown');
  });

  it('handles all supported languages', () => {
    expect(languageInCode('tsx <div />')).toBe('tsx');
    expect(languageInCode('jsx <div />')).toBe('jsx');
    expect(languageInCode('json {"key": "value"}')).toBe('json');
    expect(languageInCode('css .class {}')).toBe('css');
    expect(languageInCode('html <div></div>')).toBe('html');
    expect(languageInCode('bash echo "test"')).toBe('bash');
    expect(languageInCode('shell echo "test"')).toBe('shell');
    expect(languageInCode('rust fn main() {}')).toBe('rust');
    expect(languageInCode('go func main() {}')).toBe('go');
    expect(languageInCode('yaml key: value')).toBe('yaml');
    expect(languageInCode('markdown # Header')).toBe('markdown');
  });

  it('handles multiple whitespace between language and code', () => {
    expect(languageInCode('typescript    const x = 5;')).toBe('typescript');
    expect(languageInCode('python\t\tprint("test")')).toBe('python');
  });
});
