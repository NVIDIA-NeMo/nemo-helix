#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Post-processing script for Orval generated files.
 * Runs prettier and eslint fix on generated API files, and prefixes unused parameters with underscores.
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync, type Dirent } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import prettier from 'prettier';
import { serviceConfigs } from './constants';

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ALLOWED_SERVICE_PATHS: ReadonlySet<string> = new Set(
  Object.values(serviceConfigs).map((c) => c.path)
);
const rawServicePath = process.argv[2];

if (!rawServicePath) {
  console.error('Error: Service path is required');
  console.error('Usage: node format-generated.js <service-path>');
  process.exit(1);
}

if (!ALLOWED_SERVICE_PATHS.has(rawServicePath)) {
  console.error(`Error: Unknown service path: ${rawServicePath}`);
  console.error(`Allowed: ${[...ALLOWED_SERVICE_PATHS].join(', ')}`);
  process.exit(1);
}

const servicePath = rawServicePath;
const generatedPath = path.join(__dirname, '..', 'generated', servicePath);

console.log(`\n📝 Processing generated files in ${generatedPath}...`);

/**
 * Recursively get all .ts files in a directory
 */
function getTsFiles(dir: string): string[] {
  const files: string[] = [];

  try {
    const entries = readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        files.push(...getTsFiles(fullPath));
      } else if (entry.isFile() && entry.name.endsWith('.ts')) {
        files.push(fullPath);
      }
    }
  } catch {
    console.warn(`Warning: Could not read directory ${dir}`);
  }

  return files;
}

/**
 * Extract parameter names from a parameter string
 */
function extractParamNames(param: string): string[] {
  const names: string[] = [];

  // Handle destructured parameters: ({ signal }) or { signal } or { signal, query }
  const destructuredMatch = param.match(/^[({]?\s*\{([^}]+)\}\s*[})]?/);
  if (destructuredMatch) {
    const destructuredContent = destructuredMatch[1];
    // Split by comma, respecting nested structures
    let depth = 0;
    let current = '';
    for (const char of destructuredContent) {
      if (char === '{' || char === '[' || char === '(') depth++;
      else if (char === '}' || char === ']' || char === ')') depth--;
      else if (char === ',' && depth === 0) {
        const nameMatch = current.trim().match(/^(\w+)(?:\??\s*:\s*[^,}]+)?/);
        if (nameMatch) names.push(nameMatch[1]);
        current = '';
        continue;
      }
      current += char;
    }
    if (current.trim()) {
      const nameMatch = current.trim().match(/^(\w+)(?:\??\s*:\s*[^,}]+)?/);
      if (nameMatch) names.push(nameMatch[1]);
    }
  } else {
    // Regular parameter: name or name: Type or name?: Type
    const nameMatch = param.match(/^\s*(\w+)(?:\??\s*:\s*[^,]+)?/);
    if (nameMatch) {
      names.push(nameMatch[1]);
    }
  }

  return names;
}

/**
 * Parse parameters from a parameter list string, respecting nested structures
 */
function parseParameters(paramsStr: string): string[] {
  const params: string[] = [];
  let depth = 0;
  let current = '';

  for (let i = 0; i < paramsStr.length; i++) {
    const char = paramsStr[i];
    if (char === '(' || char === '[' || char === '{' || char === '<') depth++;
    else if (char === ')' || char === ']' || char === '}' || char === '>') depth--;
    else if (char === ',' && depth === 0) {
      if (current.trim()) {
        params.push(current.trim());
      }
      current = '';
      continue;
    }
    current += char;
  }

  if (current.trim()) {
    params.push(current.trim());
  }

  return params;
}

/**
 * Find function body boundaries - from opening brace to matching closing brace
 */
function findFunctionBody(
  content: string,
  startPos: number
): { body: string; endPos: number } | null {
  let depth = 0;
  let bodyStart = -1;

  for (let i = startPos; i < content.length; i++) {
    const char = content[i];
    if (char === '{') {
      if (depth === 0) bodyStart = i + 1;
      depth++;
    } else if (char === '}') {
      depth--;
      if (depth === 0 && bodyStart !== -1) {
        return {
          body: content.substring(bodyStart, i),
          endPos: i + 1,
        };
      }
    }
  }

  return null;
}

/**
 * Prefix unused parameters with underscore to suppress TypeScript warnings.
 * Handles the react-query client's per-tag files, which live directly under
 * the service root (as opposed to the zod client's `zod/` and `schema/`
 * subdirectories, which this must not touch).
 */
function prefixUnusedParameters(filePath: string): boolean {
  if (path.dirname(filePath) !== generatedPath) {
    return false;
  }

  console.log(`  Processing ${path.basename(filePath)}...`);

  let content = readFileSync(filePath, 'utf-8');
  let modified = false;

  // Find all function implementations (not overloads)
  // Pattern: export function name<T>(params): returnType { body }
  // or: export const name = <T>(params) => { body }
  const functionImplRegex = /(export\s+(?:function\s+\w+|const\s+\w+\s*=\s*))(<[^>]*>)?\s*\(/g;

  let match;
  const replacements: Array<{ start: number; end: number; replacement: string }> = [];

  while ((match = functionImplRegex.exec(content)) !== null) {
    const funcStart = match.index;
    const parenPos = funcStart + match[0].length - 1;

    // Extract parameters by finding matching closing parenthesis
    let paramDepth = 1;
    let paramEnd = parenPos + 1;
    let paramsStr = '';

    for (let i = parenPos + 1; i < content.length && paramDepth > 0; i++) {
      const char = content[i];
      if (char === '(' || char === '[' || char === '{' || char === '<') paramDepth++;
      else if (char === ')' || char === ']' || char === '}' || char === '>') {
        paramDepth--;
        if (paramDepth === 0) {
          paramsStr = content.substring(parenPos + 1, i);
          paramEnd = i + 1;
          break;
        }
      }
    }

    if (!paramsStr.trim()) continue;

    // Find the function body - skip overloads (those ending with ;)
    let bodyStart = paramEnd;
    let foundBody = false;
    let isOverload = false;

    // Look ahead to check if this is an overload
    const lookAhead = content.substring(paramEnd, Math.min(content.length, paramEnd + 200));
    // Overload pattern: ) : returnType ; (no { before ;)
    if (lookAhead.match(/^\s*\)\s*:\s*[^;{]+;\s*(?:export|$)/m)) {
      isOverload = true;
    }

    // Find opening brace for function body
    for (let i = paramEnd; i < Math.min(content.length, paramEnd + 1000); i++) {
      const char = content[i];

      // Semicolon before brace means it's an overload
      if (char === ';') {
        const afterSemicolon = content.substring(i + 1, Math.min(content.length, i + 50));
        if (!afterSemicolon.match(/^\s*\{/)) {
          isOverload = true;
        }
        break;
      }

      if (content.substring(i, i + 2) === '=>') {
        // Arrow function
        let arrowPos = i + 2;
        while (arrowPos < content.length && /\s/.test(content[arrowPos])) {
          arrowPos++;
        }
        if (content[arrowPos] === '{') {
          bodyStart = arrowPos;
          foundBody = true;
          break;
        }
      }

      if (char === '{') {
        bodyStart = i;
        foundBody = true;
        break;
      }
    }

    // Skip function overloads (type-only signatures)
    if (isOverload || !foundBody) continue;

    const bodyResult = findFunctionBody(content, bodyStart);
    if (!bodyResult) continue;

    const body = bodyResult.body;

    // Parse parameters
    const paramList = parseParameters(paramsStr);
    const modifiedParams: string[] = [];
    let paramsModified = false;

    for (const param of paramList) {
      if (!param.trim()) {
        modifiedParams.push(param);
        continue;
      }

      const paramNames = extractParamNames(param);
      if (paramNames.length === 0) {
        modifiedParams.push(param);
        continue;
      }

      // Check if any parameter name is used in the body
      let paramUsed = false;
      for (const paramName of paramNames) {
        // Skip if already prefixed
        if (paramName.startsWith('_')) {
          paramUsed = true;
          break;
        }

        // Check if parameter is used in the function body (as a whole word)
        // Find all occurrences and check if they're real usage (not type annotations)
        const paramUsageRegex = new RegExp(`\\b${paramName}\\b`, 'g');
        let match;
        let hasRealUsage = false;

        while ((match = paramUsageRegex.exec(body)) !== null) {
          const matchPos = match.index;
          const afterMatch = body.substring(
            matchPos + paramName.length,
            matchPos + paramName.length + 5
          );

          // If followed by :, it's likely a type annotation, skip it
          // Otherwise, it's a real usage
          if (!afterMatch.match(/^\s*:/)) {
            hasRealUsage = true;
            break;
          }
        }

        if (hasRealUsage) {
          paramUsed = true;
          break;
        }
      }

      // If parameter is not used, prefix the first name with underscore
      if (!paramUsed && paramNames.length > 0) {
        const firstParamName = paramNames[0];
        if (!firstParamName.startsWith('_')) {
          // Prefix the first occurrence of the parameter name in the param string
          const prefixedParam = param.replace(
            new RegExp(`\\b${firstParamName}\\b`),
            `_${firstParamName}`
          );
          modifiedParams.push(prefixedParam);
          paramsModified = true;
        } else {
          modifiedParams.push(param);
        }
      } else {
        modifiedParams.push(param);
      }
    }

    if (paramsModified) {
      const newParams = modifiedParams.join(', ');
      const funcNameMatch = content
        .substring(funcStart, parenPos)
        .match(/(?:function|const)\s+(\w+)/);
      const funcName = funcNameMatch ? funcNameMatch[1] : 'unknown';
      console.log(`    Found unused params in ${funcName}`);
      replacements.push({
        start: parenPos + 1,
        end: paramEnd - 1,
        replacement: newParams,
      });
      modified = true;
    }
  }

  // Apply replacements in reverse order to maintain positions
  for (let i = replacements.length - 1; i >= 0; i--) {
    const { start, end, replacement } = replacements[i];
    content = content.substring(0, start) + replacement + content.substring(end);
  }

  if (modified) {
    writeFileSync(filePath, content, 'utf-8');
    return true;
  }

  return false;
}

/**
 * Split an orval-generated zod "tags" file (one file per OpenAPI tag, containing
 * every operation's schemas/constants for that tag) into per-operation files,
 * with a barrel re-export at the original path.
 *
 * Why: orval's zod generator inlines every $ref recursively (orval#2535), so a
 * single tag file can be multiple megabytes. Rolldown does not tree-shake unused
 * exports out of that file — importing one constant pulls the whole thing into
 * the chunk. Physically splitting by operation lets the bundler include only
 * the operation files actually referenced.
 *
 * Returns the number of per-operation files written, or 0 if the file did not
 * look like a tags-mode zod output (e.g. it was already a barrel, or had no
 * top-level `export const`).
 */
const SECTION_WORDS = /Query|Body|Response|Path|Params|Schema/;

function findOperationRoot(exportName: string): string {
  const camel = exportName[0].toLowerCase() + exportName.slice(1);
  const match = camel.match(SECTION_WORDS);
  return match ? camel.slice(0, match.index) : camel;
}

function splitZodTagFile(filePath: string): number {
  const src = readFileSync(filePath, 'utf-8');

  const exportPositions: Array<{ start: number; name: string }> = [];
  const exportRegex = /^export const ([a-zA-Z_$][a-zA-Z0-9_$]*)/gm;
  let match: RegExpExecArray | null;
  while ((match = exportRegex.exec(src)) !== null) {
    exportPositions.push({ start: match.index, name: match[1] });
  }
  if (exportPositions.length === 0) return 0;

  const header = src.slice(0, exportPositions[0].start);

  // Group export blocks by operation root, preserving source order.
  const groups = new Map<string, string[]>();
  const order: string[] = [];
  for (let i = 0; i < exportPositions.length; i++) {
    const { start, name } = exportPositions[i];
    const end = i + 1 < exportPositions.length ? exportPositions[i + 1].start : src.length;
    const root = findOperationRoot(name);
    if (!groups.has(root)) {
      groups.set(root, []);
      order.push(root);
    }
    groups.get(root)!.push(src.slice(start, end));
  }

  // Skip files that contain only one operation — splitting them adds churn
  // without helping the bundler.
  if (order.length <= 1) return 0;

  const tagName = path.basename(filePath, '.ts');
  const outDir = path.join(path.dirname(filePath), tagName);
  mkdirSync(outDir, { recursive: true });

  for (const root of order) {
    const contents = header + groups.get(root)!.join('');
    writeFileSync(path.join(outDir, `${root}.ts`), contents, 'utf-8');
  }

  const barrelLines = [
    '/**',
    ' * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.',
    ' * SPDX-License-Identifier: Apache-2.0',
    ' *',
    ' * Auto-generated barrel that re-exports per-operation zod modules.',
    ' * Split from a single orval tag file to keep the bundler from pulling',
    ' * the entire tag into a route chunk. Do not edit manually.',
    ' */',
    ...order.map((root) => `export * from './${tagName}/${root}';`),
    '',
  ];
  writeFileSync(filePath, barrelLines.join('\n'), 'utf-8');

  return order.length;
}

function splitZodTagFilesIn(zodDir: string): void {
  let entries: Dirent[];
  try {
    entries = readdirSync(zodDir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.ts')) continue;
    const fullPath = path.join(zodDir, entry.name);
    const count = splitZodTagFile(fullPath);
    if (count > 0) {
      console.log(`    Split ${entry.name} into ${count} operation files`);
    }
  }
}

/**
 * Write a barrel re-exporting every react-query tag file at the service root.
 *
 * Most call sites import a handful of named hooks from a specific tag file
 * directly, which is the point of `mode: 'tags'`. A few call sites (the
 * plugin SDK bridge) need the entire service surface as one module — this
 * barrel serves that without reintroducing a single all-in-one file that
 * every narrow import pays for.
 */
function writeReactQueryBarrel(): void {
  const entries = readdirSync(generatedPath, { withFileTypes: true });
  const tagFiles = entries
    .filter((e) => e.isFile() && e.name.endsWith('.ts') && e.name !== 'index.ts')
    .map((e) => path.basename(e.name, '.ts'))
    .sort();

  const lines = [
    '/**',
    ' * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.',
    ' * SPDX-License-Identifier: Apache-2.0',
    ' *',
    ' * Auto-generated barrel re-exporting every per-tag module in this service.',
    ' * Only import this when you need the entire service surface (e.g. the',
    ' * plugin SDK bridge) — everywhere else, import the specific tag file.',
    ' * Do not edit manually.',
    ' */',
    ...tagFiles.map((tag) => `export * from './${tag}';`),
    '',
  ];
  writeFileSync(path.join(generatedPath, 'index.ts'), lines.join('\n'), 'utf-8');
}

async function formatWithPrettier(dir: string): Promise<void> {
  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await formatWithPrettier(fullPath);
      continue;
    }
    if (!entry.isFile()) continue;
    const fileInfo = await prettier.getFileInfo(fullPath);
    if (fileInfo.ignored || !fileInfo.inferredParser) continue;
    const opts = (await prettier.resolveConfig(fullPath)) ?? {};
    const source = readFileSync(fullPath, 'utf-8');
    const formatted = await prettier.format(source, { ...opts, filepath: fullPath });
    if (formatted !== source) {
      writeFileSync(fullPath, formatted, 'utf-8');
    }
  }
}

async function run(): Promise<void> {
  console.log('Prefixing unused parameters...');
  const tsFiles = getTsFiles(generatedPath);
  let modifiedCount = 0;
  for (const file of tsFiles) {
    if (prefixUnusedParameters(file)) {
      modifiedCount++;
    }
  }
  console.log(`  Modified ${modifiedCount} file(s)`);

  const zodDir = path.join(generatedPath, 'zod');
  console.log('Splitting zod tag files by operation...');
  splitZodTagFilesIn(zodDir);

  console.log('Writing react-query barrel...');
  writeReactQueryBarrel();

  console.log('Running prettier...');
  await formatWithPrettier(generatedPath);

  console.log('✅ Successfully processed generated files\n');
}

run().catch((error) => {
  console.error('❌ Error during processing:', (error as Error).message);
  process.exit(1);
});
