// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  MAX_AGENT_SPEC_FILES,
  MAX_PICKED_FILES,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type {
  PickedFile,
  UploadAgentEntry,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import {
  AgentConfigParseError,
  agentNameFromConfig,
  agentSpecFilesetName,
  collectAgentEntries,
  findNonUtf8Path,
  isIgnoredPath,
  parseAgentConfig,
  pickedFromDataTransfer,
  tooManyPickedFiles,
  validateAgentEntries,
} from '@studio/routes/agents/AgentsListRoute/NewAgentModal/utils';

const makeFile = (relativePath: string, contents = 'x'): PickedFile => ({
  file: new File([contents], relativePath.split('/').pop() ?? relativePath),
  relativePath,
});

const entry = (path: string, size = 1): UploadAgentEntry => ({
  path,
  file: { size } as File,
});

describe('collectAgentEntries', () => {
  it('strips the picked directory from each path', () => {
    const entries = collectAgentEntries([
      makeFile('calculator-agent/agent.yaml'),
      makeFile('calculator-agent/mcps/calculator.py'),
    ]);

    expect(entries.map((item) => item.path)).toEqual(['agent.yaml', 'mcps/calculator.py']);
  });

  it('drops build artifacts that the container path cannot stage', () => {
    const entries = collectAgentEntries([
      makeFile('agent/agent.yaml'),
      makeFile('agent/mcps/__pycache__/calculator.cpython-312.pyc'),
      makeFile('agent/.DS_Store'),
      makeFile('agent/.git/config'),
      makeFile('agent/node_modules/left-pad/index.js'),
    ]);

    expect(entries.map((item) => item.path)).toEqual(['agent.yaml']);
  });

  it('falls back to the file name when the picker reports no relative path', () => {
    const entries = collectAgentEntries([
      { file: new File(['x'], 'agent.yaml'), relativePath: 'agent.yaml' },
    ]);

    expect(entries.map((item) => item.path)).toEqual(['agent.yaml']);
  });
});

describe('isIgnoredPath', () => {
  it.each([
    ['mcps/__pycache__/calculator.pyc', true],
    ['.venv/lib/python3.12/site-packages/x.py', true],
    ['build/libthing.so', true],
    ['skills/review/SKILL.md', false],
    ['mcps/calculator.py', false],
  ])('%s -> %s', (path, expected) => {
    expect(isIgnoredPath(path)).toBe(expected);
  });
});

describe('validateAgentEntries', () => {
  it('requires agent.yaml at the top level', () => {
    expect(validateAgentEntries([entry('mcps/calculator.py')])).toMatch(/No agent\.yaml/);
    expect(validateAgentEntries([entry('nested/agent.yaml')])).toMatch(/No agent\.yaml/);
  });

  it('rejects an empty directory', () => {
    expect(validateAgentEntries([])).toMatch(/no uploadable files/);
  });

  it('rejects two roots that contribute the same path', () => {
    const entries = collectAgentEntries([
      makeFile('calc-agent/agent.yaml'),
      makeFile('other-agent/agent.yaml'),
    ]);

    expect(validateAgentEntries(entries)).toMatch(/more than one agent\.yaml/);
  });

  it('rejects a directory over the file-count limit', () => {
    const entries = [
      entry('agent.yaml'),
      ...Array.from({ length: MAX_AGENT_SPEC_FILES }, (_unused, index) => entry(`f${index}.md`)),
    ];

    expect(validateAgentEntries(entries)).toMatch(/the limit is 500/);
  });

  it('rejects a directory over the byte limit', () => {
    expect(validateAgentEntries([entry('agent.yaml', 900_001)])).toMatch(/the limit is 900 KB/);
  });

  it('accepts a directory within both limits', () => {
    expect(
      validateAgentEntries([entry('agent.yaml'), entry('mcps/calculator.py')])
    ).toBeUndefined();
  });
});

describe('parseAgentConfig', () => {
  it('returns the parsed config for the Fabric contract', () => {
    const config = parseAgentConfig('config_format: nemo-agents-spec-v1\nname: calc\n');

    expect(config.name).toBe('calc');
  });

  it('rejects a NAT workflow config', () => {
    expect(() => parseAgentConfig('config_format: nat-workflow-v1\n')).toThrow(
      AgentConfigParseError
    );
  });

  it('rejects a config with no config_format', () => {
    expect(() => parseAgentConfig('name: calc\n')).toThrow(AgentConfigParseError);
  });

  it('rejects YAML that is not a mapping', () => {
    expect(() => parseAgentConfig('- one\n- two\n')).toThrow(/must contain a YAML mapping/);
  });

  it('rejects malformed YAML', () => {
    expect(() => parseAgentConfig('a:\n  - b\n c: broken\n')).toThrow(/not valid YAML/);
  });
});

describe('agentNameFromConfig', () => {
  it('reads a non-empty name', () => {
    expect(agentNameFromConfig({ name: ' calc ' })).toBe('calc');
  });

  it('ignores a missing or blank name', () => {
    expect(agentNameFromConfig({})).toBeUndefined();
    expect(agentNameFromConfig({ name: '   ' })).toBeUndefined();
    expect(agentNameFromConfig({ name: 7 })).toBeUndefined();
  });
});

describe('agentSpecFilesetName', () => {
  it('matches the fileset name deployments stage from', () => {
    expect(agentSpecFilesetName('calc')).toBe('calc-ethos');
  });
});

describe('findNonUtf8Path', () => {
  const binaryEntry = (path: string, bytes: number[]): UploadAgentEntry => ({
    path,
    file: new File([new Uint8Array(bytes)], path.split('/').pop() ?? path),
  });

  const textEntry = (path: string, contents: string): UploadAgentEntry => ({
    path,
    file: new File([contents], path.split('/').pop() ?? path),
  });

  it('names the first file that is not valid UTF-8', async () => {
    const entries = [
      textEntry('agent.yaml', 'name: calc\n'),
      binaryEntry('logo.bin', [0xff, 0xfe, 0x00, 0x62]),
    ];

    await expect(findNonUtf8Path(entries)).resolves.toBe('logo.bin');
  });

  it('accepts multi-byte UTF-8, an empty file, and a BOM', async () => {
    const entries = [
      textEntry('agent.yaml', 'description: café ☕ — 名前\n'),
      textEntry('empty.md', ''),
      binaryEntry('bom.md', [0xef, 0xbb, 0xbf, 0x68, 0x69]),
    ];

    await expect(findNonUtf8Path(entries)).resolves.toBeUndefined();
  });

  it('ignores AGENT-SPEC.md, which container staging never reads', async () => {
    const entries = [binaryEntry('AGENT-SPEC.md', [0xff, 0xfe, 0x00])];

    await expect(findNonUtf8Path(entries)).resolves.toBeUndefined();
  });

  it('rejects a lone UTF-16 surrogate sequence', async () => {
    const entries = [binaryEntry('utf16.md', [0xed, 0xa0, 0x80])];

    await expect(findNonUtf8Path(entries)).resolves.toBe('utf16.md');
  });
});

describe('tooManyPickedFiles', () => {
  it('rejects a pick far larger than any agent directory', () => {
    expect(tooManyPickedFiles(880_000)).toMatch(/880,000 files/);
  });

  it('allows a pick within the inspectable ceiling', () => {
    expect(tooManyPickedFiles(MAX_PICKED_FILES)).toBeUndefined();
    expect(tooManyPickedFiles(12)).toBeUndefined();
  });
});

describe('pickedFromDataTransfer', () => {
  interface FakeTree {
    [name: string]: FakeTree | string;
  }

  const makeEntry = (name: string, fullPath: string, node: FakeTree | string): FileSystemEntry => {
    if (typeof node === 'string') {
      return {
        name,
        fullPath,
        isFile: true,
        isDirectory: false,
        file: (resolve: (file: File) => void) => resolve(new File([node], name)),
      } as unknown as FileSystemEntry;
    }

    const children = Object.entries(node).map(([child, value]) =>
      makeEntry(child, `${fullPath}/${child}`, value)
    );
    let drained = false;
    return {
      name,
      fullPath,
      isFile: false,
      isDirectory: true,
      createReader: () => ({
        readEntries: (resolve: (entries: FileSystemEntry[]) => void) => {
          resolve(drained ? [] : children);
          drained = true;
        },
      }),
    } as unknown as FileSystemEntry;
  };

  const drop = (tree: FakeTree, root = 'calc-agent'): DataTransferItem[] => [
    {
      kind: 'file',
      webkitGetAsEntry: () => makeEntry(root, `/${root}`, tree),
    } as unknown as DataTransferItem,
  ];

  it('walks a dropped directory into files with their paths', async () => {
    const picked = await pickedFromDataTransfer(
      drop({ 'agent.yaml': 'name: calc', mcps: { 'calculator.py': 'print(1)' } })
    );

    expect(picked.map((item) => item.relativePath).sort()).toEqual([
      'calc-agent/agent.yaml',
      'calc-agent/mcps/calculator.py',
    ]);
  });

  it('feeds collectAgentEntries the same shape a picked directory does', async () => {
    const picked = await pickedFromDataTransfer(
      drop({ 'agent.yaml': 'name: calc', mcps: { 'calculator.py': 'print(1)' } })
    );

    expect(collectAgentEntries(picked).map((entry) => entry.path)).toEqual([
      'agent.yaml',
      'mcps/calculator.py',
    ]);
  });

  it('does not descend into ignored directories', async () => {
    const picked = await pickedFromDataTransfer(
      drop({
        'agent.yaml': 'name: calc',
        __pycache__: { 'calculator.cpython-312.pyc': 'binary-ish' },
        node_modules: { 'left-pad': { 'index.js': 'x' } },
      })
    );

    expect(picked.map((item) => item.relativePath)).toEqual(['calc-agent/agent.yaml']);
  });

  it('ignores non-file drag items', async () => {
    const items = [{ kind: 'string', webkitGetAsEntry: () => null } as unknown as DataTransferItem];

    await expect(pickedFromDataTransfer(items)).resolves.toEqual([]);
  });
});
