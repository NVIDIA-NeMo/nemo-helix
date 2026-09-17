// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';
import ReactDiffViewer, { DiffMethod } from 'react-diff-viewer-continued';

interface ConfigDiffProps {
  before: string;
  after: string;
  //: The document's format, used only as the diff's label — the guardrail set is TOML, the sandbox
  //: policy is YAML, and the differ itself is format-agnostic (line/word based, no parsing).
  language: 'toml' | 'yaml';
}

// A git-style split diff of two config documents, themed to KUI's dark surface tokens.
const DIFF_STYLES = {
  variables: {
    dark: {
      diffViewerBackground: '#111827', // gray-900
      diffViewerColor: '#e5e7eb', // gray-200
      addedBackground: '#052e1690',
      addedColor: '#bbf7d0', // green-200
      removedBackground: '#450a0a90',
      removedColor: '#fecaca', // red-200
      wordAddedBackground: '#166534', // green-800
      wordRemovedBackground: '#991b1b', // red-800
      addedGutterBackground: '#065f4640',
      removedGutterBackground: '#7f1d1d40',
      gutterBackground: '#111827',
      gutterColor: '#6b7280', // gray-500
      codeFoldBackground: '#1f2937', // gray-800
      codeFoldGutterBackground: '#1f2937',
      emptyLineBackground: '#111827',
    },
  },
  contentText: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: '12px' },
  gutter: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: '11px' },
} as const;

export const ConfigDiff: FC<ConfigDiffProps> = ({ before, after, language }) => {
  if (before === after) {
    return (
      <Text kind="body/regular/sm" className="text-subtle">
        No changes.
      </Text>
    );
  }
  return (
    <div className="max-h-[480px] overflow-auto rounded-md border border-base">
      <ReactDiffViewer
        oldValue={before}
        newValue={after}
        splitView
        useDarkTheme
        compareMethod={DiffMethod.WORDS}
        leftTitle={`Before (${language})`}
        rightTitle={`After, hardened (${language})`}
        styles={DIFF_STYLES}
      />
    </div>
  );
};
