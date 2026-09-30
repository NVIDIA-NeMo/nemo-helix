// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeSnippet, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export interface StepCommandsProps {
  commands: readonly string[];
  /** Names the copy/expand buttons, so they differ from step to step. */
  title: string;
}

/** The CLI view's body: the step's `nemo` command(s), full width under the description. */
export const StepCommands: FC<StepCommandsProps> = ({ commands, title }) => (
  <CodeSnippet
    kind="block"
    language="bash"
    collapsible
    defaultOpen
    value={commands.join('\n')}
    // The actions row is justify-end, so an auto margin pins the prompt left and
    // leaves copy/expand on the right. Decorative, so it stays out of the a11y tree.
    slotActions={
      <Text kind="mono/md" aria-hidden="true" className="mr-auto">
        {'>_'}
      </Text>
    }
    className="mt-density-md [&_pre]:[overflow-wrap:anywhere] [&_pre]:whitespace-pre-wrap"
    // KUI hardcodes "Copy code" / "expand", which would give every step in the
    // feed the same accessible name.
    attributes={{
      CodeSnippetCopyButton: { 'aria-label': `Copy the ${title} command` },
      CodeSnippetExpanderButton: { 'aria-label': `Toggle the ${title} command` },
    }}
  />
);
