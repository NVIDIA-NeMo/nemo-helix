// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ENTITY_EMPTY_STATES,
  type EmptyStateEntityKey,
} from '@nemo/common/src/components/EntityEmptyState/registry';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  CodeSnippet,
  type CodeSnippetLanguage,
  Flex,
  SegmentedControl,
} from '@nvidia/foundations-react-core';
import { type FC, useState } from 'react';

/**
 * Fills each `<key>` placeholder whose value is known. Unknown or empty values
 * leave their placeholder in place, so the reader still sees what to supply.
 */
const resolvePlaceholders = (
  text: string | undefined,
  values: Readonly<Record<string, string | undefined>>
) =>
  text &&
  Object.entries(values).reduce(
    (resolved, [key, value]) => (value ? resolved.replaceAll(`<${key}>`, value) : resolved),
    text
  );

/** Which self-service snippet is showing. */
type HelpKind = 'cli' | 'agent';

export interface EntityAgentBridgeProps {
  entity: EmptyStateEntityKey;
  /** Resolves `<workspace>` in the CLI command and skill prompt. Omitted leaves the placeholder. */
  workspace?: string;
  /**
   * Resolves any other `<key>` the page already knows, e.g. `{ 'trace-id': traceId }`, so a
   * pasted command runs as-is. Keys left out (or empty) keep their placeholder.
   */
  params?: Readonly<Record<string, string | undefined>>;
  className?: string;
}

/**
 * The handoff from Studio to the reader's own tools: the entity's CLI command
 * and its coding-agent prompt, in one KUI `CodeSnippet` (with its built-in copy
 * affordance) whose action row hosts a tiny `SegmentedControl` to switch
 * between them. Both come from the entity registry — never from the callsite.
 *
 * Renders nothing when the entity has neither, so a callsite can mount it
 * unconditionally rather than duplicating the registry lookup to decide.
 *
 * {@link EntityEmptyState} embeds this for every empty state. It is exported
 * separately for the surfaces an empty state cannot cover — a bespoke guidance
 * panel whose own actions exceed the empty-state footer rule, for one.
 */
export const EntityAgentBridge: FC<EntityAgentBridgeProps> = ({
  entity,
  workspace,
  params,
  className,
}) => {
  const toast = useToast();
  const descriptor = ENTITY_EMPTY_STATES[entity];
  const values = { ...params, workspace };
  const cliCommand = resolvePlaceholders(descriptor.cliCommand, values);
  const skillPrompt = resolvePlaceholders(descriptor.skillPrompt, values);
  const defaultKind: HelpKind = skillPrompt ? 'agent' : 'cli';

  // `kind` should track the current descriptor's default unless the user has picked a value for
  // it. Resetting during render (rather than in an effect) when the descriptor changes avoids a
  // stale selection painting for a frame, while leaving a same-descriptor rerender's user choice
  // untouched.
  const [prevDescriptor, setPrevDescriptor] = useState({ cliCommand, skillPrompt });
  const [kind, setKind] = useState<HelpKind>(defaultKind);
  if (prevDescriptor.cliCommand !== cliCommand || prevDescriptor.skillPrompt !== skillPrompt) {
    setPrevDescriptor({ cliCommand, skillPrompt });
    setKind(defaultKind);
  }

  if (!cliCommand && !skillPrompt) return null;

  const items: { value: HelpKind; children: React.ReactNode }[] = [];
  if (skillPrompt) items.push({ value: 'agent', children: 'Ask an agent' });
  if (cliCommand) items.push({ value: 'cli', children: 'CLI' });

  const showCli = kind === 'cli' && !!cliCommand;
  const value = showCli ? (cliCommand as string) : (skillPrompt ?? cliCommand ?? '');
  const language: CodeSnippetLanguage = showCli ? 'bash' : 'markdown';

  return (
    // Unwrapped, the snippet scrolls a long value out of sight instead of showing it.
    <div
      className={`[&_pre]:whitespace-pre-wrap [&_pre]:[overflow-wrap:anywhere] ${className ?? ''}`}
      data-testid="entity-empty-state-help"
    >
      <CodeSnippet
        value={value}
        language={language}
        kind="block"
        onCopySuccess={() => toast.success('Copied to clipboard')}
        slotActions={
          items.length > 1 ? (
            <Flex className="w-full" justify="between" align="center" wrap="wrap">
              <SegmentedControl
                size="tiny"
                className="!w-fit"
                value={kind}
                onValueChange={(next) => setKind(next as HelpKind)}
                items={items}
              />
            </Flex>
          ) : undefined
        }
      />
    </div>
  );
};
