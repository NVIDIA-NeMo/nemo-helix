// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ENTITY_EMPTY_STATES,
  type EmptyStateEntityKey,
} from '@nemo/common/src/components/EntityEmptyState/registry';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  Button,
  CodeSnippet,
  type CodeSnippetLanguage,
  Flex,
  SegmentedControl,
  StatusMessage,
  Text,
} from '@nvidia/foundations-react-core';
import { type FC, useState } from 'react';
import { useNavigate } from 'react-router';

/** The governed empty-state variants. Errors are handled separately by `ErrorPanel`. */
export type EntityEmptyStateVariant = 'first-use' | 'no-results';

export interface EntityEmptyStateBaseProps {
  entity: EmptyStateEntityKey;
  className?: string;
  /** Resolves `<workspace>` in the CLI command and skill prompt. Omitted leaves the placeholder. */
  workspace?: string;
}

export type EntityEmptyStateProps = EntityEmptyStateBaseProps &
  (
    | {
        variant: 'first-use';
        /**
         * Overrides the registry create action's handler (e.g. opens a create modal).
         * When omitted, a `createAction.to` route is navigated to instead.
         */
        onCreate?: () => void;
        onClearFilters?: undefined;
      }
    | {
        variant: 'no-results';
        /** Clears the active filters/search. Required so `no-results` always offers a way out. */
        onClearFilters: () => void;
        onCreate?: undefined;
      }
  );

const WORKSPACE_PLACEHOLDER = '<workspace>';

const resolveWorkspace = (text: string | undefined, workspace: string | undefined) =>
  text && workspace ? text.replaceAll(WORKSPACE_PLACEHOLDER, workspace) : text;

/**
 * The single canonical empty state for Studio lists, tables, and panels. Copy,
 * CLI command, and skill prompt come from the entity registry, and the glyph
 * comes from the canonical `ENTITY_ICONS` map; the variant selects which
 * affordances render. See the `ui-design` skill's `empty-states` reference.
 */
export const EntityEmptyState: FC<EntityEmptyStateProps> = ({
  entity,
  variant,
  onCreate,
  onClearFilters,
  className,
  workspace,
}) => {
  const descriptor = ENTITY_EMPTY_STATES[entity];
  const navigate = useNavigate();

  if (variant === 'no-results') {
    return (
      <Centered className={className} testId="entity-empty-state-no-results">
        <StatusMessage
          slotHeading="No results found"
          slotSubheading="No items match your current search or filters."
          slotFooter={
            onClearFilters ? (
              <Button kind="tertiary" onClick={onClearFilters}>
                Clear Filters
              </Button>
            ) : null
          }
        />
      </Centered>
    );
  }

  const { heading, subheading, createAction } = descriptor;
  const Icon = ENTITY_ICONS[entity];
  const cliCommand = resolveWorkspace(descriptor.cliCommand, workspace);
  const skillPrompt = resolveWorkspace(descriptor.skillPrompt, workspace);
  const handleCreate =
    onCreate ?? (createAction?.to ? () => navigate(createAction.to as string) : undefined);
  const showCreate = !!createAction && !!handleCreate;

  return (
    <Centered className={className} testId="entity-empty-state-first-use">
      <StatusMessage
        slotMedia={<Icon className="size-12" />}
        slotHeading={heading}
        slotSubheading={subheading}
        slotFooter={
          showCreate ? (
            <Button color="brand" onClick={handleCreate}>
              {createAction.label}
            </Button>
          ) : null
        }
      />
      {(cliCommand || skillPrompt) && (
        <div className="mt-4 w-full max-w-[40rem]">
          {showCreate ? (
            <Text
              kind="body/regular/sm"
              className="block text-center text-secondary"
              data-testid="entity-empty-state-alternative"
            >
              Or use a coding agent or the CLI:
            </Text>
          ) : null}
          <SelfServiceHelp cliCommand={cliCommand} skillPrompt={skillPrompt} />
        </div>
      )}
    </Centered>
  );
};

const Centered: FC<{ children: React.ReactNode; className?: string; testId: string }> = ({
  children,
  className,
  testId,
}) => (
  <Flex
    direction="col"
    align="center"
    justify="center"
    className={`h-full w-full ${className ?? ''}`}
    data-testid={testId}
  >
    {children}
  </Flex>
);

/** Self-service help kind. */
type HelpKind = 'cli' | 'agent';

/**
 * A compact "Ask an agent · CLI" disclosure: a KUI CodeSnippet (with its
 * built-in copy affordance) whose action row hosts a tiny SegmentedControl to
 * switch between the agent prompt and the CLI command. Kept out of the
 * StatusMessage footer so the ≤2-action rule for empty states holds.
 */
const SelfServiceHelp: FC<{ cliCommand?: string; skillPrompt?: string }> = ({
  cliCommand,
  skillPrompt,
}) => {
  const toast = useToast();
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

  const items: { value: HelpKind; children: React.ReactNode }[] = [];
  if (skillPrompt)
    items.push({
      value: 'agent',
      children: 'Ask an agent',
    });
  if (cliCommand) items.push({ value: 'cli', children: 'CLI' });

  const showCli = kind === 'cli' && !!cliCommand;
  const value = showCli ? (cliCommand as string) : (skillPrompt ?? cliCommand ?? '');
  const language: CodeSnippetLanguage = showCli ? 'bash' : 'markdown';

  return (
    // Unwrapped, the snippet scrolls a long value out of sight instead of showing it.
    <div
      className="mt-density-lg [&_pre]:whitespace-pre-wrap [&_pre]:[overflow-wrap:anywhere]"
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
