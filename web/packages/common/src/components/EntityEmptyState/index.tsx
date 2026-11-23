// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EntityAgentBridge } from '@nemo/common/src/components/EntityAgentBridge';
import {
  ENTITY_EMPTY_STATES,
  type EmptyStateEntityKey,
} from '@nemo/common/src/components/EntityEmptyState/registry';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { Button, Flex, StatusMessage, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

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
  const hasBridge = !!descriptor.cliCommand || !!descriptor.skillPrompt;
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
      {hasBridge && (
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
          <EntityAgentBridge entity={entity} workspace={workspace} className="mt-density-lg" />
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
