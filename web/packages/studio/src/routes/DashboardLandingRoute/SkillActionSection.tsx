// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, Flex, Skeleton, Stack, Text } from '@nvidia/foundations-react-core';
import { Empty } from '@studio/components/Empty';
import type { SkillActionTemplate } from '@studio/routes/DashboardLandingRoute/skillActionSuggestions';
import type { FC } from 'react';

const SKILL_ACTION_CARD_WIDTH_CLASS = 'w-72';
const SKILL_ACTION_CARD_HEIGHT_CLASS = 'h-44';
const SKILL_ACTION_CARD_CLASS = `${SKILL_ACTION_CARD_HEIGHT_CLASS} ${SKILL_ACTION_CARD_WIDTH_CLASS} flex-none cursor-pointer shadow-none!`;

const HORIZONTAL_SCROLLBAR_CLASS = [
  '[scrollbar-width:thin]',
  '[scrollbar-color:var(--border-color-interaction-base)_var(--background-color-interaction-hover)]',
  '[&::-webkit-scrollbar]:h-2',
  '[&::-webkit-scrollbar-corner]:bg-transparent',
  '[&::-webkit-scrollbar-track]:rounded-full',
  '[&::-webkit-scrollbar-track]:bg-[var(--background-color-interaction-hover)]',
  '[&::-webkit-scrollbar-thumb]:rounded-full',
  '[&::-webkit-scrollbar-thumb]:bg-[var(--border-color-interaction-base)]',
  '[&::-webkit-scrollbar-thumb:hover]:bg-[var(--border-color-interaction-strong)]',
].join(' ');

export interface SkillActionCard extends SkillActionTemplate {
  skillName?: string;
  claudeName?: string;
}

interface SkillActionListProps {
  actions: SkillActionCard[];
  onSelect: (prompt: string) => void;
}

const SkillActionList: FC<SkillActionListProps> = ({ actions, onSelect }) => {
  return (
    <div
      aria-label="Skill action suggestions"
      className={`min-w-0 w-full overflow-x-auto ${HORIZONTAL_SCROLLBAR_CLASS}`}
    >
      <Flex
        align="stretch"
        gap="density-md"
        className="w-max min-w-full pb-6"
        data-testid="skill-action-row"
      >
        {actions.map((action) => (
          <div
            key={`${action.skillName ?? action.title}:${action.claudeName ?? action.prompt}`}
            className={`${SKILL_ACTION_CARD_HEIGHT_CLASS} ${SKILL_ACTION_CARD_WIDTH_CLASS} flex-none`}
            data-testid={action.skillName ? `skill-action-card-${action.skillName}` : undefined}
          >
            <Card
              asChild
              interactive
              className="h-full w-full overflow-hidden cursor-pointer shadow-none!"
            >
              <button
                type="button"
                className="flex h-full w-full flex-col gap-density-sm overflow-hidden p-density-md text-left"
                onClick={() => onSelect(action.prompt)}
              >
                <span className="flex size-8 shrink-0 items-center justify-center rounded bg-surface-raised text-accent">
                  {action.icon}
                </span>
                <Flex direction="col" gap="density-xxs" className="min-h-0 flex-1">
                  <Text kind="label/bold/sm" className="line-clamp-1 block">
                    {action.title}
                  </Text>
                  {action.skillName ? (
                    <Text
                      kind="body/regular/xs"
                      className="block truncate text-secondary"
                      data-testid="skill-action-skill-name"
                    >
                      {action.skillName}
                    </Text>
                  ) : null}
                  <Text kind="body/regular/sm" className="line-clamp-2 block text-secondary">
                    {action.description}
                  </Text>
                </Flex>
              </button>
            </Card>
          </div>
        ))}
      </Flex>
    </div>
  );
};

const SkillActionSkeleton = () => (
  <Flex gap="density-md" className="w-full overflow-hidden" data-testid="skill-actions-loading">
    {Array.from({ length: 3 }, (_, index) => (
      <Skeleton key={index} className={SKILL_ACTION_CARD_CLASS} />
    ))}
  </Flex>
);

export interface SkillActionSectionProps {
  actions: SkillActionCard[];
  isLoading: boolean;
  onSelect: (prompt: string) => void;
  totalSkillCount: number;
}

export const SkillActionSection: FC<SkillActionSectionProps> = ({
  actions,
  isLoading,
  onSelect,
  totalSkillCount,
}) => {
  if (isLoading) {
    return <SkillActionSkeleton />;
  }

  if (!totalSkillCount) {
    return (
      <div className="w-full" data-testid="skill-actions-empty">
        <Empty
          title="No skills found"
          description="NeMo Assistant skills will appear here once they are available in this workspace."
        />
      </div>
    );
  }

  if (!actions.length) {
    return (
      <Stack gap="density-sm" className="w-full text-center" data-testid="skill-actions-disabled">
        <Text className="text-secondary" kind="body/regular/sm">
          Skills are installed, but none are enabled for this workspace configuration.
        </Text>
      </Stack>
    );
  }

  return (
    <div className="min-w-0 w-full self-stretch">
      <SkillActionList actions={actions} onSelect={onSelect} />
    </div>
  );
};
