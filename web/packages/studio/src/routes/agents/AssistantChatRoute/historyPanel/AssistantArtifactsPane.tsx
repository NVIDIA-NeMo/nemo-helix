// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Collapse } from '@nemo/icons/icons';
import { Button, Divider, Flex, Stack, Text, Tooltip } from '@nvidia/foundations-react-core';
import { Empty } from '@studio/components/Empty';
import {
  ArtifactRow,
  FileArtifacts,
  JobArtifacts,
  LinkArtifacts,
  SelectionArtifacts,
  ToolArtifacts,
} from '@studio/routes/agents/AssistantChatRoute/historyPanel/ArtifactSections';
import { getSelectedArtifactModel } from '@studio/routes/agents/AssistantChatRoute/historyPanel/helpers';
import type { AssistantChatArtifacts } from '@studio/routes/agents/AssistantChatRoute/types';
import { Fragment, type ReactNode } from 'react';

interface ArtifactPaneSection {
  content: ReactNode;
  id: string;
}

export const AssistantArtifactsPane = ({
  artifacts,
  collapseLabel,
  onCollapse,
}: {
  artifacts?: AssistantChatArtifacts;
  collapseLabel: string;
  onCollapse: () => void;
}) => {
  const agent = artifacts?.agent?.trim();
  const selectedModel = artifacts ? getSelectedArtifactModel(artifacts)?.trim() : undefined;
  const selections = artifacts?.selections.filter(
    (selection) => selection.label.trim() && selection.value.trim()
  );
  const jobs = artifacts?.jobs.filter((job) => job.name.trim());
  const files = artifacts?.files.filter((file) => file.path.trim());
  const links = artifacts?.links.filter((link) => link.label.trim());
  const tools = artifacts?.tools.filter((tool) => tool.trim());
  const sections: ArtifactPaneSection[] = [];

  if (agent || selectedModel) {
    sections.push({
      id: 'summary',
      content: (
        <Stack gap="density-sm" className="min-w-0">
          <ArtifactRow label="Agent" value={agent} />
          <ArtifactRow label="Model" value={selectedModel} />
        </Stack>
      ),
    });
  }

  if (selections?.length) {
    sections.push({
      id: 'selections',
      content: <SelectionArtifacts selections={selections} />,
    });
  }

  if (jobs?.length) {
    sections.push({
      id: 'jobs',
      content: <JobArtifacts jobs={jobs} workspace={artifacts?.workspace} />,
    });
  }

  if (files?.length) {
    sections.push({ id: 'files', content: <FileArtifacts files={files} /> });
  }

  if (links?.length) {
    sections.push({ id: 'links', content: <LinkArtifacts links={links} /> });
  }

  if (tools?.length) {
    sections.push({ id: 'tools', content: <ToolArtifacts tools={tools} /> });
  }

  return (
    <section
      aria-label="Chat artifacts"
      className="flex min-h-0 basis-1/2 shrink-0 flex-col overflow-hidden rounded border border-base bg-surface-base dark:bg-surface-raised"
    >
      <Flex
        align="center"
        justify="between"
        gap="density-sm"
        className="border-b border-base px-density-md py-density-sm"
      >
        <Text kind="label/bold/md" className="min-w-0 truncate">
          Chat artifacts
        </Text>
        <Tooltip slotContent={collapseLabel} side="left">
          <Button
            aria-label={collapseLabel}
            kind="tertiary"
            size="small"
            type="button"
            onClick={onCollapse}
          >
            <Collapse size="18" />
          </Button>
        </Tooltip>
      </Flex>
      {sections.length ? (
        <Stack gap="density-sm" padding="density-md" className="min-h-0 flex-1 overflow-y-auto">
          {sections.map((section, index) => (
            <Fragment key={section.id}>
              {index > 0 && <Divider className="h-auto! flex-none!" />}
              {section.content}
            </Fragment>
          ))}
        </Stack>
      ) : (
        <Flex className="min-h-0 flex-1 px-density-md" align="center" justify="center">
          <Empty title="No artifacts yet" description="Selections and outputs will appear here." />
        </Flex>
      )}
    </section>
  );
};
