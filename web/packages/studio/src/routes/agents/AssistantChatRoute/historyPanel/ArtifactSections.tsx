// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Anchor, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { cleanAssistantArtifactText } from '@studio/routes/agents/AssistantChatRoute/artifacts';
import { ASSISTANT_STUDIO_LINK_CLASS } from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLink';
import { getStudioInternalLinkTarget } from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLinkTarget';
import { getFileLabel } from '@studio/routes/agents/AssistantChatRoute/historyPanel/helpers';
import type {
  AssistantChatFileArtifact,
  AssistantChatJobArtifact,
  AssistantChatLinkArtifact,
  AssistantChatSelectionArtifact,
} from '@studio/routes/agents/AssistantChatRoute/types';
import { getJobProgressDetailRoute } from '@studio/routes/agents/AssistantChatRoute/utils/jobProgress';
import { ArrowRight, File, Wrench } from 'lucide-react';
import { type ReactNode } from 'react';
import { Link } from 'react-router';

export const ToolCallSummary = ({ toolCalls }: { toolCalls: string[] }) => {
  if (!toolCalls.length) return null;

  return (
    <Flex className="min-w-0 text-secondary" align="center" gap="density-xs">
      <Wrench size={12} className="shrink-0" />
      <Text kind="body/regular/sm" className="truncate">
        {toolCalls.join(', ')}
      </Text>
    </Flex>
  );
};

export const ArtifactChip = ({ children }: { children: ReactNode }) => {
  const content = typeof children === 'string' ? cleanAssistantArtifactText(children) : children;

  return (
    <span className="inline-flex min-w-0 max-w-full items-center rounded-full border border-accent/30 bg-accent/10 px-2 py-0.5 text-accent">
      <Text kind="label/bold/sm" className="truncate">
        {content}
      </Text>
    </span>
  );
};

export const ArtifactRow = ({ label, value }: { label: string; value?: string }) => {
  if (!value) return null;

  return (
    <Flex align="center" gap="density-xs" className="min-w-0 flex-wrap">
      <Text kind="label/bold/sm" className="shrink-0 text-secondary">
        {label}:
      </Text>
      <ArtifactChip>{value}</ArtifactChip>
    </Flex>
  );
};

export const ArtifactSection = ({ children, title }: { children: ReactNode; title: string }) => (
  <Stack gap="density-xs" className="min-w-0">
    <Text className="text-secondary" kind="label/bold/sm">
      {title}
    </Text>
    {children}
  </Stack>
);

export const FileArtifacts = ({ files }: { files: AssistantChatFileArtifact[] }) => {
  if (!files.length) return null;

  return (
    <ArtifactSection title="Files">
      <Stack gap="density-xs">
        {files.slice(0, 6).map((file) => (
          <Flex
            key={`${file.action}-${file.path}`}
            align="center"
            gap="density-xs"
            className="min-w-0 py-density-xs"
            title={file.path}
          >
            <File aria-hidden="true" size={14} className="shrink-0 text-secondary" />
            <Text kind="label/bold/sm" className="shrink-0">
              {file.action}
            </Text>
            <Text kind="body/regular/sm" className="min-w-0 flex-1 truncate font-mono">
              {getFileLabel(file)}
            </Text>
          </Flex>
        ))}
      </Stack>
    </ArtifactSection>
  );
};

export const LinkArtifacts = ({ links }: { links: AssistantChatLinkArtifact[] }) => {
  const workspace = useWorkspaceFromPathIfExists();

  if (!links.length) return null;

  return (
    <ArtifactSection title="Studio links">
      <Flex gap="density-xs" className="min-w-0 flex-wrap">
        {links.slice(0, 6).map((link) => {
          const target = getStudioInternalLinkTarget(
            link.href ?? link.destination,
            window.location.origin,
            workspace
          );
          const label = cleanAssistantArtifactText(link.label);
          const key = `${link.label}-${link.destination ?? link.href ?? 'link'}`;

          return target ? (
            <Anchor asChild key={key}>
              <Link className={ASSISTANT_STUDIO_LINK_CLASS} to={target}>
                <span className="truncate">{label}</span>
                <ArrowRight aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
              </Link>
            </Anchor>
          ) : (
            <ArtifactChip key={key}>{label}</ArtifactChip>
          );
        })}
      </Flex>
    </ArtifactSection>
  );
};

const getJobArtifactHref = (
  job: AssistantChatJobArtifact,
  workspace: string | undefined
): string | undefined => {
  if (job.href) return job.href;
  if (!workspace) return undefined;

  return getJobProgressDetailRoute({
    jobName: job.name,
    jobType: job.job_type,
    source: job.source,
    workspace,
  });
};

export const JobArtifacts = ({
  jobs,
  workspace: artifactWorkspace,
}: {
  jobs: AssistantChatJobArtifact[];
  workspace?: string;
}) => {
  const currentWorkspace = useWorkspaceFromPathIfExists();
  const workspace = currentWorkspace ?? artifactWorkspace;

  if (!jobs.length) return null;

  return (
    <ArtifactSection title="Jobs">
      <Flex gap="density-xs" className="min-w-0 flex-wrap">
        {jobs.slice(0, 6).map((job) => {
          const target = getStudioInternalLinkTarget(
            getJobArtifactHref(job, workspace),
            window.location.origin,
            workspace
          );
          const label = cleanAssistantArtifactText(job.name);

          return target ? (
            <Anchor asChild key={job.name}>
              <Link className={ASSISTANT_STUDIO_LINK_CLASS} to={target}>
                <span className="truncate">{label}</span>
                <ArrowRight aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
              </Link>
            </Anchor>
          ) : (
            <ArtifactChip key={job.name}>{label}</ArtifactChip>
          );
        })}
      </Flex>
    </ArtifactSection>
  );
};

export const SelectionArtifacts = ({
  selections,
}: {
  selections: AssistantChatSelectionArtifact[];
}) => {
  if (!selections.length) return null;

  return (
    <ArtifactSection title="Selections">
      <Stack gap="density-xs">
        {selections.slice(0, 6).map((selection) => (
          <ArtifactRow key={selection.label} label={selection.label} value={selection.value} />
        ))}
      </Stack>
    </ArtifactSection>
  );
};

export const ToolArtifacts = ({ tools }: { tools: string[] }) => {
  if (!tools.length) return null;

  return (
    <ArtifactSection title="Tools">
      <Flex gap="density-xs" className="min-w-0 flex-wrap">
        {tools.slice(0, 8).map((tool) => (
          <ArtifactChip key={tool}>{tool}</ArtifactChip>
        ))}
      </Flex>
    </ArtifactSection>
  );
};
