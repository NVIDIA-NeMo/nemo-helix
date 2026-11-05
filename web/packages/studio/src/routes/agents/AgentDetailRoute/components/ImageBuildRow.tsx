// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Warning } from '@nemo/icons/icons';
import {
  Button,
  Flex,
  Spinner,
  Stack,
  StatusIndicator,
  Text,
} from '@nvidia/foundations-react-core';
import type { PackageAgentBuild } from '@studio/api/agents/usePackageAgent';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import type { FC, ReactNode } from 'react';
import { useNavigate } from 'react-router';

type Tone = 'progress' | 'warning' | 'danger' | 'success';

interface BuildState {
  tone: Tone;
  title: string;
  detail?: ReactNode;
}

/**
 * What the latest build has to say, or nothing. A build from an earlier page load
 * only speaks while it is still going: its outcome is already in the list, or is
 * old news.
 */
const buildState = (build: PackageAgentBuild): BuildState | null => {
  const { isQueued, isRunning, isStalled, isUnreachable, isRestored } = build;
  if (isStalled) {
    return {
      tone: 'warning',
      title: 'Waiting for the image build to start',
      detail:
        'The job was accepted but has not started. Check that the platform is running a jobs controller.',
    };
  }
  if (isQueued) return { tone: 'progress', title: 'Waiting for a build to start…' };
  if (isRunning) {
    return { tone: 'progress', title: 'Building image', detail: 'This takes a few minutes.' };
  }
  if (isUnreachable) {
    return {
      tone: 'danger',
      title: 'Lost track of this build',
      detail: 'Its status could not be read. Open the job, or start another build.',
    };
  }
  if (isRestored) return null;
  if (build.isFailed) {
    return {
      tone: 'danger',
      title: 'Packaging failed',
      detail: 'Open the job for the build output.',
    };
  }
  if (build.isResultPending) return { tone: 'progress', title: 'Reading the build result…' };
  if (build.resultError) {
    return {
      tone: 'danger',
      title: 'The build finished, but its result could not be read',
      detail: 'Open the job for the tag.',
    };
  }
  if (build.isComplete && !build.image) {
    return {
      tone: 'warning',
      title: 'The job finished without reporting an image tag',
      detail: 'Open the job to see why.',
    };
  }
  if (build.isComplete && build.published) {
    return { tone: 'success', title: 'Image built', detail: `Pushed to ${build.published}` };
  }
  return null;
};

const TONE_TEXT: Record<Tone, string> = {
  progress: 'text-secondary',
  warning: 'text-feedback-warning',
  danger: 'text-feedback-danger',
  success: 'text-secondary',
};

const ToneIcon: FC<{ tone: Tone }> = ({ tone }) => {
  if (tone === 'progress') {
    return <Spinner size="small" aria-label="Building image" className="mt-0.5 shrink-0" />;
  }
  if (tone === 'success') return <StatusIndicator color="green" size="small" className="mt-1.5" />;
  return (
    <Warning
      aria-hidden
      className={`mt-0.5 size-4 shrink-0 ${tone === 'danger' ? 'text-feedback-danger' : 'text-feedback-warning'}`}
    />
  );
};

/** The latest image build, as a row above the images it produces. */
export const ImageBuildRow: FC<{ workspace: string; build: PackageAgentBuild }> = ({
  workspace,
  build,
}) => {
  const navigate = useNavigate();
  const state = buildState(build);
  if (!state) return null;
  const { jobName } = build;

  return (
    <Flex align="start" gap="2" className="px-4 py-3" data-testid="image-build-row">
      <ToneIcon tone={state.tone} />
      <Stack gap="0" className="min-w-0 flex-1">
        <Text kind="body/semibold/sm">{state.title}</Text>
        {state.detail ? (
          <Text kind="body/regular/xs" className={TONE_TEXT[state.tone]}>
            {state.detail}
          </Text>
        ) : null}
      </Stack>
      {jobName ? (
        <Button
          kind="tertiary"
          size="small"
          className="shrink-0"
          onClick={() => navigate(getWorkspaceJobDetailRoute(workspace, jobName))}
        >
          View job
        </Button>
      ) : null}
    </Flex>
  );
};
