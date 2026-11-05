// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExternalLink } from '@nemo/common/src/components/ExternalLink';
import type { AgentSpecSource } from '@studio/api/agents/useAgentSpecFileset';
import { shortRevision } from '@studio/routes/agents/AgentDetailRoute/helpers';
import type { FC } from 'react';

/** A commit, linked when the source it came from is still known and has a web view. */
export const CommitLink: FC<{ source?: AgentSpecSource; revision: string }> = ({
  source,
  revision,
}) =>
  source?.commitUrl ? (
    <ExternalLink href={source.commitUrl(revision)} textKind="body/regular/xs">
      {shortRevision(revision)}
    </ExternalLink>
  ) : (
    <>{shortRevision(revision)}</>
  );
