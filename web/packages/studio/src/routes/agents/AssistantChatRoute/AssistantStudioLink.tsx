// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ArrowRight } from '@nemo/icons/icons';
import { Anchor } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { getStudioInternalLinkTarget } from '@studio/routes/agents/AssistantChatRoute/AssistantStudioLinkTarget';
import type { ReactNode } from 'react';
import { Link } from 'react-router';

interface AssistantStudioLinkProps {
  children?: ReactNode;
  href?: string;
}

export const ASSISTANT_STUDIO_LINK_CLASS =
  'inline-flex min-h-7 max-w-full items-center gap-density-xs rounded border border-base bg-[linear-gradient(135deg,var(--background-color-accent-green-subtle),var(--background-color-accent-teal-subtle)_58%,var(--background-color-interaction-base))] px-density-sm py-density-xs align-baseline text-sm font-medium leading-none no-underline shadow-sm transition-[background,box-shadow] hover:bg-[linear-gradient(135deg,var(--background-color-accent-green-subtle-hover),var(--background-color-accent-teal-subtle-hover)_58%,var(--background-color-interaction-hover))] hover:shadow';

export const AssistantStudioLink = ({ href, children }: AssistantStudioLinkProps) => {
  const workspace = useWorkspaceFromPathIfExists();
  const target = getStudioInternalLinkTarget(href, window.location.origin, workspace);

  if (!target) {
    return <span>{children}</span>;
  }

  return (
    <Anchor asChild>
      <Link className={ASSISTANT_STUDIO_LINK_CLASS} to={target}>
        <span className="truncate">{children}</span>
        <ArrowRight aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
      </Link>
    </Anchor>
  );
};
