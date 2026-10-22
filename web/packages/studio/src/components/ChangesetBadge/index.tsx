// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Anchor, Badge } from '@nvidia/foundations-react-core';
import { OpenExternal } from '@nvidia/react-gui-icons/icons';
import { type FC } from 'react';

interface ChangesetBadgeProps {
  /** The experiment's `source_link` URL. */
  href: string;
}

/**
 * Blue "Changeset" badge linking to an experiment's source changeset. Used on both the experiment
 * group table and the experiment detail header. The trailing external-link icon signals the link
 * opens the source in a new tab. `stopPropagation` keeps the link from also triggering a clickable
 * parent (e.g. a table row's row-click navigation).
 */
export const ChangesetBadge: FC<ChangesetBadgeProps> = ({ href }) => (
  <Anchor href={href} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
    <Badge color="blue" kind="solid">
      Changeset <OpenExternal width={14} height={14} aria-hidden />
    </Badge>
  </Anchor>
);
