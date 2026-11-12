// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import type { PrincipalNameLayout } from '@studio/components/PrincipalName/types';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import { usePrincipalDisplay } from '@studio/hooks/usePrincipalDisplay';
import { useMe } from '@studio/providers/auth/useMe';
import type { FC } from 'react';

interface PrincipalNameProps {
  principal: string | null | undefined;
  layout?: PrincipalNameLayout;
  bold?: boolean;
}

/**
 * A principal's display name. Users can set their own names, so the underlying principal is always
 * shown alongside it (and the identity provider's name too, when the name is an override); otherwise
 * anyone could appear as someone else.
 */
export const PrincipalName: FC<PrincipalNameProps> = ({ principal, layout = 'inline', bold }) => {
  const { data: me } = useMe();
  const { data: resolved } = usePrincipalDisplay(principal);
  const kind = bold ? 'body/bold/md' : 'body/regular/md';

  if (!principal) {
    // Resources created with auth off record no creator; they all belong to the local account.
    const localName = me?.auth_method === 'disabled' ? me.profile.display_name : EMPTY_FIELD_VALUE;
    return <Text kind={kind}>{localName}</Text>;
  }

  const name = resolved?.display_name;
  if (!name || name === principal) {
    return (
      <Text kind={kind} className="truncate" title={principal}>
        {principal}
      </Text>
    );
  }

  const identity =
    resolved.display_name_is_override && resolved.idp_display_name
      ? `${principal} · ${resolved.idp_display_name}`
      : principal;

  if (layout === 'inline') {
    // A self-chosen name could imitate anyone, so it never appears without the principal.
    return (
      <Text kind={kind} className="truncate" title={identity}>
        {name}
        {resolved.display_name_is_override && (
          <span className="text-secondary"> ({principal})</span>
        )}
      </Text>
    );
  }
  return (
    <Stack gap="density-xxs" className="min-w-0">
      <Text kind={kind} className="truncate" title={name}>
        {name}
      </Text>
      <Text kind="body/regular/sm" className="truncate text-secondary" title={identity}>
        {identity}
      </Text>
    </Stack>
  );
};
