// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Me, MeAuthMethod } from '@nemo/sdk/generated/platform/schema';
import { useMe } from '@studio/providers/auth/useMe';
import { useWebSession } from '@studio/providers/auth/useWebSession';
import type { User } from 'oidc-client-ts';
import { useMemo } from 'react';
import { useAuth } from 'react-oidc-context';

export type AuthProfile = {
  /** Principal id the platform records as `created_by` on resources this user creates. */
  id?: string;
  name: string;
  email: string;
  /** Personal workspace name derived from the email; only used to provision it on first login. */
  workspace: string;
  avatarUrl?: string | null;
  authMethod?: MeAuthMethod;
  state?: {
    path: string;
    search: string;
  };
};

const getWorkspace = (user: User) => {
  const email = user.profile.email ?? `${user.profile.unique_name}`;
  return email
    ? email.split('@')[0]
    : (user.profile.name?.replace(/\s+/g, '-').toLowerCase() ?? '');
};

export const getUserAuthProfile = (user: User): AuthProfile => ({
  name: user.profile.name ?? '',
  id: user.profile.sub,
  email: user.profile.email ?? `${user.profile.unique_name}`,
  workspace: getWorkspace(user),
  state: user.state as AuthProfile['state'],
});

const withMe = (profile: AuthProfile | undefined, me: Me | undefined): AuthProfile | undefined => {
  if (!me) return profile;
  return {
    ...profile,
    id: profile?.id ?? me.principal_id,
    name: me.profile.display_name,
    email: profile?.email || me.email || '',
    workspace: profile?.workspace ?? '',
    avatarUrl: me.profile.avatar_url,
    authMethod: me.auth_method,
  };
};

export const useAuthProfile = (): AuthProfile | undefined => {
  const auth = useAuth();
  const user = auth?.user;
  const { isServerSession, session } = useWebSession();
  const { data: me } = useMe();

  return useMemo(() => {
    if (isServerSession) {
      if (!session) return undefined;
      const email = session.email || session.id;
      return withMe({ id: session.id, name: email, email, workspace: email.split('@')[0] }, me);
    }
    return withMe(user ? getUserAuthProfile(user) : undefined, me);
  }, [isServerSession, session, user, me]);
};
