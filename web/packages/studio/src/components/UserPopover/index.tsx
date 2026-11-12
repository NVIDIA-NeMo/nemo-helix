// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Pencil, Route } from '@nemo/icons/icons';
import {
  Avatar,
  Button,
  Divider,
  DropdownContent,
  DropdownHeading,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
  Flex,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { ReportTraceModal } from '@studio/components/ReportTraceModal';
import { EditProfileModal } from '@studio/components/UserPopover/EditProfileModal';
import { TELEMETRY_ENABLED } from '@studio/constants/environment';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { useMe } from '@studio/providers/auth/useMe';
import { useAuthSignOut } from '@studio/providers/auth/useWebSession';
import { useState } from 'react';

export const UserPopover = () => {
  const [openModal, setOpenModal] = useState<'trace' | 'profile' | undefined>(undefined);
  const profile = useAuthProfile();
  const { data: me } = useMe();
  const signOut = useAuthSignOut();

  if (!profile && !TELEMETRY_ENABLED) {
    return null;
  }

  return (
    <span data-tour="nav-user">
      <DropdownRoot defaultOpen={false}>
        <DropdownTrigger asChild>
          <Button color="neutral" kind="tertiary" className="p-0">
            <Avatar
              fallback={profile?.name.charAt(0) || 'N'}
              src={profile?.avatarUrl ?? undefined}
              alt=""
              referrerPolicy="no-referrer"
              interactive
            />
            {profile && (
              <Stack align="start" className="px-2" gap="density-xs">
                <Text fontSize="14">{profile.name}</Text>
              </Stack>
            )}
          </Button>
        </DropdownTrigger>
        <DropdownContent align="end" className="w-[300px] mt-[8px]" density="spacious">
          {profile && (
            <DropdownHeading>
              <Stack align="center" className="w-full" gap="density-xxs">
                <Text fontSize="14" kind="label/bold/sm">
                  {profile.name}
                </Text>
                {profile.email && profile.email !== profile.name && (
                  <Text fontSize="12" kind="body/regular/sm" className="text-secondary">
                    {profile.email}
                  </Text>
                )}
              </Stack>
            </DropdownHeading>
          )}
          {me?.can_edit_profile && (
            <DropdownItem onClick={() => setOpenModal('profile')} slotStart={<Pencil />}>
              Edit profile
            </DropdownItem>
          )}
          {TELEMETRY_ENABLED && (
            <DropdownItem onClick={() => setOpenModal('trace')} slotStart={<Route />}>
              Report a Trace
            </DropdownItem>
          )}
          {/* With auth off there is no session to end; the local account is the only user. */}
          {profile && profile.authMethod !== 'disabled' && (
            <>
              <Divider />
              <DropdownItem onClick={signOut}>
                <Flex gap="density-lg" align="center">
                  Sign Out
                </Flex>
              </DropdownItem>
            </>
          )}
        </DropdownContent>
      </DropdownRoot>
      {openModal === 'profile' && me && (
        <EditProfileModal me={me} onClose={() => setOpenModal(undefined)} />
      )}
      {openModal === 'trace' && (
        <ReportTraceModal open={openModal === 'trace'} onClose={() => setOpenModal(undefined)} />
      )}
    </span>
  );
};
