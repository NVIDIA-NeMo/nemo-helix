// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
import { TELEMETRY_ENABLED } from '@studio/constants/environment';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { useAuthSignOut } from '@studio/providers/auth/useWebSession';
import { Route } from 'lucide-react';
import { useState } from 'react';

export const UserPopover = () => {
  const [openModal, setOpenModal] = useState<'trace' | undefined>(undefined);
  const profile = useAuthProfile();
  const signOut = useAuthSignOut();

  if (!profile && !TELEMETRY_ENABLED) {
    return null;
  }

  return (
    <span data-tour="nav-user">
      <DropdownRoot defaultOpen={false}>
        <DropdownTrigger asChild>
          <Button color="neutral" kind="tertiary" className="p-0">
            <Avatar fallback={profile?.name.charAt(0) || 'N'} interactive />
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
              <Text fontSize="14" kind="label/bold/sm" className="text-center w-full">
                {profile.email}
              </Text>
            </DropdownHeading>
          )}
          {TELEMETRY_ENABLED && (
            <DropdownItem onClick={() => setOpenModal('trace')} slotStart={<Route />}>
              Report a Trace
            </DropdownItem>
          )}
          {profile && (
            <>
              <Divider />
              <DropdownItem onClick={() => void signOut()}>
                <Flex gap="density-lg" align="center">
                  Sign Out
                </Flex>
              </DropdownItem>
            </>
          )}
        </DropdownContent>
      </DropdownRoot>
      {openModal === 'trace' && (
        <ReportTraceModal open={openModal === 'trace'} onClose={() => setOpenModal(undefined)} />
      )}
    </span>
  );
};
