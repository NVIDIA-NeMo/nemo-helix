// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormModal } from '@nemo/common/src/components/FormModal';
import {
  getAuthGetMeQueryKey,
  useAuthUpdateMyProfile,
} from '@nemo/sdk/generated/platform/profiles';
import type { Me } from '@nemo/sdk/generated/platform/schema';
import { FormField, TextInput } from '@nvidia/foundations-react-core';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, type FormEvent, useState } from 'react';

const MAX_DISPLAY_NAME_LENGTH = 256;

interface EditProfileModalProps {
  me: Me;
  onClose: () => void;
}

export const EditProfileModal: FC<EditProfileModalProps> = ({ me, onClose }) => {
  const queryClient = useQueryClient();
  const [displayName, setDisplayName] = useState(me.profile.display_name_override ?? '');
  const { mutateAsync, isPending, error } = useAuthUpdateMyProfile();

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // An empty name clears the override, falling back to the identity provider's name.
    const updated = await mutateAsync({ data: { display_name: displayName.trim() || null } });
    queryClient.setQueryData(getAuthGetMeQueryKey(), updated);
    onClose();
  };

  return (
    <FormModal
      open
      title="Edit profile"
      submitButtonText="Save"
      onSubmit={onSubmit}
      onClose={onClose}
      loading={isPending}
      errorText={error ? 'Could not update your profile. Try again.' : null}
      submitDisabled={displayName.trim() === (me.profile.display_name_override ?? '')}
    >
      <FormField
        slotLabel="Display name"
        slotHelp="Shown to other members next to your sign-in identity. Leave empty to use the name from your identity provider."
      >
        <TextInput
          value={displayName}
          maxLength={MAX_DISPLAY_NAME_LENGTH}
          placeholder={me.profile.idp_display_name ?? me.email ?? undefined}
          onChange={(e) => setDisplayName(e.target.value)}
        />
      </FormField>
    </FormModal>
  );
};
