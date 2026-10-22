// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Stack, Text } from '@nvidia/foundations-react-core';
import { IconError } from '@nvidia/react-gui-icons/icons';
import { FC } from 'react';
import { useNavigate } from 'react-router';

interface Props {
  header?: string;
  subheader?: string;
  message?: string;
}

/**
 * A generic Not Found message with a link button in a full height container. It's most likely
 * only going to be used in the router's `defaultNotFoundComponent`, but it could be reused
 * for any route's `notFoundComponent`.
 */
export const NotFound: FC<Props> = ({
  header = '404 Error',
  subheader = "Even AI can't find this page!",
  message = "If you're logged in, this might be a permissions issue. Check with your Org or Team Admin. Otherwise, you can return to your previous screen by clicking the link below.",
}) => {
  const navigate = useNavigate();
  const onBackClick = () => navigate(-1);

  return (
    <Stack className="h-full justify-center mx-auto max-w-[640px]" gap="density-md">
      <IconError className="size-16 stroke-2" color="var(--text-color-feedback-danger)" />
      <Text kind="display/xl">{header}</Text>
      <Text kind="title/lg">{subheader}</Text>
      <Text lineHeight="150">{message}</Text>
      <Button onClick={onBackClick} color="brand">
        Go Back
      </Button>
    </Stack>
  );
};
