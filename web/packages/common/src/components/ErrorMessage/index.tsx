// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  StatusMessage,
  StatusMessageProps,
  Flex,
  Stack,
  Button,
} from '@nvidia/foundations-react-core';
import { IconError } from '@nvidia/react-gui-icons/icons';
import { FC, ReactNode } from 'react';
import { useNavigate } from 'react-router';

export interface ErrorMessageProps {
  header?: ReactNode;
  message?: ReactNode;
  slotMedia?: StatusMessageProps['slotMedia'];
  slotFooter?: StatusMessageProps['slotFooter'];
  height?: string;
}

export const ErrorMessage: FC<ErrorMessageProps> = ({
  header = 'Error',
  message = 'An unexpected error occurred',
  slotMedia,
  slotFooter,
  height = '100%',
}) => {
  const navigate = useNavigate();

  const defaultRenderAction = (
    <Flex gap="density-sm">
      <Button kind="secondary" onClick={() => navigate(-1)}>
        Go Back
      </Button>
      <Button kind="secondary" onClick={() => window.location.reload()}>
        Refresh Page
      </Button>
    </Flex>
  );

  return (
    // eslint-disable-next-line no-restricted-syntax
    <Stack gap="density-md" align="center" justify="center" style={{ height }}>
      <StatusMessage
        slotHeading={header}
        slotMedia={slotMedia ?? <IconError className="size-16" />}
        slotSubheading={message}
        slotFooter={slotFooter ?? defaultRenderAction}
        className="max-w-[640px]"
      />
    </Stack>
  );
};
