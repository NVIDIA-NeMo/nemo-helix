// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Spinner } from '@nvidia/foundations-react-core';
import type { ComponentProps, FC, PropsWithChildren } from 'react';

interface Props extends ComponentProps<typeof Button> {
  loading?: boolean;
  height?: number;
}

/**
 * Button for displaying a spinner to convey an in-progress button state
 */
export const LoadingButton: FC<PropsWithChildren<Props>> = ({
  children,
  loading,
  height = 40,
  ...buttonProps
}) => {
  const includeDisabled = {
    ...buttonProps,
    disabled: loading || buttonProps.disabled,
    style: { height, width: 'auto', ...buttonProps.style },
  };
  return (
    <Button color="brand" {...includeDisabled}>
      <Flex gap="density-md" align="center">
        {children}
        {loading && (
          <Spinner
            aria-label="Loading..."
            size="small"
            /* eslint-disable-next-line no-restricted-syntax */
            style={{ height: height / 2 }}
            className="[&_div]:h-full"
          />
        )}
      </Flex>
    </Button>
  );
};
