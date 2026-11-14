// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text } from '@nvidia/foundations-react-core';

interface FileRowProps {
  path: string;
  onRemove: () => void;
}

/** One staged loose file with its remove button. */
export function FileRow({ path, onRemove }: FileRowProps) {
  return (
    <Flex align="center" justify="between" gap="density-md">
      <Text kind="body/regular/md" className="min-w-0 break-all">
        {path}
      </Text>
      <Button
        type="button"
        kind="tertiary"
        size="small"
        aria-label={`Remove ${path}`}
        onClick={onRemove}
      >
        Remove
      </Button>
    </Flex>
  );
}
