// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Book } from '@nemo/icons/icons';
import { Button, Tooltip } from '@nvidia/foundations-react-core';
import { LINK_DOCS_STUDIO } from '@studio/constants/links';

export function DocumentationLink() {
  return (
    <Tooltip slotContent="Documentation" side="bottom">
      <Button asChild aria-label="Documentation" color="neutral" kind="tertiary" size="medium">
        <a href={LINK_DOCS_STUDIO} target="_blank" rel="noopener noreferrer">
          <Book />
        </a>
      </Button>
    </Tooltip>
  );
}
