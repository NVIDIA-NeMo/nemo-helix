// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeEditor } from '@nemo/common/src/components/CodeEditor';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import type { RailsConfig } from '@nemo/sdk/generated/platform/schema';
import { Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { Code } from '@nvidia/react-gui-icons/icons';
import type { FC } from 'react';

/**
 * The complete guardrail document, as it will be saved.
 *
 * Shown rather than hidden behind a disclosure: it is the only view of the fields Studio
 * cannot configure yet, and the only way to see what switching a rail on actually wrote.
 */
export const RawConfigSection: FC<{ data: RailsConfig }> = ({ data }) => (
  <Panel slotHeading="Configuration JSON" slotIcon={<Code />} elevation="high" density="compact">
    <Stack gap="density-sm">
      <Text kind="body/regular/sm" className="text-text-secondary">
        The full configuration, including any settings this page does not surface. Read-only.
      </Text>
      {/*
        A definite height, not min/max: CodeMirror sizes itself to the container, and with
        only a max-height the editor grows to the full document (3000px+ for a real config)
        and spills out of the panel instead of scrolling.
      */}
      <CodeEditor
        className="h-[420px]"
        content={JSON.stringify(data, null, 2)}
        readOnly
        contentType={ContentType.JSON}
      />
    </Stack>
  </Panel>
);
