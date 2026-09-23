// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CodeSnippet,
  Flex,
  SidePanelContent,
  SidePanelDialog,
  SidePanelHeading,
  SidePanelRoot,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { GeneratedConfigPanelProps } from '@studio/components/CreateFilesetStart/types';
import { FileJson } from 'lucide-react';
import type { FC } from 'react';

/**
 * Read-only view of what the model actually returned, for checking a draft before loading it
 * — or working out why one was rejected. Shows the tool-call arguments verbatim, so it can
 * differ from the config that lands on the canvas when the builder substitutes a model.
 */
export const GeneratedConfigPanel: FC<GeneratedConfigPanelProps> = ({ open, config, onClose }) => (
  <SidePanelRoot open={open} onOpenChange={onClose} modal>
    <SidePanelDialog>
      <SidePanelContent className="w-[720px]" bordered>
        <SidePanelHeading>
          <Flex gap="density-md" align="center">
            <FileJson size={20} aria-hidden />
            Generated config
          </Flex>
        </SidePanelHeading>
        <Stack gap="density-md" padding="4" className="h-full min-h-0 overflow-y-auto">
          <Text kind="body/regular/sm" className="text-secondary">
            Raw output from the model. Any model the builder had to substitute is listed as a
            warning next to the result.
          </Text>
          <CodeSnippet
            value={config}
            language="json"
            kind="block"
            attributes={{
              CodeSnippetCode: {
                className:
                  'max-h-[calc(100vh-220px)] [&_code]:whitespace-pre-wrap [&_code]:break-words [&_pre]:whitespace-pre-wrap',
              },
            }}
          />
        </Stack>
      </SidePanelContent>
    </SidePanelDialog>
  </SidePanelRoot>
);
