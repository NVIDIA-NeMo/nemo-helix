// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { Block, Button, Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { DescribeWithAiPanel } from '@studio/components/CreateFilesetStart/DescribeWithAiPanel';
import { ArrowLeft } from 'lucide-react';
import { useCallback, useState, type FC } from 'react';

interface Props {
  workspace: string;
  onBack: () => void;
  onContinue: (jobRequest: DataDesignerJobRequest) => void;
}

/**
 * The second screen of the "describe with AI" path. Kept apart from the options page
 * rather than folded into it: drafting is a whole screen's worth of work, and reaching it
 * is a step the user can go back from.
 */
export const AiDraftStep: FC<Props> = ({ workspace, onBack, onContinue }) => {
  const [jobRequest, setJobRequest] = useState<DataDesignerJobRequest | null>(null);

  // Identity-stable so it can be a dependency of the panel's generate callback.
  const handleValidConfig = useCallback(
    (request: DataDesignerJobRequest | null) => setJobRequest(request),
    []
  );

  return (
    <Stack className="h-full">
      <Block className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <Stack gap="density-2xl" padding="density-2xl" className="min-h-0 flex-1">
          <PageHeader
            slotHeading="Describe your fileset"
            slotDescription="Say what you need in plain language. AI drafts the columns and prompts, and you refine everything on the next screen."
          />
          <Flex justify="center" className="min-h-0 w-full flex-1">
            <div className="flex min-h-0 w-full max-w-[768px] flex-1 flex-col">
              <DescribeWithAiPanel workspace={workspace} onValidConfig={handleValidConfig} />
            </div>
          </Flex>
        </Stack>
      </Block>

      <Flex justify="center" className="shrink-0 border-t border-base bg-surface-base px-10 py-3">
        <Flex align="center" justify="between" className="w-full max-w-[768px]">
          <Button kind="tertiary" onClick={onBack}>
            <ArrowLeft size={16} aria-hidden />
            Back
          </Button>
          <Button
            color="brand"
            kind="primary"
            disabled={jobRequest === null}
            onClick={() => jobRequest && onContinue(jobRequest)}
          >
            Continue
          </Button>
        </Flex>
      </Flex>
    </Stack>
  );
};
