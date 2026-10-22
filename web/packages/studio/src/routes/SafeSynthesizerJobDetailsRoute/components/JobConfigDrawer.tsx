// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeEditor } from '@nemo/common/src/components/CodeEditor';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import type { GenerateJob } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { SidePanel, Stack } from '@nvidia/foundations-react-core';
import { Cog } from '@nvidia/react-gui-icons/icons';
import { FC, useMemo } from 'react';
import yaml from 'yaml';

interface JobConfigDrawerProps {
  job: GenerateJob;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const JobConfigDrawer: FC<JobConfigDrawerProps> = ({ job, open, onOpenChange }) => {
  const toast = useToast();

  const configYaml = useMemo(() => {
    try {
      return yaml.stringify(job.spec.config);
    } catch {
      toast.error('Error serializing configuration', { durationMs: 3000 });
      return 'N/A';
    }
  }, [job.spec.config, toast]);

  return (
    <SidePanel
      slotHeading={
        <>
          <Cog />
          Job Configuration
        </>
      }
      side="right"
      open={open}
      onOpenChange={onOpenChange}
      bordered
      modal
      className="max-w-[800px] w-full"
      attributes={{
        SidePanelMain: { className: 'p-0' },
      }}
      data-testid="job-config-drawer"
    >
      <Stack className="m-density-xl">
        <CodeEditor content={configYaml} contentType={ContentType.YAML} readOnly />
      </Stack>
    </SidePanel>
  );
};
