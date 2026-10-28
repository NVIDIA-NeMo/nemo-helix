// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeSnippet, Modal } from '@nvidia/foundations-react-core';
import { paramValue } from '@studio/components/dataViews/OptimizationJobsDataView/utils';
import type { Trial } from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import type { FC } from 'react';

export interface TrialTextModalProps {
  trial: Trial | null;
  /** The parameter whose full text is shown, such as `system_prompt`. */
  name: string;
  onClose: () => void;
}

export const TrialTextModal: FC<TrialTextModalProps> = ({ trial, name, onClose }) => (
  <Modal
    open={trial !== null}
    onOpenChange={onClose}
    slotHeading={trial ? `Trial ${trial.number}: ${name}` : undefined}
    className="w-[90vw] max-w-[1100px]"
  >
    {trial && (
      <CodeSnippet
        value={paramValue(trial, name) ?? ''}
        language="markdown"
        kind="block"
        attributes={{ CodeSnippetCode: { className: 'max-h-[60vh] overflow-auto' } }}
      />
    )}
  </Modal>
);
