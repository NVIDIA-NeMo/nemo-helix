// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkCodeSnippet } from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkCodeSnippet';
import {
  type FileEdit,
  getFileLanguage,
  getFileName,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/workParts';
import {
  WORK_ICON_SIZE,
  WorkStep,
} from '@nemo/common/src/components/AssistantChat/chainOfThought/WorkStep';
import { Pencil } from '@nemo/icons/icons';

export const FileEditStep = ({
  defaultOpen,
  edit,
}: {
  readonly defaultOpen?: boolean;
  readonly edit: FileEdit;
}) => (
  <WorkStep
    data-testid="assistant-work-file-edit"
    defaultOpen={defaultOpen}
    icon={<Pencil size={WORK_ICON_SIZE} />}
    title={`Edited ${getFileName(edit.path)}`}
  >
    <WorkCodeSnippet label={edit.path} language={getFileLanguage(edit.path)} value={edit.content} />
  </WorkStep>
);
