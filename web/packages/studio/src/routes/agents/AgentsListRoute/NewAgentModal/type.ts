// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FormModalProps } from '@nemo/common/src/components/FormModal';
import type { uploadAgentFormSchema } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type { z } from 'zod';

export type UploadAgentFormData = z.infer<typeof uploadAgentFormSchema>;

/** A file plus the path it was picked or dropped under, still including the root directory. */
export interface PickedFile {
  readonly file: File;
  readonly relativePath: string;
}

/** A picked file paired with its path inside the agent spec fileset. */
export interface UploadAgentEntry {
  readonly path: string;
  readonly file: File;
}

/** Which route into a new agent the modal is showing: a prompt to hand off, a directory to upload, or a repository to import. */
export type NewAgentTab = 'coding-agent-prompt' | 'upload' | 'github';

export interface NewAgentModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
}
