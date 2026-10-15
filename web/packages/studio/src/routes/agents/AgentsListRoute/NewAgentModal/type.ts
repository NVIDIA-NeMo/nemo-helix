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

/** What one source (a pick, a drop or an image read) currently contributes to the form. */
export interface AgentSourceSelection {
  readonly entries: UploadAgentEntry[];
  /** Shown with the file count: the picked root folder or the image reference. */
  readonly label: string;
  readonly error?: string;
  /** The image the entries were read from, so they stop counting once the reference changes. */
  readonly image?: string;
}

/** Which route into a new agent the modal is showing: an agent already seen in ingested traces, a
 *  prompt to hand off, files to upload, a repository to import, or a packaged image to deploy. */
export type NewAgentTab = 'imported-traces' | 'coding-agent-prompt' | 'upload' | 'github' | 'image';

export interface NewAgentModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  workspace: string;
  /** Prefills the name, and keeps it when an uploaded config names a different agent. */
  initialName?: string;
}
