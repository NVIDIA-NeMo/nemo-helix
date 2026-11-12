// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateFilesetRequest } from '@nemo/sdk/generated/platform/schema';
import type { SourceFields, SourceTab } from '@studio/components/CreateDatasetPanel/parseSource';

export interface BuildCreateRequestInput {
  tab: SourceTab | 'upload';
  name: string;
  description: string;
  fields?: SourceFields;
  secrets?: { token?: string; apiKey?: string; accessKeyId?: string; secretAccessKey?: string };
  advanced?: {
    endpoint?: string;
    apiHost?: string;
    apiBaseUrl?: string;
    region?: string;
    endpointUrl?: string;
    signatureVersion?: string;
  };
}

export function buildCreateRequest(_input: BuildCreateRequestInput): CreateFilesetRequest {
  throw new Error('not implemented');
}
