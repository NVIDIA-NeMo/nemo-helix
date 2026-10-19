// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TraceFilter } from '@nemo/sdk/generated/platform/schema';

/** Column id of the Traces list's agent filter, which the table sends as this `TraceFilter` key. */
export const AGENT_NAME_FILTER_ID = 'agent_name' satisfies keyof TraceFilter;
