// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkersContextValue } from '@studio/providers/workers/types';
import { createContext } from 'react';

export const WorkersContext = createContext<WorkersContextValue | null>(null);
