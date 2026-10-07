// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkersContext } from '@studio/providers/workers/WorkersContext';
import { useContext } from 'react';

export const useWorkers = () => {
  const context = useContext(WorkersContext);
  if (!context) {
    throw new Error('useWorkers must be used within a WorkersProvider');
  }
  return context;
};
