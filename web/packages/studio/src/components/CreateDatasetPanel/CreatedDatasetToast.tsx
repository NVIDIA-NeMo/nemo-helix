// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getDatasetDetailRoute } from '@studio/routes/utils';
import { Link } from 'react-router';

interface CreatedDatasetToastProps {
  workspace: string;
  name: string;
}

/**
 * Toast body for a dataset that finished creating after its panel closed, so `onCreated` no longer
 * applies. A router `Link`: Studio mounts its ToastProvider inside the root layout route, which
 * keeps the toast within the router context and lets the link navigate without a page reload.
 */
export function CreatedDatasetToast({ workspace, name }: CreatedDatasetToastProps) {
  return (
    <span>
      Dataset <strong>{name}</strong> was created.{' '}
      <Link className="underline" to={getDatasetDetailRoute(workspace, name)}>
        View dataset
      </Link>
    </span>
  );
}
