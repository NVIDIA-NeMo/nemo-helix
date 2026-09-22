// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EvaluationResultsLayout } from '@studio/routes/evaluation/EvaluationResultsLayout';
import { EvaluationResultsRoute } from '@studio/routes/evaluation/EvaluationResultsRoute';
import { FC } from 'react';

/**
 * `/workspaces/:workspace/evaluation/new` -- the Evaluations list with the
 * wizard already open.
 *
 * The wizard is a dialog rather than a page, so this route exists only to keep
 * the URL linkable. It renders the list itself instead of relying on an
 * `Outlet`, because this path is a sibling of the results route rather than a
 * child of it.
 */
export const EvaluationNewRoute: FC = () => (
  <EvaluationResultsLayout newEvaluationOpen>
    <EvaluationResultsRoute />
  </EvaluationResultsLayout>
);
