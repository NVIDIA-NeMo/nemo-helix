// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceDropdown } from '@studio/components/WorkspaceDropdown';
import { ROUTES, ROUTE_PARAMS as RP } from '@studio/constants/routes';
import { customizationJob1 } from '@studio/mocks/customizer/customization-jobs';
import { dataset1 } from '@studio/mocks/entity-store/datasets';
import { entityStorePromptTunedModel1 } from '@studio/mocks/entity-store/models';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { metricEvaluationJob1 } from '@studio/mocks/evaluation/v1/evaluations';
import { GuardrailChecksSubTab } from '@studio/routes/guardrails/GuardrailChecksTab/constants';
import { renderWithRouter, waitFor } from '@studio/tests/util/render';
import { generatePath } from 'react-router';

const pathParams = {
  [RP.workspace]: workspace1.name!,
  [RP.evaluationJobId]: metricEvaluationJob1.id!,
  [RP.customizationJobName]: customizationJob1.name!,
  [RP.modelNamespace]: entityStorePromptTunedModel1.workspace!,
  [RP.modelName]: entityStorePromptTunedModel1.name!,
  [RP.filesetId]: dataset1.id!,
  [RP.filesetName]: dataset1.name!,
  [RP.filePathEncoded]: '',
  [RP.folderPathEncoded]: '',
  [RP.evalConfigNamespace]: '',
  [RP.evalConfigName]: '',
  [RP.safeSynthesizerJobName]: '',
  [RP.dataDesignerJobName]: '',
  [RP.sessionId]: 'session-1',
  [RP.anonymizerJobName]: 'test-anonymizer-job',
  [RP.deploymentName]: '',
  [RP.deploymentPanelView]: '',
  [RP.agentName]: '',
  [RP.agentDeploymentName]: '',
  [RP.agentEvalJobName]: 'test-agent-eval-job',
  [RP.optimizeJobName]: 'test-optimize-job',
  [RP.jobName]: 'test-job',
  [RP.benchmarkName]: 'test-benchmark',
  [RP.experimentName]: 'test-experiment-group',
  [RP.evaluationName]: 'test-experiment',
  [RP.insightId]: 'test-insight',
  [RP.guardrailConfigName]: 'test-guardrail-config',
  [RP.guardrailChecksSubTab]: GuardrailChecksSubTab.Tests,
  [RP.virtualModelName]: 'test-virtual-model',
  [RP.pluginName]: '',
};

describe('AccessibleTitleE2E', () => {
  // TODO, why aren't we using it.each here?
  [{ label: 'project', routeSet: ROUTES.workspace }].forEach((routeWrapper) => {
    Object.entries(routeWrapper.routeSet).forEach(([, route]) => {
      it(`render ${routeWrapper.label} routes: ${route}`, async () => {
        renderWithRouter({
          history: route.includes(':') ? generatePath(route, pathParams) : route,
          overrideRoutes: [
            {
              path: ROUTES.workspace.index,
              element: <WorkspaceDropdown />,
            },
          ],
        });
        await waitFor(() => {
          expect(document.title).not.toEqual('Studio');
        });
      });
    });
  });
});
