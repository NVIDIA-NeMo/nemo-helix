// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromNamedEntityRef, NamedEntityRef } from '@nemo/common/src/namedEntity';
import {
  AGENTS_ENABLED,
  ANONYMIZER_ENABLED,
  BASE_MODELS_ENABLED,
  ASSISTANT_STUDIO_ENABLED,
  CUSTOMIZER_ENABLED,
  DASHBOARD_ENABLED,
  DATA_DESIGNER_ENABLED,
  DATASETS_ENABLED,
  DEPLOYMENTS_ENABLED,
  EVALUATOR_BENCHMARKS_ENABLED,
  EVALUATOR_ENABLED,
  EXPERIMENT_ENABLED,
  FILESET_DETAILS_ENABLED,
  GUARDRAILS_ENABLED,
  INFERENCE_PROVIDER_ENABLED,
  INTAKE_ENABLED,
  JOBS_ENABLED,
  MEMBERS_ENABLED,
  MODEL_COMPARE_ENABLED,
  OPTIMIZER_ENABLED,
  PLUGINS_ENABLED,
  SAFE_SYNTHESIZER_ENABLED,
  SECRETS_ENABLED,
  SETTINGS_ENABLED,
} from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { FilesetDetailTab } from '@studio/routes/FilesetDetailRoute/constants';
import type { GuardrailChecksSubTab } from '@studio/routes/guardrails/GuardrailChecksTab/constants';
import { generatePath, RouteObject } from 'react-router';

const gateRoutes = (enabled: boolean, routes: RouteObject | RouteObject[]) => {
  if (!enabled) return [];
  return Array.isArray(routes) ? routes : [routes];
};

export const gateBaseModelsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(BASE_MODELS_ENABLED, routes);

export const gateCustomizationRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(CUSTOMIZER_ENABLED, routes);

export const gateDashboardRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(DASHBOARD_ENABLED || ASSISTANT_STUDIO_ENABLED, routes);

export const gateDatasetsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(DATASETS_ENABLED, routes);

export const gateFilesetDetailsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(FILESET_DETAILS_ENABLED, routes);

export const gateJobsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(JOBS_ENABLED, routes);

export const gateSettingsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(SETTINGS_ENABLED, routes);

export const gateIntakeRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(INTAKE_ENABLED, routes);

export const gateSafeSynthesizerRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(SAFE_SYNTHESIZER_ENABLED, routes);

export const gateDataDesignerRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(DATA_DESIGNER_ENABLED, routes);

export const gateAnonymizerRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(ANONYMIZER_ENABLED, routes);

export const gateEvaluationRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(EVALUATOR_ENABLED, routes);

export const gateEvaluationBenchmarksRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(EVALUATOR_ENABLED && EVALUATOR_BENCHMARKS_ENABLED, routes);

export const gateExperimentRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(EXPERIMENT_ENABLED, routes);

export const gateSecretsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(SECRETS_ENABLED, routes);

export const gateGuardrailsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(GUARDRAILS_ENABLED, routes);

export const gateInferenceProviderRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(INFERENCE_PROVIDER_ENABLED, routes);

export const gateMembersRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(MEMBERS_ENABLED, routes);

export const agentsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(AGENTS_ENABLED, routes);

export const gatePluginRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(PLUGINS_ENABLED, routes);

export const gateAssistantStudioRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(ASSISTANT_STUDIO_ENABLED, routes);

export const gateDeploymentsRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(DEPLOYMENTS_ENABLED, routes);

export const gateModelCompareRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(MODEL_COMPARE_ENABLED, routes);

export const gateOptimizerRoutes = (routes: RouteObject | RouteObject[]) =>
  gateRoutes(OPTIMIZER_ENABLED, routes);

type WorkspacePathParams = {
  workspace: string;
};

type ModelRouteParams = {
  modelNamespace: string;
  modelName: string;
};

const getModelRouteParamsFromEntityRef = (model: NamedEntityRef): ModelRouteParams => {
  const { workspace, name } = getPartsFromNamedEntityRef(model);
  return { modelNamespace: workspace, modelName: name };
};

export const getCustomizationJobListRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.customizationJobList, { workspace });
};

export const getWorkspacePathParamsFromName = (workspace: string): WorkspacePathParams => {
  return { workspace };
};

export const getWorkspaceIndexRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.index, { workspace });
};

export const getWorkspaceDetailsDefaultRoute = (workspace: string) => {
  if (DASHBOARD_ENABLED || ASSISTANT_STUDIO_ENABLED) return getWorkspaceDashboardRoute(workspace);
  if (AGENTS_ENABLED) return getAgentsListRoute(workspace);
  if (BASE_MODELS_ENABLED) return getWorkspaceBaseModelsRoute(workspace);
  if (JOBS_ENABLED) return getWorkspaceJobsRoute(workspace);
  return getWorkspaceIndexRoute(workspace);
};

export const getWorkspaceDashboardRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.dashboard, { workspace });
};

export const getWorkspaceJobsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.jobs, { workspace });
};

export const getWorkspaceJobDetailRoute = (workspace: string, jobName: string) => {
  return generatePath(ROUTES.workspace.jobDetail, { workspace, jobName });
};

export type BaseModelsPanelTab = 'model-details' | 'chat-playground';

export const getWorkspaceBaseModelsRoute = (
  workspace: string,
  options?: { model?: string; tab?: BaseModelsPanelTab; searchParams?: URLSearchParams }
): string => {
  let path: string;
  const searchParams = new URLSearchParams(options?.searchParams);
  if (options?.model) {
    path = generatePath(ROUTES.workspace.baseModelsModel, {
      workspace,
      modelName: options.model,
    });
    if (options?.tab) {
      searchParams.set('tab', options.tab);
    }
  } else {
    path = generatePath(ROUTES.workspace.baseModels, { workspace });
  }

  const query = searchParams.toString();
  return query ? `${path}?${query}` : path;
};

export const getWorkspaceFilesetsRoute = (workspace: string) => {
  const searchParams = new URLSearchParams(window.location.search);
  const baseUrl = generatePath(ROUTES.workspace.filesets, { workspace });

  return searchParams.size ? `${baseUrl}?${searchParams.toString()}` : baseUrl;
};

export const getWorkspaceCustomizationJobDetailsRoute = (
  workspace: string,
  customizationJobName: string
) => {
  return generatePath(ROUTES.workspace.customizationJobDetails, {
    workspace,
    customizationJobName,
  });
};

export const getWorkspaceCustomizationJobListRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.customizationJobList, { workspace });
};

export const getWorkspaceEvaluationRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.evaluation, { workspace });
};

export const getWorkspaceInferenceProvidersRoute = (
  workspace: string,
  options?: { preset?: string }
): string => {
  const base = generatePath(ROUTES.workspace.inferenceProviders, { workspace });
  if (options?.preset) {
    return `${base}?create=true&preset=${encodeURIComponent(options.preset)}`;
  }
  return base;
};

export const getWorkspaceVirtualModelsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.virtualModels, { workspace });
};

export const getVirtualModelDetailRoute = (workspace: string, virtualModelName: string): string => {
  return generatePath(ROUTES.workspace.virtualModelDetail, { workspace, virtualModelName });
};

export const getVirtualModelDetailsRoute = (
  workspace: string,
  virtualModelName: string
): string => {
  return generatePath(ROUTES.workspace.virtualModelDetails, { workspace, virtualModelName });
};

export const getVirtualModelChatRoute = (workspace: string, virtualModelName: string): string => {
  return generatePath(ROUTES.workspace.virtualModelChat, { workspace, virtualModelName });
};

export const getWorkspaceDeploymentsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.deployments, { workspace });
};

/**
 * Create Deployment wizard page.
 *
 * `model` and `fileset` prefill the Workspace source; `model` wins when both are
 * given, matching the wizard's own precedence.
 */
export const getWorkspaceNewDeploymentRoute = (
  workspace: string,
  options?: { model?: string; fileset?: string }
) => {
  const path = generatePath(ROUTES.workspace.deploymentsNew, { workspace });
  const searchParams = new URLSearchParams();
  if (options?.model) {
    searchParams.set(QUERY_PARAMETERS.model, options.model);
  } else if (options?.fileset) {
    searchParams.set(QUERY_PARAMETERS.fileset, options.fileset);
  }
  const query = searchParams.toString();
  return query ? `${path}?${query}` : path;
};

/** Default segment for the deployment details side panel URL. */
export const DEPLOYMENT_DETAILS_PANEL_VIEW_DETAILS = 'details' as const;

export const getWorkspaceDeploymentDetailsRoute = (
  workspace: string,
  deploymentName: string,
  panelView: string = DEPLOYMENT_DETAILS_PANEL_VIEW_DETAILS
) => {
  return generatePath(ROUTES.workspace.deploymentsDeployment, {
    workspace,
    deploymentName: deploymentName,
    deploymentPanelView: panelView,
  });
};

export const getWorkspaceSafeSynthesizerRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.safeSynthesizer, { workspace });
};

export const getEvaluationRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.evaluation, { workspace });
};

export const getEvaluationMetricsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.evaluationMetrics, { workspace });
};

export const getNewEvaluationMetricRoute = (workspace: string, options?: { model?: string }) => {
  const searchParams = new URLSearchParams();
  if (options?.model) {
    searchParams.append('model', options.model);
  }
  const baseUrl = generatePath(ROUTES.workspace.evaluationMetricNew, {
    workspace,
  });
  return searchParams.size ? `${baseUrl}?${searchParams.toString()}` : baseUrl;
};

export const getEvaluationMetricDetailsRoute = (workspace: string, metricId: string) => {
  return generatePath(ROUTES.workspace.evaluationMetricDetails, {
    workspace,
    id: metricId,
  });
};

/** Run panel without a pre-selected metric (user picks from within the panel). */
export const getEvaluationMetricsRunRoute = (workspace: string, options?: { model?: string }) => {
  const basePath = generatePath(ROUTES.workspace.evaluationMetricsRun, { workspace });
  if (options?.model) {
    return `${basePath}?${QUERY_PARAMETERS.model}=${encodeURIComponent(options.model)}`;
  }
  return generatePath(ROUTES.workspace.evaluationMetricsRun, { workspace });
};

/** Run panel pre-populated for a specific metric. */
export const getEvaluationMetricRunRoute = (
  workspace: string,
  metricId: string,
  options?: { model?: string }
) => {
  const basePath = generatePath(ROUTES.workspace.evaluationMetricRun, { workspace, id: metricId });
  if (options?.model) {
    return `${basePath}?${QUERY_PARAMETERS.model}=${encodeURIComponent(options.model)}`;
  }
  return basePath;
};

export const getEvaluationBenchmarkListRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.evaluationBenchmarks, { workspace });
};

export const getEvaluationBenchmarkDetailsRoute = (workspace: string, benchmarkName: string) => {
  return generatePath(ROUTES.workspace.evaluationBenchmarkDetails, {
    workspace,
    benchmarkName,
  });
};

export const getEvaluationResultsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.evaluationResults, { workspace });
};

export const getEvaluationResultDetailsRoute = (workspace: string, jobName: string) => {
  return generatePath(ROUTES.workspace.evaluationResultDetails, {
    workspace,
    id: jobName,
  });
};

export const getExperimentRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.experiment, { workspace });
};

export const getExperimentDetailRoute = (workspace: string, experimentName: string) => {
  return generatePath(ROUTES.workspace.experimentDetail, {
    workspace,
    experimentName: experimentName,
  });
};

export const getEvaluationDetailRoute = (
  workspace: string,
  experimentName: string,
  evaluationName: string
) => {
  return generatePath(ROUTES.workspace.evaluationDetail, {
    workspace,
    experimentName: experimentName,
    evaluationName: evaluationName,
  });
};

export const getEvaluationSessionDetailRoute = (
  workspace: string,
  experimentName: string,
  evaluationName: string,
  sessionId: string
): string => {
  return generatePath(ROUTES.workspace.evaluationSessionDetail, {
    workspace,
    experimentName: experimentName,
    evaluationName: evaluationName,
    sessionId: sessionId,
  });
};

export const getEvaluationSessionTraceDetailRoute = (
  workspace: string,
  experimentName: string,
  evaluationName: string,
  sessionId: string,
  traceId: string,
  options?: { spanId?: string }
): string => {
  const path = getEvaluationSessionDetailRoute(
    workspace,
    experimentName,
    evaluationName,
    sessionId
  );
  const searchParams = new URLSearchParams({ [QUERY_PARAMETERS.traceId]: traceId });
  if (options?.spanId) searchParams.set(QUERY_PARAMETERS.spanId, options.spanId);
  return `${path}?${searchParams.toString()}`;
};

export const getNewCustomizationJobRoute = (workspace: string, options?: { model?: string }) => {
  const basePath = generatePath(ROUTES.workspace.newCustomizationJob, { workspace });
  if (options?.model) {
    return `${basePath}?${QUERY_PARAMETERS.model}=${encodeURIComponent(options.model)}`;
  }
  return basePath;
};

export const getNewFilesetRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.filesetNew, { workspace });
};

export const getSecretsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.secrets, { workspace });
};

export const getGuardrailsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.guardrails, { workspace });
};

export const getOptimizerRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.optimizer, { workspace });
};

export const getOptimizerInsightRoute = (workspace: string, insightId: string) => {
  return generatePath(ROUTES.workspace.optimizerInsight, { workspace, insightId });
};

export const getGuardrailDetailRoute = (workspace: string, guardrailConfigName: string) => {
  return generatePath(ROUTES.workspace.guardrailDetail, {
    workspace,
    guardrailConfigName,
  });
};

export const getGuardrailConfigRoute = (workspace: string, guardrailConfigName: string) => {
  return generatePath(ROUTES.workspace.guardrailConfig, {
    workspace,
    guardrailConfigName,
  });
};

export const getGuardrailChecksRoute = (workspace: string, guardrailConfigName: string) => {
  return generatePath(ROUTES.workspace.guardrailChecks, {
    workspace,
    guardrailConfigName,
  });
};

export const getGuardrailVirtualModelsRoute = (workspace: string, guardrailConfigName: string) => {
  return generatePath(ROUTES.workspace.guardrailVirtualModels, {
    workspace,
    guardrailConfigName,
  });
};

export const getGuardrailChecksSubTabRoute = (
  workspace: string,
  guardrailConfigName: string,
  subTab: GuardrailChecksSubTab
): ReturnType<typeof generatePath> => {
  return generatePath(ROUTES.workspace.guardrailChecksSubTab, {
    workspace,
    guardrailConfigName,
    guardrailChecksSubTab: subTab,
  });
};

export const getWorkspaceSettingsRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.settings, { workspace });
};

export const getWorkspaceMembersRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.members, { workspace });
};

export const getModelCompareRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.modelCompare, { workspace });
};

export const getFilesetDetailsRoute = (
  workspace: string,
  filesetId: string,
  filesetFolder?: string,
  resetPage?: boolean
) => {
  const searchParams = new URLSearchParams(window.location.search);
  if (filesetFolder) {
    searchParams.set(QUERY_PARAMETERS.filesetFolder, filesetFolder);
  } else {
    searchParams.delete(QUERY_PARAMETERS.filesetFolder);
  }
  const baseUrl = generatePath(ROUTES.workspace.filesetDetails, {
    workspace,
    filesetId,
  });

  if (resetPage) {
    // To make sure the new fileset is front and center we need to remove some query params
    // If we are OK not showing the current fileset in the table this could be removed
    searchParams.delete('page');
    searchParams.delete('sort_by');
    searchParams.delete('order');
  }

  return searchParams.size ? `${baseUrl}?${searchParams.toString()}` : baseUrl;
};

export const getFilesetDetailRoute = (
  workspace: string,
  filesetName: string,
  options?: { tab?: FilesetDetailTab }
) => {
  const base = generatePath(ROUTES.workspace.filesetDetail, { workspace, filesetName });
  return options?.tab ? `${base}?${QUERY_PARAMETERS.tab}=${options.tab}` : base;
};

export const getFilesetRoute = (
  workspace: string,
  filesetRef: string,
  options?: { tab?: FilesetDetailTab }
) => {
  if (FILESET_DETAILS_ENABLED) {
    return getFilesetDetailRoute(workspace, filesetRef.split('/').pop() ?? filesetRef, options);
  }
  return getFilesetDetailsRoute(
    workspace,
    filesetRef.includes('/') ? filesetRef : `${workspace}/${filesetRef}`
  );
};

export const getFilesetFileRoute = (workspace: string, fileset: string, filePath: string) => {
  return generatePath(ROUTES.workspace.filesetFile, {
    workspace,
    filesetId: fileset,
    filePathEncoded: filePath,
  });
};

export const getIntakeTracesRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.intakeTraces, { workspace });
};

export const getIntakeSpansRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.intakeSpans, { workspace });
};

export const getIntakeSessionRoute = (workspace: string, sessionId: string) => {
  return generatePath(ROUTES.workspace.intakeSession, {
    workspace,
    sessionId: sessionId,
  });
};

export const getIntakeSessionTraceRoute = (
  workspace: string,
  sessionId: string,
  traceId: string,
  options?: { spanId?: string }
) => {
  const searchParams = new URLSearchParams({ [QUERY_PARAMETERS.traceId]: traceId });
  if (options?.spanId) searchParams.set(QUERY_PARAMETERS.spanId, options.spanId);
  return `${getIntakeSessionRoute(workspace, sessionId)}?${searchParams.toString()}`;
};

export const getSafeSynthesizerRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.safeSynthesizer, { workspace });
};

export const getNewSafeSynthesizerRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.safeSynthesizerNew, { workspace });
};

export const getGenerateJobRoute = (workspace: string, safeSynthesizerJobName: string) => {
  return generatePath(ROUTES.workspace.safeSynthesizerJob, {
    workspace,
    safeSynthesizerJobName,
  });
};

export const getGenerateJobReportRoute = (workspace: string, safeSynthesizerJobName: string) => {
  return generatePath(ROUTES.workspace.safeSynthesizerJobReport, {
    workspace,
    safeSynthesizerJobName,
  });
};

export const getDataDesignerJobListRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.dataDesignerJobList, { workspace });
};

export const getDataDesignerJobDetailsRoute = (workspace: string, dataDesignerJobName: string) => {
  return generatePath(ROUTES.workspace.dataDesignerJobDetails, {
    workspace,
    dataDesignerJobName,
  });
};

export const getNewDataDesignerJobRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.dataDesignerJobNew, { workspace });
};

export const getDataDesignerJobBuildRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.dataDesignerJobBuild, { workspace });
};

/** Not linked from any UI — reachable only by typing the URL. */
export const getLegacyNewDataDesignerJobRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.dataDesignerJobNewLegacy, { workspace });
};

export const getWorkspaceAnonymizerRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.anonymizer, { workspace });
};

export const getNewAnonymizerRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.anonymizerNew, { workspace });
};

export const getAnonymizerJobRoute = (workspace: string, anonymizerJobName: string) => {
  return generatePath(ROUTES.workspace.anonymizerJob, { workspace, anonymizerJobName });
};

export const getModelChatRoute = (model: NamedEntityRef) => {
  const { modelNamespace, modelName } = getModelRouteParamsFromEntityRef(model);
  return generatePath(ROUTES.models.modelChat, { modelNamespace, modelName });
};

export const getAgentsListRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.agentsList, { workspace });
};

export const getAssistantChatRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.assistantChat, { workspace });
};

export const getAgentDetailRoute = (workspace: string, agentName: string) => {
  return generatePath(ROUTES.workspace.agentDetail, { workspace, agentName });
};

export const getAgentEvaluationsTabRoute = (workspace: string, agentName: string) => {
  return `${getAgentDetailRoute(workspace, agentName)}?tab=evaluations`;
};

export const getAgentOptimizationsTabRoute = (workspace: string, agentName: string) => {
  return `${getAgentDetailRoute(workspace, agentName)}?tab=optimizations`;
};

export const getAgentOptimizationDetailRoute = (workspace: string, optimizeJobName: string) => {
  return generatePath(ROUTES.workspace.agentOptimizationDetail, {
    workspace,
    optimizeJobName,
  });
};

export const getAgentDeploymentsListRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.agentDeploymentsList, { workspace });
};

export const getAgentDeploymentDetailRoute = (workspace: string, agentDeploymentName: string) => {
  return generatePath(ROUTES.workspace.agentDeploymentDetail, {
    workspace,
    agentDeploymentName,
  });
};

export const getAgentMonitorRoute = (workspace: string) => {
  return generatePath(ROUTES.workspace.agentMonitor, { workspace });
};

export const getAgentEvaluationDetailRoute = (workspace: string, agentEvalJobName: string) => {
  return generatePath(ROUTES.workspace.agentEvaluationDetail, {
    workspace,
    agentEvalJobName,
  });
};
