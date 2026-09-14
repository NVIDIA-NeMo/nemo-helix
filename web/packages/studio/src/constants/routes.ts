/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

export const ROUTE_PARAMS = {
  completionId: 'completionId',
  evaluationJobId: 'id',
  customizationJobId: 'customizationJobId',
  customizationJobName: 'customizationJobName',
  filesetId: 'filesetId',
  filesetName: 'filesetName',
  filePathEncoded: 'filePathEncoded',
  folderPathEncoded: 'folderPathEncoded',
  workspace: 'workspace',
  modelNamespace: 'modelNamespace',
  modelName: 'modelName',
  evalConfigNamespace: 'configNamespace',
  evalConfigName: 'configName',
  safeSynthesizerJobName: 'safeSynthesizerJobName',
  dataDesignerJobName: 'dataDesignerJobName',
  sessionId: 'sessionId',
  anonymizerJobName: 'anonymizerJobName',
  deploymentConfigName: 'deploymentConfigName',
  deploymentName: 'deploymentName',
  /** Side panel mode under deployments (e.g. `details`). */
  deploymentPanelView: 'deploymentPanelView',
  agentName: 'agentName',
  agentDeploymentName: 'agentDeploymentName',
  agentEvalJobName: 'agentEvalJobName',
  jobName: 'jobName',
  /** Benchmark entity name segment under evaluation/benchmarks/:name */
  benchmarkName: 'benchmarkName',
  experimentName: 'experimentName',
  insightId: 'insightId',
  evaluationName: 'evaluationName',
  guardrailConfigName: 'guardrailConfigName',
  guardrailChecksSubTab: 'guardrailChecksSubTab',
  virtualModelName: 'virtualModelName',
  pluginName: 'pluginName',
} as const;

// Just an alias to make the routes more readable
const P = ROUTE_PARAMS;

export const ROUTES = {
  auth: {
    success: '/auth/success',
  },
  workspace: {
    /**  Just redirects to the workspace dashboard route */
    index: `/workspaces/:${P.workspace}`,
    dashboard: `/workspaces/:${P.workspace}/dashboard`,
    jobs: `/workspaces/:${P.workspace}/jobs`,
    jobDetail: `/workspaces/:${P.workspace}/jobs/:${P.jobName}`,
    newCustomizationJob: `/workspaces/:${P.workspace}/customizations/fine-tuned/new`,
    baseModels: `/workspaces/:${P.workspace}/base-models`,
    /** Base models list with a specific model panel open (model name in path) */
    baseModelsModel: `/workspaces/:${P.workspace}/base-models/:${P.modelName}`,
    evaluation: `/workspaces/:${P.workspace}/evaluation`,
    evaluationMetrics: `/workspaces/:${P.workspace}/evaluation/metrics`,
    evaluationMetricNew: `/workspaces/:${P.workspace}/evaluation/metrics/new`,
    /** Run panel without a pre-selected metric — user picks from within the panel */
    evaluationMetricsRun: `/workspaces/:${P.workspace}/evaluation/metrics/run`,
    evaluationMetricDetails: `/workspaces/:${P.workspace}/evaluation/metrics/:${P.evaluationJobId}`,
    evaluationMetricRun: `/workspaces/:${P.workspace}/evaluation/metrics/:${P.evaluationJobId}/run`,
    evaluationBenchmarks: `/workspaces/:${P.workspace}/evaluation/benchmarks`,
    evaluationBenchmarkDetails: `/workspaces/:${P.workspace}/evaluation/benchmarks/:${P.benchmarkName}`,
    evaluationResults: `/workspaces/:${P.workspace}/evaluation/results`,
    evaluationResultDetails: `/workspaces/:${P.workspace}/evaluation/results/:${P.evaluationJobId}`,
    /** Empty landing page for the EXPERIMENT feature (gated by VITE_FF_EXPERIMENT). */
    experiment: `/workspaces/:${P.workspace}/experiment`,
    experimentDetail: `/workspaces/:${P.workspace}/experiment/:${P.experimentName}`,
    evaluationDetail: `/workspaces/:${P.workspace}/experiment/:${P.experimentName}/:${P.evaluationName}`,
    evaluationSessionDetail: `/workspaces/:${P.workspace}/experiment/:${P.experimentName}/:${P.evaluationName}/sessions/:${P.sessionId}`,
    customizationJobList: `/workspaces/:${P.workspace}/customizations`,
    customizationJobDetails: `/workspaces/:${P.workspace}/customizations/:${P.customizationJobName}`,
    filesets: `/workspaces/:${P.workspace}/filesets`,
    filesetNew: `/workspaces/:${P.workspace}/filesets/new`,
    filesetDetails: `/workspaces/:${P.workspace}/filesets/:${P.filesetId}`,
    filesetFile: `/workspaces/:${P.workspace}/filesets/:${P.filesetId}/file/:${P.filePathEncoded}`,
    /**
     * Unified fileset detail page (gated by VITE_FF_FILESET_DETAILS_ENABLED).
     * One route serves all purposes — the page branches on `fileset.purpose`
     * to render Model/Dataset/Generic card content.
     */
    filesetDetail: `/workspaces/:${P.workspace}/filesets/:${P.filesetName}/detail`,
    inferenceProviders: `/workspaces/:${P.workspace}/inference-providers`,
    virtualModels: `/workspaces/:${P.workspace}/virtual-models`,
    virtualModelDetail: `/workspaces/:${P.workspace}/virtual-models/:${P.virtualModelName}`,
    virtualModelDetails: `/workspaces/:${P.workspace}/virtual-models/:${P.virtualModelName}/details`,
    virtualModelChat: `/workspaces/:${P.workspace}/virtual-models/:${P.virtualModelName}/chat`,
    deploymentConfigs: `/workspaces/:${P.workspace}/deployment-configs`,
    deployments: `/workspaces/:${P.workspace}/deployments`,
    /**
     * Dedicated Create Deployment wizard page.
     *
     * `~new`, not `new`: `~` is outside the entity-name charset (see
     * `resourceRefRegExp`), so this segment can never collide with a deployment
     * name — including if deployments ever get a single-segment detail route.
     */
    deploymentsNew: `/workspaces/:${P.workspace}/deployments/~new`,
    /** Deployments list with details side panel (deployment name + panel segment, e.g. `details`). */
    deploymentsDeployment: `/workspaces/:${P.workspace}/deployments/:${P.deploymentName}/:${P.deploymentPanelView}`,
    intake: `/workspaces/:${P.workspace}/intake`,
    intakeTraces: `/workspaces/:${P.workspace}/intake/traces`,
    intakeSpans: `/workspaces/:${P.workspace}/intake/spans`,
    intakeSession: `/workspaces/:${P.workspace}/intake/sessions/:${P.sessionId}`,
    safeSynthesizer: `/workspaces/:${P.workspace}/safe-synthesizer`,
    safeSynthesizerNew: `/workspaces/:${P.workspace}/safe-synthesizer/new`,
    safeSynthesizerJob: `/workspaces/:${P.workspace}/safe-synthesizer/job/:${P.safeSynthesizerJobName}`,
    safeSynthesizerJobReport: `/workspaces/:${P.workspace}/safe-synthesizer/job/:${P.safeSynthesizerJobName}/report`,
    dataDesignerJobList: `/workspaces/:${P.workspace}/data-designer`,
    dataDesignerJobDetails: `/workspaces/:${P.workspace}/data-designer/:${P.dataDesignerJobName}`,
    dataDesignerJobNew: `/workspaces/:${P.workspace}/data-designer/new`,
    dataDesignerJobBuild: `/workspaces/:${P.workspace}/data-designer/new/build`,
    /** Legacy job-creation form, not linked from any UI — reachable only by typing the URL. */
    dataDesignerJobNewLegacy: `/workspaces/:${P.workspace}/data-designer/new/legacy`,
    anonymizer: `/workspaces/:${P.workspace}/anonymizer`,
    anonymizerNew: `/workspaces/:${P.workspace}/anonymizer/new`,
    anonymizerJob: `/workspaces/:${P.workspace}/anonymizer/:${P.anonymizerJobName}`,
    secrets: `/workspaces/:${P.workspace}/secrets`,
    guardrails: `/workspaces/:${P.workspace}/guardrails`,
    optimizer: `/workspaces/:${P.workspace}/optimizer`,
    optimizerInsight: `/workspaces/:${P.workspace}/optimizer/:${P.insightId}`,
    guardrailDetail: `/workspaces/:${P.workspace}/guardrails/:${P.guardrailConfigName}`,
    guardrailConfig: `/workspaces/:${P.workspace}/guardrails/:${P.guardrailConfigName}/config`,
    guardrailChecks: `/workspaces/:${P.workspace}/guardrails/:${P.guardrailConfigName}/checks`,
    guardrailChecksSubTab: `/workspaces/:${P.workspace}/guardrails/:${P.guardrailConfigName}/checks/:${P.guardrailChecksSubTab}`,
    guardrailVirtualModels: `/workspaces/:${P.workspace}/guardrails/:${P.guardrailConfigName}/virtual-models`,
    settings: `/workspaces/:${P.workspace}/settings`,
    /** Workspace members and role-based access (Entities role bindings) */
    members: `/workspaces/:${P.workspace}/members`,
    agentsList: `/workspaces/:${P.workspace}/agents`,
    assistantChat: `/workspaces/:${P.workspace}/dashboard/assistant`,
    agentDetail: `/workspaces/:${P.workspace}/agents/:${P.agentName}`,
    agentDeploymentsList: `/workspaces/:${P.workspace}/agent-deployments`,
    agentDeploymentDetail: `/workspaces/:${P.workspace}/agent-deployments/:${P.agentDeploymentName}`,
    /** Detail view for a single agent-evaluation job. */
    agentEvaluationDetail: `/workspaces/:${P.workspace}/agents/evaluations/:${P.agentEvalJobName}`,
    modelCompare: `/workspaces/:${P.workspace}/playground`,
    agentMonitor: `/workspaces/:${P.workspace}/agents/monitor`,
    /** Plugin-owned page; the plugin's internal router owns sub-paths via a `/*` suffix. */
    plugin: `/workspaces/:${P.workspace}/plugin/:${P.pluginName}`,
  },
  models: {
    index: '/models',
    modelChat: `/models/:${P.modelNamespace}/:${P.modelName}/chat`,
  },
} as const;
