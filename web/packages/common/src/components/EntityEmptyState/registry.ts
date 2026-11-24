// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EntityKey } from '@nemo/common/src/constants/entityIcons';

/**
 * A create call-to-action for a first-use empty state.
 *
 * Provide `to` for a route-driven create flow; the {@link EntityEmptyState}
 * navigates there on click. For modal-driven creates (no dedicated route),
 * omit `to` and pass `onCreate` at the callsite — the label still comes from
 * here so copy stays centralized.
 */
export interface EmptyStateCreateAction {
  label: string;
  to?: string;
}

/**
 * Per-entity copy and self-service affordances for an empty state. One entry
 * per entity lives in {@link ENTITY_EMPTY_STATES}; callsites never inline this
 * content. The glyph is not here on purpose — it comes from `ENTITY_ICONS`, so
 * an empty state cannot show a different icon than the rest of the entity's
 * surfaces.
 */
export interface EmptyStateDescriptor {
  /** Sentence-case, entity-specific first-use heading. */
  heading: string;
  /** 1–2 sentences answering "why would I create one?". */
  subheading: string;
  /** Omit for entities with no in-app create flow (e.g. Agents, Members). */
  createAction?: EmptyStateCreateAction;
  /** Concrete, copy-pasteable CLI command with `<placeholder>` args. Omit when none exists. */
  cliCommand?: string;
  /** Copy-to-clipboard prompt that triggers the entity's skill. Omit when none exists. */
  skillPrompt?: string;
}

/**
 * Keys of entities that have a standardized empty state. Every {@link EntityKey}
 * now has one, so this is `EntityKey` itself; keying off it is what guarantees
 * every empty state resolves an icon.
 */
export type EmptyStateEntityKey = EntityKey;

export type { EntityKey };

/**
 * Canonical empty-state registry. Grows one entry at a time as entities migrate
 * onto {@link EntityEmptyState}.
 */
export const ENTITY_EMPTY_STATES: Record<EmptyStateEntityKey, EmptyStateDescriptor> = {
  guardrails: {
    heading: 'No guardrail configs yet',
    subheading:
      'Guardrail configs add content-safety, jailbreak, and PII rails to the models in this workspace.',
    // Create is a modal owned by the route, so the callsite supplies `onCreate`.
    createAction: { label: 'Create Guardrail Config' },
    cliCommand: 'nemo guardrail configs create <config-name>',
    skillPrompt: 'Help me create my first guardrail config with the nemo-guardrails skill',
  },
  guardrailChecks: {
    heading: 'No tests yet',
    subheading: 'Add a test case on the Tests tab, then run it to see its result here.',
    cliCommand:
      'nemo guardrail check --model <workspace>/<model> --guardrails \'{"config_id":"<workspace>/<config-name>"}\' --messages \'[{"role":"user","content":"<message>"}]\'',
    skillPrompt: 'Help me verify my guardrail config with the nemo-guardrails skill',
  },
  datasets: {
    heading: 'No datasets yet',
    subheading:
      'Datasets hold the training and evaluation data your models and agents learn from and are measured against.',
    createAction: { label: 'Create Dataset' },
    cliCommand:
      'nemo files filesets create <dataset-name> --purpose dataset --workspace <workspace>',
    skillPrompt: 'Help me create my first dataset with the nemo-files skill',
  },
  filesets: {
    heading: 'No filesets yet',
    subheading:
      'Filesets group the files your agents and jobs read from — training data, models, or other artifacts.',
    createAction: { label: 'Create Model Fileset' },
    cliCommand: 'nemo files filesets create <fileset-name> --workspace <workspace>',
    skillPrompt: 'Help me create my first fileset with the nemo-files skill',
  },
  filesetFiles: {
    heading: 'No files yet',
    subheading: 'Upload files to this fileset to make them available to agents and jobs.',
    createAction: { label: 'Upload Files' },
    cliCommand: 'nemo files upload <local-path> <fileset-name> --workspace <workspace>',
    skillPrompt: 'Help me upload files to a fileset with the nemo-files skill',
  },
  customModels: {
    heading: 'No custom models yet',
    subheading: 'Customize a model with fine-tuning to meet your specific needs.',
    createAction: { label: 'Customize Model' },
    cliCommand: 'nemo customization automodel --job-json <job-spec>.json --workspace <workspace>',
    skillPrompt: 'Help me create my first custom model with the nemo-customizer skill',
  },
  baseModels: {
    heading: 'No base models available',
    subheading:
      'Registered inference providers in this workspace automatically surface their base models here.',
  },
  deployments: {
    heading: 'No deployments yet',
    subheading: 'Deploy a custom model to serve it for inference.',
    createAction: { label: 'Create Deployment' },
    cliCommand: 'nemo inference deployments create <deployment-name> --input-file <config>.json',
    skillPrompt: 'Help me deploy my first model with the nemo-customizer skill',
  },
  inferenceProviders: {
    heading: 'No inference providers yet',
    subheading:
      'Register an inference provider to make its models available for chat and evaluation.',
    createAction: { label: 'Add Inference Provider' },
    cliCommand:
      'nemo inference providers create <provider-name> --workspace <workspace> --host-url "<host-url>" --api-key-secret-name "<secret-name>"',
    skillPrompt: 'Help me create my first inference provider with the nemo-inference skill',
  },
  virtualModels: {
    heading: 'No virtual models yet',
    subheading:
      'Virtual models route inference traffic across one or more providers, with optional switchyard and guardrail middleware.',
    createAction: { label: 'Create Virtual Model' },
    cliCommand:
      'nemo inference virtual-models create <model-name> --workspace <workspace> --models \'[{"model":"<workspace>/<provider-model>","backend_format":"OPENAI_CHAT"}]\'',
    skillPrompt: 'Help me create my first virtual model with the nemo-inference skill',
  },
  secrets: {
    heading: 'No secrets yet',
    subheading:
      'Store API keys and credentials as secrets so providers and jobs can reference them securely.',
    createAction: { label: 'Create Secret' },
    cliCommand:
      'nemo secrets create <secret-name> --value "<secret-value>" --workspace <workspace>',
    skillPrompt: 'Help me create my first secret with the nemo-secrets skill',
  },
  members: {
    heading: 'No members yet',
    subheading:
      'Add a member to grant Viewer, Editor, or Admin access beyond the implicit workspace owners.',
    createAction: { label: 'Add Member' },
    cliCommand:
      'nemo workspaces members create --workspace <workspace-id> --principal <email> --roles <RoleName>',
  },
  jobs: {
    heading: 'No jobs yet',
    subheading:
      'Jobs from customization, evaluation, anonymization, and data generation appear here once submitted.',
  },
  recentJobs: {
    heading: 'No recent active or finished jobs',
    subheading: 'Jobs show here while they run and for 48 hours after they finish.',
    createAction: { label: 'Fine-tune a Model' },
  },
  anonymizerJobs: {
    heading: 'No anonymizer jobs yet',
    subheading:
      'Detect and protect PII in your datasets through context-aware replacement and rewriting.',
    createAction: { label: 'Anonymize Data' },
    cliCommand: 'nemo anonymizer run --spec-file <run-spec>.yaml --workspace <workspace>',
    skillPrompt: 'Help me create my first anonymizer job with the nemo-anonymizer skill',
  },
  dataDesignerJobs: {
    heading: 'No Data Designer jobs yet',
    subheading: 'Create and manage Data Designer jobs to generate or transform synthetic datasets.',
    createAction: { label: 'New Job' },
    cliCommand:
      'nemo data-designer create <config-path>.yaml --num-records <n> --workspace <workspace>',
    skillPrompt:
      'Help me create my first synthetic dataset with the nemo-data-designer-plugin skill',
  },
  // A job writes its own output fileset, so there is nothing for the user to
  // upload or run here — no create action, CLI command, or skill prompt.
  dataDesignerJobOutput: {
    heading: 'No output files yet',
    subheading: 'This job has not written any files, or they are not visible yet.',
  },
  safeSynthesizerJobs: {
    heading: 'No Safe Synthesizer jobs yet',
    subheading: 'Generate a private version of a sensitive tabular dataset.',
    createAction: { label: 'Synthesize Data' },
    cliCommand:
      'nemo safe-synthesizer generate --workspace <workspace> --spec-file <job-spec>.json',
    skillPrompt:
      'Help me create my first Safe Synthesizer job with the nemo-safe-synthesizer skill',
  },
  agentEvaluations: {
    heading: 'No evaluation jobs yet',
    subheading:
      'Apply a model_optimization suggestion or submit an evaluate-agent job to see results here.',
    cliCommand: 'nemo evals agent-evaluate --spec-file <spec>.json --workspace <workspace>',
    skillPrompt: 'Help me create my first agent evaluation with the nemo-evals-plugin skill',
  },
  evaluationResults: {
    heading: 'No evaluations yet',
    subheading: 'Submit an evaluation job to score a model or agent against a benchmark.',
    createAction: { label: 'Create Evaluation' },
    cliCommand: 'nemo evals evaluate --spec-file <spec>.json --workspace <workspace>',
    skillPrompt: 'Help me create my first evaluation with the nemo-evals-plugin skill',
  },
  tasks: {
    heading: 'No tasks yet',
    subheading:
      'A task is one unit of agent work with its grading: a Harbor task or an inline instruction scored by metrics.',
    cliCommand: 'nemo evals tasksets create --harbor-dir <path> --workspace <workspace>',
    skillPrompt: 'Help me publish my first tasks with the nemo-evals-plugin skill',
  },
  tasksets: {
    heading: 'No tasksets yet',
    subheading:
      'A taskset groups stored tasks into a versioned suite you can run an agent evaluation against.',
    cliCommand: 'nemo evals tasksets create --harbor-dir <path> --workspace <workspace>',
    skillPrompt: 'Help me create my first taskset with the nemo-evals-plugin skill',
  },
  evaluationSessions: {
    heading: 'No test cases',
    subheading: 'Run an experiment to see test case results here.',
    // Test-case results are an experiments sub-resource (`experiments/{name}/
    // evaluation-sessions`) with no `nemo` subcommand of its own, so the bridge
    // is the skill rather than a command that does not exist.
    skillPrompt:
      'Help me compare the test case results for an experiment in the "<workspace>" workspace with the nemo-experiments-upload skill, and tell me which run wins and why',
  },
  experiments: {
    heading: 'No experiments yet',
    subheading: 'Log an experiment to compare evaluation runs across models and configurations.',
    createAction: { label: 'Create Experiment' },
    cliCommand: 'nemo experiments create <experiment-name> --input-file <config>.json',
    skillPrompt: 'Help me log my first experiment with the nemo-experiments-upload skill',
  },
  evalComparison: {
    heading: 'No evaluations selected',
    subheading: 'Select evaluations to compare their results side by side.',
  },
  optimizerInsights: {
    heading: 'No insights yet',
    subheading: 'Run an optimizer analysis on an agent to surface insights here.',
    // Creates the workspace's Compass agent first when it has none.
    cliCommand:
      '(nemo agents get compass-analyst --workspace <workspace> || nemo agents create compass.analyst --workspace <workspace>) && nemo agents execute --agent compass-analyst --spec \'{"context": {"agent": "<agent-name>"}}\' --workspace <workspace>',
    skillPrompt: 'Help me run my first analysis of my agent with the nemo-compass skill',
  },
  insightExperiments: {
    heading: 'No experiments yet',
    subheading: 'This insight has no linked experiments yet.',
    cliCommand: 'nemo experiments list --workspace <workspace>',
    skillPrompt:
      'Help me compare the evaluation runs across the experiments in the "<workspace>" workspace with the nemo-experiments-upload skill, and pick a winner',
  },
  insightTraces: {
    heading: 'No traces yet',
    subheading: 'This insight has no linked traces yet.',
    // Quoted for the same reason as the telemetryTraces command: a bare
    // <trace-id> reads as a shell redirection and dies on paste.
    cliCommand: "nemo intake traces get '<trace-id>' --workspace <workspace>",
    skillPrompt:
      'Help me analyze the traces behind this insight in the "<workspace>" workspace with the nemo-compass skill',
  },
  telemetryTraces: {
    heading: 'No traces yet',
    subheading: 'Trace summaries will appear here after spans are ingested.',
    cliCommand: 'nemo intake ingest spans create --input-file <spans>.json --workspace <workspace>',
    skillPrompt:
      'Help me import traces into the "<workspace>" workspace with the nemo-intake skill.',
  },
  telemetrySpans: {
    heading: 'No spans yet',
    subheading: 'Spans will appear here once your agent starts sending telemetry.',
    cliCommand: "nemo intake spans list --filter.trace-id '<trace-id>' --workspace <workspace>",
    skillPrompt:
      'Help me analyze the spans in trace "<trace-id>" in the "<workspace>" workspace with the nemo-intake skill — where did this run go wrong?',
  },
  agentMonitorRuns: {
    heading: 'No runs yet',
    subheading:
      'Agent invocations populate this list once telemetry reaches the nemo-agent-telemetry fileset.',
    cliCommand: "nemo intake spans list --filter.agent-name '<agent-name>' --workspace <workspace>",
    skillPrompt:
      'Help me find recurring failures in the recent runs of agent "<agent-name>" in the "<workspace>" workspace with the nemo-compass skill',
  },
  agentOptimizations: {
    heading: 'No optimizations yet',
    subheading:
      "Sweep your agent's parameters against an evaluation to find a config that scores better.",
    cliCommand:
      'nemo agents optimize run-strategy --strategy <strategy> --optimize-config-fileset <workspace>/<bundle> --optimize-config <config>.yaml --agent <agent-name> --workspace <workspace>',
    skillPrompt: 'Help me run my first agent optimization with the nemo-agents-optimize skill',
    createAction: { label: 'Optimize' },
  },
  agents: {
    heading: 'No agents yet',
    subheading: 'Build and deploy an agent to see it listed here.',
    cliCommand:
      'nemo agents register --name <agent-name> --agent-config <agent.yaml> --workspace <workspace>',
    skillPrompt: 'Help me create my first agent with the nemo-build-agent skill',
  },
  optimizationTrials: {
    heading: 'No trials',
    subheading:
      'Each trial is one parameter set the study scored. They appear here as the study runs.',
    // A trial is not created directly — it is produced by a study — so the
    // bridge re-runs the study rather than offering a create command.
    cliCommand:
      'nemo agents optimize run-strategy --strategy <strategy> --optimize-config-fileset <workspace>/<bundle> --optimize-config <config>.yaml --agent <agent-name> --workspace <workspace>',
    skillPrompt:
      'Help me understand why this optimization study recorded no trials, and re-run it, with the nemo-agents-optimize skill',
  },

  // NeMo Assistant surfaces. These are in-app chat state, not platform
  // entities, so they deliberately carry no `cliCommand` or `skillPrompt`:
  // there is nothing a coding agent could run to populate them.
  assistantSessions: {
    heading: 'No chats yet',
    subheading: 'NeMo Assistant sessions appear here once you start a conversation.',
  },
  assistantArtifacts: {
    heading: 'No artifacts yet',
    subheading: 'Selections and outputs from this conversation appear here.',
  },
  assistantSkills: {
    heading: 'No skills found',
    subheading: 'NeMo Assistant skills appear here once they are available in this workspace.',
  },
};
