// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FeatureFlags } from '@studio/constants/featureFlags/featureFlags';
import {
  BarChart3,
  Database,
  Gauge,
  GitBranch,
  Hammer,
  KeyRound,
  Search,
  SearchCheck,
  ShieldCheck,
  Sliders,
  Sparkles,
  Terminal,
} from 'lucide-react';
import type { ReactNode } from 'react';

export interface SkillActionTemplate {
  title: string;
  description: string;
  prompt: string;
  icon: ReactNode;
  requiredFeatureFlags?: readonly FeatureFlagName[];
}

export interface SkillActionSuggestion extends SkillActionTemplate {
  skillName: string;
  claudeName: string;
}

type FeatureFlagName = keyof FeatureFlags;

export const SKILL_ACTION_TEMPLATES = {
  'agents-optimize': {
    title: 'Optimize an agent',
    description: 'Find routing, prompt, model, and skill changes that improve cost or quality.',
    prompt:
      'Use the agents-optimize skill to inspect a deployed NeMo agent and recommend changes that improve cost, latency, or quality.',
    icon: <Gauge size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'agents-secure': {
    title: 'Harden an agent',
    description: 'Audit safety, PII, guardrails, and secret exposure risks.',
    prompt:
      'Use the agents-secure skill to audit a deployed NeMo agent for safety, PII exposure, missing guardrails, and leaked secrets.',
    icon: <SearchCheck size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  anonymizer: {
    title: 'Anonymize a dataset',
    description: 'Detect and replace PII in CSV or Parquet data.',
    prompt:
      'Use the anonymizer skill to detect and anonymize PII in a dataset. Inspect the available files first and recommend a replacement strategy.',
    icon: <Database size={18} />,
    requiredFeatureFlags: ['datasetsEnabled'],
  },
  auditor: {
    title: 'Run a security audit',
    description: 'Scan an agent target for vulnerabilities and risky behavior.',
    prompt:
      'Use the auditor skill to configure and run a security audit for a NeMo Platform agent target.',
    icon: <SearchCheck size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'guardrails-plugin': {
    title: 'Debug guardrails middleware',
    description: 'Create, attach, and verify guardrail configs through inference middleware.',
    prompt:
      'Use the guardrails-plugin skill to create, attach, or debug guardrail middleware for a NeMo inference path.',
    icon: <ShieldCheck size={18} />,
    requiredFeatureFlags: ['guardrailsEnabled'],
  },
  inference: {
    title: 'Configure inference',
    description: 'Register providers, virtual models, routing, and translation middleware.',
    prompt:
      'Use the inference skill to configure a NeMo inference provider, virtual model, or middleware route in this workspace.',
    icon: <GitBranch size={18} />,
    requiredFeatureFlags: ['inferenceProviderEnabled'],
  },
  'nemo-build-agent': {
    title: 'Build an agent',
    description: 'Scaffold and deploy a NAT workflow from an agent Ethos.',
    prompt:
      'Use the nemo-build-agent skill to scaffold and deploy a NeMo agent from an existing Ethos. Inspect the workspace first and ask for the target Ethos if needed.',
    icon: <Hammer size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'nemo-customizer': {
    title: 'Fine-tune a model',
    description: 'Train a custom model on your own dataset.',
    prompt:
      'Use the nemo-customizer skill to fine-tune a model. Help me choose a base model, dataset, and training configuration, then launch and monitor the customization job.',
    icon: <Sliders size={18} />,
    requiredFeatureFlags: ['customizerEnabled'],
  },
  'nemo-data-designer-plugin': {
    title: 'Generate synthetic data',
    description: 'Create a dataset or data generation workflow.',
    prompt:
      'Use the nemo-data-designer-plugin skill to create a synthetic dataset for this workspace. Ask for any missing dataset requirements before generating files.',
    icon: <Sparkles size={18} />,
    requiredFeatureFlags: ['dataDesignerEnabled'],
  },
  'nemo-eval-history': {
    title: 'Review eval history',
    description: 'Inspect previous evaluation runs and compare outcomes.',
    prompt:
      'Use the nemo-eval-history skill to review previous evaluation runs and summarize the most important changes or failures.',
    icon: <BarChart3 size={18} />,
    requiredFeatureFlags: ['evaluatorEnabled'],
  },
  'nemo-evaluator': {
    title: 'Run model evaluations',
    description: 'Create metrics or benchmarks and inspect results.',
    prompt:
      'Use the nemo-evaluator skill to create or run an evaluation, then summarize the metric results and any follow-up recommendations.',
    icon: <BarChart3 size={18} />,
    requiredFeatureFlags: ['evaluatorEnabled'],
  },
  'nemo-evaluator-plugin': {
    title: 'Use evaluator plugin',
    description: 'Work with evaluator jobs, SDK specs, and plugin-owned skills.',
    prompt:
      'Use the nemo-evaluator-plugin skill to inspect or update evaluator plugin jobs, SDK specs, or plugin-owned evaluator skills.',
    icon: <BarChart3 size={18} />,
    requiredFeatureFlags: ['evaluatorEnabled'],
  },
  'nemo-explore': {
    title: 'Explore an agent idea',
    description: 'Capture the job, audience, tools, model, and constraints.',
    prompt:
      'Use the nemo-explore skill to guide an agent design conversation and capture the important decisions before writing a spec.',
    icon: <Search size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'nemo-files': {
    title: 'Manage filesets',
    description: 'Upload, download, and inspect datasets or JSONL artifacts.',
    prompt:
      'Use the nemo-files skill to inspect filesets and help upload, download, or manage dataset artifacts for this workspace.',
    icon: <Database size={18} />,
    requiredFeatureFlags: ['datasetsEnabled'],
  },
  'nemo-fine-tune': {
    title: 'Check fine-tuning status',
    description: 'Confirm what fine-tuning path is currently available.',
    prompt:
      'Use the nemo-fine-tune skill to explain the current fine-tuning status for NeMo Platform and avoid unsupported training paths.',
    icon: <Gauge size={18} />,
    requiredFeatureFlags: ['customizerEnabled'],
  },
  'nemo-guardrails': {
    title: 'Add guardrails to an agent',
    description: 'Configure content safety and apply rails to inference.',
    prompt:
      'Use the nemo-guardrails skill to add input and output guardrails to an agent in this workspace. Start by inspecting what exists, then propose and implement the safest path.',
    icon: <ShieldCheck size={18} />,
    requiredFeatureFlags: ['guardrailsEnabled'],
  },
  'nemo-model-selection': {
    title: 'Choose a model',
    description: 'Compare model options for an agent or workflow.',
    prompt:
      'Use the nemo-model-selection skill to compare model options for a NeMo Platform agent or workflow and recommend a starting point.',
    icon: <SearchCheck size={18} />,
    requiredFeatureFlags: ['baseModelsEnabled'],
  },
  'nemo-secrets': {
    title: 'Manage secrets',
    description: 'Create, list, or update credentials for platform workflows.',
    prompt:
      'Use the nemo-secrets skill to help manage credentials for this workspace. Start by checking what secret operation is needed.',
    icon: <KeyRound size={18} />,
    requiredFeatureFlags: ['secretsEnabled'],
  },
  'nemo-skill-selection': {
    title: 'Pick the right NeMo skill',
    description: 'Route a broad task to the right specialized workflow.',
    prompt:
      'Use the nemo-skill-selection skill to route this NeMo Platform task to the right specialized skill before taking action.',
    icon: <GitBranch size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'nemo-ethos': {
    title: 'Write an agent Ethos',
    description: 'Turn exploration notes into a durable agent contract.',
    prompt:
      'Use the nemo-ethos skill to turn the current agent design notes into a durable NeMo Platform agent Ethos.',
    icon: <Hammer size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'nemo-status': {
    title: 'Check platform status',
    description: 'Summarize health, providers, deployed agents, and available models.',
    prompt:
      'Use the nemo-status skill to check NeMo Platform health, deployed agents, providers, and available models.',
    icon: <Terminal size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'nemo-teardown': {
    title: 'Shut down platform',
    description: 'Choose a safe stop or cleanup path for local services.',
    prompt:
      'Use the nemo-teardown skill to guide a safe NeMo Platform shutdown or cleanup. Confirm before any destructive action.',
    icon: <Terminal size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'nemo-try-agent': {
    title: 'Try a deployed agent',
    description: 'Send a query to an agent or fall back to model chat.',
    prompt:
      'Use the nemo-try-agent skill to send a query to a deployed NeMo Platform agent, announcing the routing decision before sending.',
    icon: <Terminal size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
  'safe-synthesizer': {
    title: 'Generate safety data',
    description: 'Create safety-focused synthetic data with Safe Synthesizer.',
    prompt:
      'Use the safe-synthesizer skill to create safety-focused synthetic data for a NeMo Platform workflow.',
    icon: <Sparkles size={18} />,
    requiredFeatureFlags: ['safeSynthesizerEnabled'],
  },
  'skills-optimization': {
    title: 'Optimize agent skills',
    description: 'Run skill evaluation and improvement loops.',
    prompt:
      'Use the skills-optimization skill to evaluate and improve an agent skill suite, then summarize the recommended changes.',
    icon: <Gauge size={18} />,
    requiredFeatureFlags: ['agentsEnabled'],
  },
} satisfies Record<string, SkillActionTemplate>;

export type SkillActionTemplateName = keyof typeof SKILL_ACTION_TEMPLATES;
