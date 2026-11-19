// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Mirrors the `nemo-distillation` plugin's models, served at `/apis/distillation`. */

export type WorkflowStatus =
  | 'pending'
  | 'running'
  | 'cancelling'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type NodeStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped' | 'cancelled';

export type StageStatus = 'pending' | 'active' | 'completed' | 'failed';

export interface GraphNode {
  kind: string;
  inputs: Record<string, unknown>;
  depends_on: string[];
  always_run: boolean;
}

export interface RuntimeNode {
  status: NodeStatus;
  attempt: number;
  external: Record<string, string>;
  outputs: Record<string, unknown>;
  error: string | null;
}

export interface WorkflowStage {
  id: string;
  name: string;
  status: StageStatus;
  progress: number;
  metrics: Record<string, unknown>;
  job_ref: string | null;
}

export interface WorkflowArm {
  score: number | null;
  input_tokens_per_task: number | null;
  output_tokens_per_task: number | null;
  median_latency_sec: number | null;
  estimated_cost_per_1k_tasks: number | null;
  eval_job: string | null;
  filesets: (string | null)[];
}

export interface WorkflowResults {
  original: WorkflowArm | null;
  base: WorkflowArm;
  distilled: WorkflowArm;
  prices: Record<string, unknown>;
}

export interface DistillationWorkflow {
  name: string;
  workspace: string;
  created_at?: string;
  template: string;
  status: WorkflowStatus;
  source_agent: string;
  inputs: Partial<DistillInputs>;
  fileset: string;
  graph: Record<string, GraphNode>;
  nodes: Record<string, RuntimeNode>;
  experiment: string | null;
  stages: WorkflowStage[];
  results: WorkflowResults | null;
}

export interface DeploymentInput {
  engine?: 'vllm' | 'nim';
  gpu?: number;
  tool_call_config?: Record<string, unknown> | null;
  additional_args?: string[];
}

export interface TeacherInput extends DeploymentInput {
  model: string;
}

export interface AgentDeployInput {
  mode?: 'subprocess' | 'docker' | 'k8s' | 'openshell';
  image?: string;
  use_image_entrypoint?: boolean;
}

export interface EvalInput {
  name: string;
  kind: 'agent' | 'dataset';
  config: string;
  dataset?: string | null;
}

export interface Hyperparameters {
  max_seq_length?: number;
  lora?: {
    rank?: number;
    alpha?: number;
    dropout?: number;
    target_modules?: string[];
    exclude_modules?: string[];
  };
  schedule?: {
    epochs?: number;
    max_steps?: number;
    validation_split?: number;
    checkpoint_selection?: 'best' | 'last' | 'both';
    seed?: number;
  };
  batch?: { global_batch_size?: number; micro_batch_size?: number; sequence_packing?: boolean };
  optimizer?: {
    learning_rate?: number;
    min_learning_rate?: number;
    weight_decay?: number;
    warmup_steps?: number;
    lr_decay_style?: 'cosine' | 'linear' | 'constant';
  };
}

export interface CandidateSpec {
  name: string;
  finetuning_type?: 'lora' | 'all_weights';
  gpu?: number;
  hyperparameters?: Hyperparameters;
}

/** Give either `teacher` (the workflow runs the original evals) or `trajectory_filesets`, not both. */
export interface DistillInputs {
  source_agent: string;
  student_model: string;
  teacher?: TeacherInput | null;
  trajectory_filesets?: string[];
  gpu_budget?: number;
  agent_deployment: AgentDeployInput;
  evals: EvalInput[];
  eval_results?: string[];
  pass_filter?: Record<string, unknown>;
  candidates?: CandidateSpec[];
  deployment?: DeploymentInput;
  prices?: Record<string, unknown>;
}

export interface CreateWorkflowBody {
  name?: string;
  inputs: DistillInputs;
}

export interface RenderCounts {
  nodes: number;
  candidates: number;
  passing_trials: number;
  estimated_tokens: number;
  gpu_hours: number;
  parallel_jobs: number;
}

export interface RenderResponse {
  graph: Record<string, GraphNode>;
  counts: RenderCounts;
}

export interface WorkflowPage {
  data: DistillationWorkflow[];
  pagination: {
    page: number;
    page_size: number;
    current_page_size: number;
    total_pages: number;
    total_results: number;
  };
}
