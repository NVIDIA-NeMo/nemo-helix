// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FILESET_NAME_MAX_LENGTH, toValidFilesetName } from '@nemo/common/src/utils/filesetName';
import { generateDefaultName } from '@nemo/common/src/utils/generateDefaultName';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { parse as parseYaml, stringify as stringifyYaml } from 'yaml';

/** Sentinel ``evalConfig`` value that switches the form into create mode. */
export const CREATE_NEW = '__create_new__';

export const MODE_DEFAULT = 'default';
/** Re-run an existing Experiment, which owns the fileset holding its eval config. */
export const MODE_EXPERIMENT = 'experiment';

/** Suggested name for a new experiment (e.g. "wise-blue"). */
export const generateEvalConfigName = (): string => generateDefaultName({ length: 2 });

/** The fileset that stores an experiment's eval config and data artifacts. */
export const filesetNameForExperiment = (experimentName: string): string =>
  `${experimentName}-data`;

/** Default parallelism for a submitted eval (Studio default; the config value is a hint). */
export const DEFAULT_MAX_CONCURRENT_TASKS = 1;

/** ``RunConfigOnline`` for an agent target, shared by both submit paths.
 *
 *  ``ignore_request_failure`` is what keeps one bad row from aborting the job: NAT
 *  reports workflow failures (including output truncation) as 422, which is not
 *  retried, so a failed row degrades to NaN instead of killing the run. That is the
 *  safety property — it does not depend on running serially.
 *
 *  ``parallelism`` is a wall-clock knob only. Rows are independent, so running them
 *  concurrently changes no score. It matters because an agent row's cost is dominated
 *  by the model's hidden reasoning tokens, which can dwarf the visible answer, so
 *  submitting serially leaves the endpoint idle between rows.
 *
 *  ``max_retries`` is paired with it deliberately. A single-process deployment starts
 *  returning 502s under sustained load, and 502 is classified transient by the SDK
 *  (``resilience/classifier.py``), so retries absorb it at the source. Without the extra
 *  attempts, concurrency would convert overload into NaN rows via
 *  ``ignore_request_failure`` and the job would still report success. Raise parallelism
 *  further only alongside retries, and only if the endpoint keeps up. */
const AGENT_RUN_PARAMS = {
  parallelism: 4,
  request_timeout: 300,
  max_retries: 5,
  ignore_request_failure: true,
} as const;

const JOB_FILESET_PREFIX = 'job-fileset-';
export const JOB_NAME_MAX_LENGTH = FILESET_NAME_MAX_LENGTH - JOB_FILESET_PREFIX.length;

export const buildEvalJobName = (filesetName: string): string => {
  const suffix = Math.random().toString(36).slice(2, 10).padEnd(8, '0');
  const base = toValidFilesetName(filesetName)
    .slice(0, JOB_NAME_MAX_LENGTH - suffix.length - 1)
    .replace(/-+$/, '');
  return `${base}-${suffix}`;
};

// ---------------------------------------------------------------------------
// eval-config.json shape (stored in a fileset, read at submit)
// ---------------------------------------------------------------------------

/** One inline metric bundle as stored in eval-config.json. Kept loose: Studio
 *  does not re-validate the built-in metric shape, it only overrides the judge
 *  model on metrics that name one. */
export interface InlineMetricBundle {
  bundle_kind: string;
  bundle_format_version: string;
  metric_type: string;
  metadata?: Record<string, unknown>;
  outputs?: unknown[];
  secrets?: Record<string, unknown>;
  payload: {
    kind: 'inline';
    metric: Record<string, unknown> & { model?: unknown };
  };
}

/** A task as authored in eval-config.json: carries its own metrics. */
export interface EvalSpecTask {
  id: string;
  intent: string;
  inputs?: { instruction?: string | null };
  reference?: Record<string, unknown>;
  metrics: InlineMetricBundle[];
  views?: Record<string, unknown>;
}

/** The eval config, both as authored under public/sample-agents and as stored in
 *  a fileset: an `AgentEvalInputSpec` minus `target` (target is per-run). */
export interface PersistedEvalSpec {
  tasks: EvalSpecTask[];
  max_concurrent_tasks?: number;
}

/** A dataset-driven config: an `EvaluateInputSpec` minus `target`. One metric set
 *  scores every row of a dataset, instead of each task carrying its own. */
export interface DatasetEvalSpec {
  dataset: string | Record<string, unknown>[];
  metrics: InlineMetricBundle[];
  prompt_template?: string | Record<string, unknown>;
  field_mapping?: Record<string, unknown> | null;
}

export type EvalSpec = PersistedEvalSpec | DatasetEvalSpec;

/** Which evaluator endpoint a config targets, decided by its shape alone: a
 *  `dataset` + `metrics` pair means row-based `evaluate/jobs`, `tasks[]` means
 *  task-based `agent-evaluate/jobs`. No extra registry metadata to keep in sync. */
export const isDatasetEvalSpec = (spec: EvalSpec): spec is DatasetEvalSpec =>
  'dataset' in spec && Array.isArray((spec as DatasetEvalSpec).metrics);

// ---------------------------------------------------------------------------
// Submit-time selections + request assembly
// ---------------------------------------------------------------------------

export interface SubmitSelections {
  workspace: string;
  /** Agent (bare name) to evaluate; used to build the generic target. */
  agent: string;
  /** Eval-config fileset name, stored under spec.labels.eval_config_fileset for display. */
  filesetName?: string;
  /** Experiment this run belongs to; names the job so it reads as one of that experiment's runs. */
  experimentName?: string;
  /** Name of an existing Intake Evaluation to publish results under. The job fails if it
   *  names nothing — the worker never creates it. Omitted means the run publishes nowhere. */
  evaluationId?: string;
}

/** ``spec.publication`` for a run that asked to publish, or nothing at all. ``agent_name`` is
 *  left off deliberately: the backend derives it from the agent target. */
const publicationSpec = (evaluationId: string | undefined) =>
  evaluationId ? { publication: { intake: { evaluation_id: evaluationId } } } : {};

/** ``{ name }`` for the job, stemmed from the experiment it belongs to and falling back to the
 *  fileset for a submit that names no experiment. Absent when neither is known. */
const jobName = (selections: SubmitSelections) => {
  const stem = selections.experimentName ?? selections.filesetName;
  return stem ? { name: buildEvalJobName(stem) } : {};
};

/** Strip an optional ``workspace/`` prefix, returning the bare model/agent name. */
export const bareName = (value: string): string =>
  value.includes('/') ? (value.split('/').pop() ?? value) : value;

/** Chat completions is the one endpoint both config formats serve, so this needs no branch. */
const agentEndpoint = (workspace: string, agent: string, promptVar: string) => ({
  format: 'generic' as const,
  url: `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/${encodeURIComponent(workspace)}/agents/${encodeURIComponent(bareName(agent))}/-/v1/chat/completions`,
  name: bareName(agent),
  body: {
    model: bareName(agent),
    messages: [{ role: 'user', content: `{{ ${promptVar} }}` }],
    stream: false,
  },
  response_path: '$.choices[0].message.content',
  stream: false,
});

export const buildAgentTarget = (workspace: string, agent: string) => ({
  kind: 'agent' as const,
  agent: agentEndpoint(workspace, agent, 'instruction'),
  params: AGENT_RUN_PARAMS,
});

/** Override a metric's judge model with a ``workspace/name`` ModelRef (resolved
 *  to a reachable Model server-side). Metrics that name no model — string-check,
 *  exact-match, f1 — are returned untouched. Does not mutate input. */
export const injectJudgeModel = (
  metric: InlineMetricBundle,
  judgeModel: string
): InlineMetricBundle =>
  metric.payload?.metric && (metric.metric_type === 'llm-judge' || 'model' in metric.payload.metric)
    ? {
        ...metric,
        payload: {
          ...metric.payload,
          metric: { ...metric.payload.metric, model: judgeModel },
        },
      }
    : metric;

/** Apply the picked judge model across every task's metrics. With no judge
 *  supplied the config's own model strings are used as authored. */
export const applyJudgeModel = (
  tasks: EvalSpecTask[],
  judgeModel: string | null
): EvalSpecTask[] =>
  judgeModel
    ? tasks.map((task) => ({
        ...task,
        metrics: task.metrics.map((metric) => injectJudgeModel(metric, judgeModel)),
      }))
    : tasks;

/** Build the spec persisted to a fileset: the config as authored, with the
 *  picked judge model applied. */
export const buildPersistedSpec = (
  config: PersistedEvalSpec,
  judgeModel: string | null
): PersistedEvalSpec => ({
  tasks: applyJudgeModel(config.tasks, judgeModel),
  max_concurrent_tasks: config.max_concurrent_tasks ?? DEFAULT_MAX_CONCURRENT_TASKS,
});

/** Build the ``agent-evaluate/jobs`` POST body from a persisted spec + selections. */
export const buildAgentEvalRequestBody = (
  spec: PersistedEvalSpec,
  selections: SubmitSelections
) => ({
  ...jobName(selections),
  spec: {
    tasks: spec.tasks,
    target: buildAgentTarget(selections.workspace, selections.agent),
    max_concurrent_tasks: spec.max_concurrent_tasks ?? DEFAULT_MAX_CONCURRENT_TASKS,
    ...(selections.filesetName ? { labels: { eval_config_fileset: selections.filesetName } } : {}),
    ...publicationSpec(selections.evaluationId),
  },
});

/** The bare agent target for ``evaluate/jobs``. Unlike the agent-evaluate target
 *  this is NOT wrapped in {kind, agent}: EvaluateInputSpec forbids extra keys and
 *  takes the agent object directly. The body renders the row-based ``prompt``
 *  rather than a task ``instruction``. */
export const buildDatasetAgentTarget = (workspace: string, agent: string) =>
  agentEndpoint(workspace, agent, 'prompt');

/** Build the ``evaluate/jobs`` POST body from a dataset-driven config. ``params``
 *  must be exactly RunConfigOnline for an agent target, and ``prompt_template``
 *  is required — the job 500s without both. */
export const buildDatasetEvalRequestBody = (
  spec: DatasetEvalSpec,
  selections: SubmitSelections,
  judgeModel: string | null
) => ({
  ...jobName(selections),
  spec: {
    dataset: spec.dataset,
    metrics: judgeModel
      ? spec.metrics.map((metric) => injectJudgeModel(metric, judgeModel))
      : spec.metrics,
    target: buildDatasetAgentTarget(selections.workspace, selections.agent),
    prompt_template: spec.prompt_template,
    ...(spec.field_mapping ? { field_mapping: spec.field_mapping } : {}),
    params: AGENT_RUN_PARAMS,
    ...publicationSpec(selections.evaluationId),
  },
});

/** The serialization an eval config is stored in. Both are accepted on upload and read back
 *  on reuse; a run keeps whichever the author uploaded. */
export type EvalConfigFormat = 'json' | 'yaml';

/** Serialize a spec for storage in a fileset, in the format its author uploaded. */
export const serializeEvalConfig = (spec: EvalSpec, format: EvalConfigFormat): string =>
  format === 'yaml' ? stringifyYaml(spec) : JSON.stringify(spec, null, 2);

/** Reject metric entries that cannot be read as a metric bundle. Studio does not validate the
 *  metric body, but it does reach into ``payload.metric`` to decide whether a judge applies —
 *  so an entry without one would throw at render, past every error boundary this form has. */
/** The evaluator recomputes an llm-judge's ``output_spec()`` from ``payload.metric.scores`` and
 *  rejects the job if it disagrees with the declared ``outputs`` — but only after the run's
 *  fileset already exists, so the mismatch is caught here instead. llm-judge only: other metric
 *  types derive their outputs by different rules. */
const assertJudgeOutputsMatchScores = (metric: InlineMetricBundle, where: string): void => {
  if (metric.metric_type !== 'llm-judge') return;
  const scores = metric.payload.metric.scores;
  if (!Array.isArray(scores) || !Array.isArray(metric.outputs)) return;

  const scoreNames = scores
    .map((score) => (score as { name?: unknown })?.name)
    .filter((name): name is string => typeof name === 'string');
  const declared = new Set(
    metric.outputs
      .map((output) => (output as { name?: unknown })?.name)
      .filter((name): name is string => typeof name === 'string')
  );

  const missing = scoreNames.filter((name) => !declared.has(name));
  if (missing.length > 0) {
    throw new Error(
      `${where} has an llm-judge metric whose "outputs" do not list its score${missing.length > 1 ? 's' : ''} ${missing.map((name) => `"${name}"`).join(', ')}. Every entry in "scores" needs a matching entry in "outputs".`
    );
  }
};

const assertMetricBundles = (metrics: InlineMetricBundle[], where: string): void => {
  for (const metric of metrics) {
    if (!metric?.payload?.metric || typeof metric.payload.metric !== 'object') {
      throw new Error(`${where} has a metric without a "payload.metric" object`);
    }
    assertJudgeOutputsMatchScores(metric, where);
  }
};

/** Validate a parsed eval config. Every task must carry its own ``metrics[]`` —
 *  the config is submitted as authored, with only the judge model and the
 *  per-run agent target applied on top. */
const validateEvalConfig = (raw: Partial<PersistedEvalSpec & DatasetEvalSpec>): EvalSpec => {
  if (raw.dataset !== undefined) {
    if (!Array.isArray(raw.metrics) || raw.metrics.length === 0) {
      throw new Error('a dataset-driven eval-config.json must contain a non-empty "metrics" array');
    }
    if (!raw.prompt_template) {
      throw new Error('a dataset-driven eval-config.json must contain a "prompt_template"');
    }
    assertMetricBundles(raw.metrics, 'a dataset-driven eval-config.json');
    return {
      dataset: raw.dataset,
      metrics: raw.metrics,
      prompt_template: raw.prompt_template,
      field_mapping: raw.field_mapping ?? null,
    };
  }

  const parsed = raw;
  if (!Array.isArray(parsed.tasks) || parsed.tasks.length === 0) {
    throw new Error('eval-config.json must contain "tasks" or a "dataset"');
  }
  for (const task of parsed.tasks) {
    if (!Array.isArray(task.metrics) || task.metrics.length === 0) {
      throw new Error(
        `eval-config.json task "${task.id}" must contain a non-empty "metrics" array`
      );
    }
    assertMetricBundles(task.metrics, `eval-config.json task "${task.id}"`);
  }
  return { tasks: parsed.tasks, max_concurrent_tasks: parsed.max_concurrent_tasks };
};

/** Parse a stored eval config, in either serialization. */
export const parseEvalConfig = (text: string): EvalSpec =>
  validateEvalConfig(parseConfigDocument(text));

/** Read a config as JSON or YAML. JSON is tried first even though YAML 1.2 is a
 *  superset of it: a JSON document indented with tabs is legal JSON but illegal YAML, and
 *  a JSON file should fail with a JSON parser's message rather than a YAML one. */
const parseConfigDocument = (text: string): Partial<PersistedEvalSpec & DatasetEvalSpec> => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (jsonErr) {
    try {
      parsed = parseYaml(text);
    } catch {
      throw jsonErr;
    }
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('An eval config must be a JSON or YAML object.');
  }
  return parsed as Partial<PersistedEvalSpec & DatasetEvalSpec>;
};

/** Parse a user-uploaded eval config for the "create experiment" flow, which only runs
 *  dataset-driven configs. The dataset arrives as a separate upload and its ref is written in
 *  at submit, so a config that omits ``dataset`` entirely is accepted and filled in here —
 *  otherwise an author would have to hand-write a placeholder that is immediately discarded. */
export const parseUploadedDatasetConfig = (text: string): DatasetEvalSpec => {
  const raw = parseConfigDocument(text);
  const spec = validateEvalConfig(
    raw.tasks === undefined && raw.dataset === undefined ? { ...raw, dataset: '' } : raw
  );
  if (!isDatasetEvalSpec(spec)) {
    throw new Error(
      'This eval config is task-driven. Uploading a config requires the dataset-driven form: a "prompt_template" and a non-empty "metrics" array scored over the uploaded dataset.'
    );
  }
  return spec;
};
