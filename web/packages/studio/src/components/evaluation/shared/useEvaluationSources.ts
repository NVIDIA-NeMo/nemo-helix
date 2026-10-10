// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SelectItemOption } from '@nemo/common/src/components/form/ControlledSearchableSelect';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { evaluationFilesetName } from '@studio/components/evaluation/experimentEvalConfig';
import { useMemo } from 'react';

/** API maximum. The picker still shows one page, so an agent past this many is truncated. */
export const LIST_PAGE_SIZE = 1000;

/** Group key for an evaluation whose experiment fell outside the fetched page, or that somehow
 *  belongs to none. It still has a reusable config, so it stays selectable rather than vanishing. */
const UNGROUPED = '__ungrouped__';

const UNGROUPED_LABEL = 'Other evaluations';

/** One reusable evaluation, resolved to the experiment it is grouped under. */
export interface EvaluationSource {
  evaluation: EvaluationResponse;
  experiment?: ExperimentResponse;
  /** The experiment's name, or undefined when it could not be resolved. */
  experimentName?: string;
  /** No run of this evaluation is tagged with the agent the list is scoped to. */
  fromOtherAgent: boolean;
}

export interface UseEvaluationSourcesResult {
  /** Every evaluation in view that carries a reusable eval config, newest first. */
  sources: EvaluationSource[];
  /** Options for a single grouped picker: sections are experiments, items are their evaluations. */
  options: SelectItemOption[];
  /** Section headings, keyed by the group each option carries. */
  groupLabels: Record<string, string>;
  /** Every reusable evaluation in the workspace, in view or not, so a pick always resolves. */
  byName: Record<string, EvaluationSource>;
  /** Every experiment loaded, for a "which experiment" filter or lookup. */
  experiments: ExperimentResponse[];
  isLoading: boolean;
  /** True once loading has settled and nothing reusable is in view. */
  isEmpty: boolean;
  hasAgentSources: boolean;
  hasOtherAgentSources: boolean;
}

interface UseEvaluationSourcesParams {
  workspace: string;
  /** Scope the list to one agent's evaluations. Omitted lists the whole workspace. */
  agent?: string;
  /** Also list evaluations of other agents, so a run can join another agent's experiment. */
  includeOtherAgents?: boolean;
  enabled?: boolean;
}

/**
 * The reusable evaluations a new run can be based on, grouped by the experiment they belong to.
 *
 * The eval config is identified on each Evaluation by Studio convention (`metadata`), not by the
 * API or CLI — so an evaluation without that pointer cannot be re-run and is filtered out here
 * rather than offered and then rejected at submit.
 */
export const useEvaluationSources = ({
  workspace,
  agent,
  includeOtherAgents = false,
  enabled = true,
}: UseEvaluationSourcesParams): UseEvaluationSourcesResult => {
  // Scoped client-side, so one request also answers whether other agents have runs to offer.
  const { data: evaluationsResponse, isLoading: isEvaluationsLoading } = useListEvaluations(
    workspace,
    { page_size: LIST_PAGE_SIZE, sort: '-created_at' },
    { query: { enabled } }
  );

  const { data: experimentsResponse, isLoading: isExperimentsLoading } = useListExperiments(
    workspace,
    { page_size: LIST_PAGE_SIZE, sort: '-created_at' },
    { query: { enabled } }
  );

  const evaluations = useMemo(() => evaluationsResponse?.data ?? [], [evaluationsResponse]);
  const experiments = useMemo(() => experimentsResponse?.data ?? [], [experimentsResponse]);

  return useMemo(() => {
    const experimentById = new Map(experiments.map((experiment) => [experiment.id, experiment]));

    // agent_names is denormalized from ingested span telemetry, so an evaluation only counts as
    // this agent's once it has runs tagged with it.
    const allSources: EvaluationSource[] = evaluations
      .filter((evaluation) => evaluationFilesetName(evaluation) != null)
      .map((evaluation) => {
        // An evaluation can belong to several experiments; the first resolvable one names its
        // section, matching how its detail route is nested.
        const experiment = evaluation.experiment_ids
          .map((id) => experimentById.get(id))
          .find((candidate): candidate is ExperimentResponse => candidate !== undefined);
        return {
          evaluation,
          experiment,
          experimentName: experiment?.name,
          fromOtherAgent: !!agent && !(evaluation.agent_names ?? []).includes(agent),
        };
      });
    const sources = includeOtherAgents
      ? allSources
      : allSources.filter((source) => !source.fromOtherAgent);

    const groupLabels: Record<string, string> = {};
    for (const source of sources) {
      if (source.experiment) groupLabels[source.experiment.id] = source.experiment.name;
    }
    if (sources.some((source) => !source.experiment)) groupLabels[UNGROUPED] = UNGROUPED_LABEL;

    const options: SelectItemOption[] = sources.map((source) => ({
      value: source.evaluation.name,
      label: source.evaluation.name,
      group: source.experiment?.id ?? UNGROUPED,
      // Typing an experiment's name narrows the list to that section; typing an evaluation's
      // name keeps every experiment that has a run by that name. Matching both in one string is
      // what lets a single picker stand in for an experiment filter plus an evaluation filter.
      // Agent names join in so another agent's runs are findable by who they evaluated.
      searchText: [
        source.experimentName ?? UNGROUPED_LABEL,
        source.evaluation.name,
        ...(source.evaluation.agent_names ?? []),
      ].join(' '),
    }));

    const byName: Record<string, EvaluationSource> = {};
    for (const source of allSources) byName[source.evaluation.name] = source;

    const isLoading = isEvaluationsLoading || isExperimentsLoading;
    return {
      sources,
      options,
      groupLabels,
      byName,
      experiments,
      isLoading,
      isEmpty: !isLoading && sources.length === 0,
      hasAgentSources: allSources.some((source) => !source.fromOtherAgent),
      hasOtherAgentSources: allSources.some((source) => source.fromOtherAgent),
    };
  }, [
    evaluations,
    experiments,
    agent,
    includeOtherAgents,
    isEvaluationsLoading,
    isExperimentsLoading,
  ]);
};
