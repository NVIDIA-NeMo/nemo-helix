// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { Root as DataViewRoot } from '@nemo/common/src/components/DataView/internal';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { JOB_POLLING_INTERVAL_LONG } from '@nemo/common/src/constants';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getSortParamWithWhitelist } from '@nemo/common/src/utils/query';
import {
  agentsListDeployments,
  getAgentsListDeploymentsQueryKey,
  useAgentsDeleteDeployment,
  useAgentsListDeployments,
} from '@nemo/sdk/generated/agents/agent-deployments';
import {
  getAgentsListAgentsQueryKey,
  useAgentsDeleteAgent,
  useAgentsListAgents,
} from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Button, type DropdownEntry, Text } from '@nvidia/foundations-react-core';
import { getAgentModelNames } from '@studio/components/dataViews/AgentsDataView/utils';
import { MODEL_COMPARE_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getModelCompareRoute } from '@studio/routes/utils';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { Trash } from 'lucide-react';
import { ComponentProps, FC, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

export type { Agent, AgentDeployment };

/** A model entry in a `nemo-agents-spec-v1` config, under `models` or on a harness. */
export interface AgentSpecModel {
  provider?: string;
  model?: string;
  api_key_env?: string;
  temperature?: number;
  settings?: Record<string, unknown>;
}

export interface AgentConfig {
  functions?: Record<string, { _type: string }>;
  llms?: Record<
    string,
    {
      _type: string;
      model_name?: string;
      api_key?: string;
      temperature?: number;
      base_url?: string;
    }
  >;
  /** `nemo-agents-spec-v1` named models, e.g. `{ default: { model: 'nvidia-…' } }`. */
  models?: Record<string, AgentSpecModel>;
  harnesses?: Record<string, { kind?: string; model?: AgentSpecModel }>;
  workflow?: {
    _type: string;
    tool_names?: string[];
    llm_name?: string;
    verbose?: boolean;
    parse_agent_response_max_retries?: number;
  };
}

export type AgentItem = Agent & { id: string };
export type AgentEntity = AgentDeployment & { id: string };

const TERMINAL_DEPLOYMENT_STATUSES = new Set(['running', 'failed', 'stopped', 'error']);

export type AgentTableRow = {
  id: string;
  name: string;
  workspace: string;
  description?: string;
  config?: AgentConfig;
  config_format?: string;
  created_at?: string;
  models: string[];
  deploymentsStatus: string;
  deploymentsDeploying: boolean;
};

const SORTABLE_FIELDS = ['name', 'created_at'] as const;
const DEFAULT_SORT = '-created_at';

type DeleteState =
  | { kind: 'agent'; item: AgentTableRow }
  | { kind: 'bulk'; items: AgentTableRow[] }
  | null;

export interface CombinedAgentsTableProps {
  onAgentRowClick?: (agent: AgentTableRow, event: React.MouseEvent) => void;
  onCreateDeployment?: (agentName: string) => void;
  onCloneAgent?: (agent: AgentTableRow) => void;
  onAgentsLoaded?: (agents: Agent[]) => void;
  canTestModels?: boolean;
}

export const AgentsTable: FC<CombinedAgentsTableProps> = ({
  onAgentRowClick,
  onCreateDeployment,
  onCloneAgent,
  onAgentsLoaded,
  canTestModels = MODEL_COMPARE_ENABLED,
}) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const toast = useToast();
  const [deleteState, setDeleteState] = useState<DeleteState>(null);

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  // `keepPreviousData` keeps the previous workspace's rows on screen after a switch,
  // so a selection made there would still resolve — and delete by name against the new
  // workspace. Drop it as soon as the workspace changes.
  const clearRowSelection = dataViewState.rowSelection.set;
  useEffect(() => {
    clearRowSelection({});
  }, [workspace, clearRowSelection]);

  const page = dataViewState.pagination.state.pageIndex + 1;
  const pageSize = dataViewState.pagination.state.pageSize;
  const sortParam = getSortParamWithWhitelist(
    dataViewState.sorting.state,
    SORTABLE_FIELDS,
    DEFAULT_SORT
  );

  const {
    data: agentsResponse,
    isLoading: agentsLoading,
    error: agentsError,
  } = useAgentsListAgents(
    workspace,
    { page, page_size: pageSize, sort: sortParam },
    {
      query: {
        placeholderData: keepPreviousData,
        refetchInterval: JOB_POLLING_INTERVAL_LONG,
      },
    }
  );

  const { data: deploymentsResponse } = useAgentsListDeployments(workspace, undefined, {
    query: { refetchInterval: JOB_POLLING_INTERVAL_LONG },
  });

  const agentsData = agentsResponse?.data;

  useEffect(() => {
    if (agentsData) onAgentsLoaded?.(agentsData);
  }, [agentsData, onAgentsLoaded]);

  const totalCount = agentsResponse?.pagination?.total_results ?? agentsData?.length ?? 0;
  const deploymentsData = deploymentsResponse?.data;

  const tableData = useMemo<AgentTableRow[]>(() => {
    const deployments = deploymentsData ?? [];
    return (agentsData ?? []).map((agent) => {
      const agentDeployments = deployments.filter((d) => d.agent === agent.name);
      const total = agentDeployments.length;
      const healthy = agentDeployments.filter((d) => d.status === 'running').length;
      const deploymentsDeploying = agentDeployments.some(
        (d) => d.status && !TERMINAL_DEPLOYMENT_STATUSES.has(d.status) && d.status !== 'deleting'
      );
      const deploymentsStatus = total === 0 ? 'No Deployments' : `${healthy}/${total} Healthy`;
      const config = agent.config as AgentConfig | undefined;
      return {
        id: agent.id ?? agent.name ?? '',
        name: agent.name ?? '',
        workspace: agent.workspace,
        description: agent.description,
        config,
        config_format: agent.config_format,
        created_at: agent.created_at,
        models: getAgentModelNames(config),
        deploymentsStatus,
        deploymentsDeploying,
      };
    });
  }, [agentsData, deploymentsData]);

  const deleteAgentMutation = useAgentsDeleteAgent();
  const deleteDeploymentMutation = useAgentsDeleteDeployment();

  const refreshAfterDelete = () => {
    void queryClient.refetchQueries({ queryKey: getAgentsListAgentsQueryKey(workspace) });
    void queryClient.invalidateQueries({ queryKey: getAgentsListDeploymentsQueryKey(workspace) });
  };

  const fetchAgentDeploymentNames = async (agentName: string): Promise<string[]> => {
    const PAGE_SIZE = 100;
    const MAX_PAGES = 50;
    const names: string[] = [];
    for (let page = 1; page <= MAX_PAGES; page += 1) {
      const resp = await agentsListDeployments(workspace, { page, page_size: PAGE_SIZE });
      for (const d of resp.data ?? []) {
        if (d.agent === agentName && d.name) names.push(d.name);
      }
      if (page >= (resp.pagination?.total_pages ?? 1)) break;
    }
    return names;
  };

  const deleteAgentWithDeployments = async (agentName: string): Promise<void> => {
    const deploymentNames = await fetchAgentDeploymentNames(agentName);
    await Promise.all(
      deploymentNames.map((name) =>
        deleteDeploymentMutation.mutateAsync({ workspace, name }).catch((err) => {
          // A deployment already gone (404) is fine; anything else aborts this agent.
          if ((err as { response?: { status?: number } })?.response?.status === 404) return;
          throw err;
        })
      )
    );
    await deleteAgentMutation.mutateAsync({ workspace, name: agentName });
  };

  const handleDelete = async (): Promise<boolean> => {
    if (!deleteState) return false;
    const agents = deleteState.kind === 'agent' ? [deleteState.item] : deleteState.items;
    const results = await Promise.allSettled(agents.map((a) => deleteAgentWithDeployments(a.name)));
    const failed = results.filter((r) => r.status === 'rejected').length;

    refreshAfterDelete();
    if (failed < agents.length) dataViewState.rowSelection.set({});

    if (failed > 0) {
      toast.error(
        agents.length === 1
          ? 'Failed to delete agent.'
          : `Failed to delete ${failed} of ${agents.length} agents.`
      );
      return false;
    }
    toast.success(agents.length === 1 ? 'Agent deleted.' : `${agents.length} agents deleted.`);
    return true;
  };

  const makeColumns: ComponentProps<typeof DataViewRoot<AgentTableRow>>['makeColumns'] = (
    { accessor },
    { rowActionsColumn, rowSelectionColumn }
  ) => [
    rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
    accessor('name', {
      header: 'Name',
      enableSorting: true,
    }),
    accessor('description', {
      header: 'Description',
      enableSorting: false,
      cell: ({ row }) => <Text>{row.original.description || '-'}</Text>,
    }),
    accessor('models', {
      header: 'Model',
      enableSorting: false,
      cell: ({ row }) => <Text>{row.original.models.join(', ') || '-'}</Text>,
    }),
    accessor('deploymentsStatus', {
      header: 'Deployments',
      enableSorting: false,
      cell: ({ row }) =>
        row.original.deploymentsDeploying ? (
          <Text>Deploying...</Text>
        ) : (
          <Text>{row.original.deploymentsStatus}</Text>
        ),
    }),
    accessor('created_at', {
      header: 'Created',
      size: 200,
      enableSorting: true,
      cell: ({ row }) =>
        row.original.created_at ? (
          <RelativeTime datetime={row.original.created_at} />
        ) : (
          <Text>-</Text>
        ),
    }),
    rowActionsColumn({
      size: ROW_ACTIONS_COLUMN_SIZE,
      enableResizing: false,
      rowActions: (row: AgentTableRow): DropdownEntry[] => [
        {
          children: 'Deploy',
          onSelect: () => onCreateDeployment?.(row.name),
        },
        ...(canTestModels
          ? [
              {
                children: 'Test models',
                onSelect: () => {
                  const target = getModelCompareRoute(workspace);
                  const model = row.models[0];
                  const urn = model ? `${row.workspace}/${model}` : null;
                  navigate(urn ? `${target}?model=${encodeURIComponent(urn)}` : target);
                },
              },
            ]
          : []),
        {
          children: 'Clone',
          onSelect: () => onCloneAgent?.(row),
        },
        { kind: 'divider' as const },
        {
          children: 'Delete',
          danger: true,
          onSelect: () => setDeleteState({ kind: 'agent', item: row }),
        },
      ],
    }),
  ];

  return (
    <>
      <StudioDataView
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        renderBulkActions={({ selectedRows }) => (
          <Button
            kind="tertiary"
            aria-label="Delete selected agents"
            onClick={() => setDeleteState({ kind: 'bulk', items: selectedRows })}
          >
            <Trash /> Delete
          </Button>
        )}
        onRowClick={(row: AgentTableRow, _index, event) => {
          onAgentRowClick?.(row, event);
        }}
        attributes={{
          DataViewRoot: {
            data: tableData,
            totalCount,
            requestStatus: agentsError
              ? 'error'
              : agentsLoading && !agentsData
                ? 'loading'
                : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
              hasFiltersApplied || hasSearchApplied ? (
                <EntityEmptyState
                  entity="agents"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState entity="agents" variant="first-use" />
              ),
            renderErrorState: () => (
              <ErrorPanel
                errorMessage={getErrorMessage(agentsError ?? new Error('Failed to load agents.'))}
              />
            ),
          },
        }}
      />
      {deleteState && (
        <DeleteConfirmationModal
          open
          title={
            deleteState.kind === 'bulk'
              ? `Delete ${deleteState.items.length} Agent${deleteState.items.length === 1 ? '' : 's'}`
              : 'Delete Agent'
          }
          description={
            deleteState.kind === 'bulk'
              ? `Are you sure you want to delete ${deleteState.items.length} agent${deleteState.items.length === 1 ? '' : 's'} and all their deployments?`
              : 'Are you sure you want to delete this agent and all its deployments?'
          }
          onDelete={handleDelete}
          onClose={() => setDeleteState(null)}
          simpleConfirm
          suppressResultToasts
        />
      )}
    </>
  );
};
