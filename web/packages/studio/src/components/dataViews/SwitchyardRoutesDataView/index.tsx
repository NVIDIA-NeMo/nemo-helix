// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import {
  routingStrategyTitle,
  useRouteColumns,
} from '@studio/components/dataViews/SwitchyardRoutesDataView/useRouteColumns';
import type { SwitchyardRoute } from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/switchyardResults';
import { type FC, memo, useMemo } from 'react';

type SortableField = 'virtualModel' | 'capable' | 'efficient';

const sortValue = (route: SwitchyardRoute, sortId: string): string =>
  sortId === 'configType'
    ? routingStrategyTitle(route.configType)
    : (route[sortId as SortableField] ?? '');

/** What the search bar matches against: the VirtualModel, strategy, and every model. */
const searchableText = (route: SwitchyardRoute): string =>
  [
    route.virtualModel,
    routingStrategyTitle(route.configType),
    route.capable,
    route.efficient,
    route.judge ?? '',
  ]
    .join(' ')
    .toLowerCase();

export interface SwitchyardRoutesDataViewProps {
  routes: SwitchyardRoute[];
  /** Workspace the VirtualModels live in. */
  workspace: string;
  testingVirtualModel?: string;
  /** Called when a row is clicked, to test that route's VirtualModel. */
  onTest?: (route: SwitchyardRoute) => void;
  onDeploy?: (route: SwitchyardRoute) => void;
}

export const SwitchyardRoutesDataView: FC<SwitchyardRoutesDataViewProps> = memo(
  ({ routes, workspace, testingVirtualModel, onTest, onDeploy }) => {
    const dataViewState = useStudioDataViewState({
      defaultPageSize: 25,
      defaultSort: [{ id: 'capable', desc: false }],
    });

    const { debouncedSearchBar } = dataViewState;
    const sorting = dataViewState.sorting.state;
    const { pageIndex, pageSize } = dataViewState.pagination.state;

    const processedRoutes = useMemo(() => {
      const search = debouncedSearchBar.trim().toLowerCase();
      const filtered = search
        ? routes.filter((route) => searchableText(route).includes(search))
        : routes;

      const [sort] = sorting;
      if (!sort) return filtered;
      return [...filtered].sort((a, b) => {
        const order = sortValue(a, sort.id).localeCompare(sortValue(b, sort.id));
        return sort.desc ? -order : order;
      });
    }, [routes, debouncedSearchBar, sorting]);

    const lastPageIndex = Math.max(0, Math.ceil(processedRoutes.length / pageSize) - 1);
    const safePageIndex = Math.min(pageIndex, lastPageIndex);
    const pageRoutes = useMemo(
      () => processedRoutes.slice(safePageIndex * pageSize, safePageIndex * pageSize + pageSize),
      [processedRoutes, safePageIndex, pageSize]
    );

    const makeColumns = useRouteColumns({ workspace, testingVirtualModel, onDeploy });

    return (
      <StudioDataView<SwitchyardRoute>
        dataViewState={dataViewState}
        searchField="virtualModel"
        makeColumns={makeColumns}
        onRowClick={onTest}
        attributes={{
          DataViewSearchBar: { placeholder: 'Search virtual models or models...' },
          DataViewRoot: {
            data: pageRoutes,
            totalCount: processedRoutes.length,
          },
          DataViewTableContent: {
            renderEmptyState: () => (
              <TableEmptyState
                header={debouncedSearchBar.trim() ? 'No matching routes' : 'No routes'}
                emptyMessage={
                  debouncedSearchBar.trim()
                    ? 'No virtual model matches this search.'
                    : 'This run did not create any virtual models.'
                }
              />
            ),
          },
        }}
      />
    );
  }
);

SwitchyardRoutesDataView.displayName = 'SwitchyardRoutesDataView';
