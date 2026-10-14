// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile } from '@nemo/common/src/components/StatTile';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema/RunStrategyJob';
import { Grid, Text } from '@nvidia/foundations-react-core';
import { SwitchyardRoutesDataView } from '@studio/components/dataViews/SwitchyardRoutesDataView';
import { DeployRouteModal } from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/DeployRouteModal';
import { RoutingTestPanel } from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/RoutingTestPanel';
import type {
  SwitchyardResults,
  SwitchyardRoute,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/switchyardResults';
import { type FC, useCallback, useRef, useState } from 'react';

const SwitchyardStatTiles: FC<{ results: SwitchyardResults }> = ({ results }) => {
  const { routes } = results;
  const strategies = new Set(routes.map((route) => route.configType));

  return (
    <Grid cols={{ base: 1, md: 2, xl: 4 }} gap="density-xl" className="shrink-0">
      <StatTile variant="metric" label="Virtual models" value={String(routes.length)} />
      <StatTile variant="metric" label="Routing strategies" value={String(strategies.size)} />
    </Grid>
  );
};

export interface SwitchyardResultsViewProps {
  workspace: string;
  job: RunStrategyJob;
  results: SwitchyardResults;
}

/**
 * A finished Switchyard run: the VirtualModels it created, a chat to test one's routing (pick a
 * row to test its VirtualModel), and a deploy action that registers the source agent pointed at a
 * chosen VirtualModel.
 */
export const SwitchyardResultsView: FC<SwitchyardResultsViewProps> = ({
  workspace,
  job,
  results,
}) => {
  const [selectedRoute, setSelectedRoute] = useState<SwitchyardRoute | null>(null);
  const [deployRoute, setDeployRoute] = useState<SwitchyardRoute | null>(null);
  const testPanelRef = useRef<HTMLDivElement>(null);

  // The first route is under test until a row is picked.
  const testing = selectedRoute ?? results.routes[0];

  const onTest = useCallback((route: SwitchyardRoute) => {
    setSelectedRoute(route);
    testPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);
  const onDeploy = useCallback((route: SwitchyardRoute) => setDeployRoute(route), []);

  if (!results.routes.length) {
    return (
      <Text kind="body/regular/md" className="text-secondary">
        This run did not create any virtual models.
      </Text>
    );
  }

  return (
    <>
      <SwitchyardStatTiles results={results} />
      <SwitchyardRoutesDataView
        routes={results.routes}
        workspace={results.workspace}
        testingVirtualModel={testing?.virtualModel}
        onTest={onTest}
        onDeploy={onDeploy}
      />
      {testing && (
        <div ref={testPanelRef} className="scroll-mt-6">
          <RoutingTestPanel workspace={results.workspace} route={testing} />
        </div>
      )}
      <DeployRouteModal
        workspace={workspace}
        job={job}
        results={results}
        route={deployRoute}
        onClose={() => setDeployRoute(null)}
      />
    </>
  );
};
