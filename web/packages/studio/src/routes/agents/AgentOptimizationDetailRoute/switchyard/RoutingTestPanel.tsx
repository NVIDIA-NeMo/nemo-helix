// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AssistantMessageCompletion } from '@nemo/common/src/components/AssistantChat/types';
import { ChatMulti } from '@nemo/icons/icons';
import { useModelsListProviders } from '@nemo/sdk/generated/platform/model-providers';
import {
  Anchor,
  Badge,
  Block,
  Button,
  Divider,
  Flex,
  Grid,
  Panel,
  ProgressBar,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { DEFAULT_INFERENCE_PARAMS, type InferenceParams } from '@studio/components/chat/params';
import { ParamsPopover } from '@studio/components/chat/ParamsPopover';
import {
  routeSettingText,
  routingStrategyTitle,
} from '@studio/components/dataViews/SwitchyardRoutesDataView/useRouteColumns';
import { ModelChat } from '@studio/components/ModelChat';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import {
  type RoleTally,
  type RoutedResponse,
  type RouteRole,
  servedNamesByEntity,
  tallyRoutes,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/routingTally';
import {
  routeThreshold,
  type SwitchyardRoute,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/switchyardResults';
import { getVirtualModelDetailRoute } from '@studio/routes/utils';
import { type FC, useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router';

const ROLE_BADGE: Record<RouteRole, { label: string; color: 'purple' | 'green' | 'gray' }> = {
  capable: { label: 'Capable', color: 'purple' },
  efficient: { label: 'Efficient', color: 'green' },
  unknown: { label: 'Other', color: 'gray' },
};

const formatShare = (share: number): string =>
  share.toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 0 });

const formatThreshold = (value: number): string =>
  value.toLocaleString(undefined, { maximumFractionDigits: 3 });

/** How the route's strategy picks a model, so the observed split can be read against it. */
const routingExplanation = (route: SwitchyardRoute): string => {
  const threshold = routeThreshold(route);
  if (route.configType === 'random_routing') {
    return threshold === undefined
      ? 'Each request goes to one of the two models at random, regardless of what it asks.'
      : `Each request goes to the capable model with probability ${formatShare(threshold)}, regardless of what it asks.`;
  }
  if (route.configType === 'stage_router') {
    const below = threshold === undefined ? '' : ` below ${formatThreshold(threshold)}`;
    return `Conversations start on the efficient model and escalate to the capable model only when the router's confidence drops${below}, based on signals from earlier turns. A single self-contained prompt usually stays on the efficient model; try a longer, multi-turn conversation.`;
  }
  if (route.configType === 'llm_classifier') {
    return 'A judge model rates each request, and requests it rates past the threshold go to the capable model. Compare an easy prompt with a hard one.';
  }
  return "Routing depends on the route's middleware configuration.";
};

const RoleBadge: FC<{ role: RouteRole }> = ({ role }) => (
  <Badge kind="outline" color={ROLE_BADGE[role].color}>
    {ROLE_BADGE[role].label}
  </Badge>
);

const TallyRow: FC<{ tally: RoleTally }> = ({ tally }) => (
  <Stack gap="1" role="listitem" aria-label={`${ROLE_BADGE[tally.role].label} model`}>
    <Flex align="center" justify="between" gap="density-md">
      <Flex align="center" gap="density-sm" className="min-w-0">
        <RoleBadge role={tally.role} />
        <Text kind="body/regular/sm" className="truncate" title={tally.models.join(', ')}>
          {tally.models.join(', ')}
        </Text>
      </Flex>
      <Text kind="body/semibold/sm" className="shrink-0 tabular-nums">
        {tally.count} · {formatShare(tally.share)}
      </Text>
    </Flex>
    <ProgressBar
      kind="determinate"
      value={tally.share * 100}
      aria-label={`${ROLE_BADGE[tally.role].label} share`}
    />
  </Stack>
);

export interface RoutingTestPanelProps {
  /** Workspace the VirtualModels live in. */
  workspace: string;
  route: SwitchyardRoute;
}

/**
 * Chat with one route's VirtualModel and watch which upstream model answers. Each response's
 * `model` is matched against the route's capable and efficient models through the providers'
 * served model names, so the split shows in model terms rather than raw upstream ids.
 */
export const RoutingTestPanel: FC<RoutingTestPanelProps> = ({ workspace, route }) => {
  const { virtualModel } = route;
  const [inferenceParams, setInferenceParams] = useState<InferenceParams>({
    ...DEFAULT_INFERENCE_PARAMS,
    max_tokens: 1024,
  });
  // Kept per VirtualModel, so switching routes and back keeps what was already observed.
  const [responsesByModel, setResponsesByModel] = useState<Record<string, RoutedResponse[]>>({});
  const responses = useMemo(
    () => responsesByModel[virtualModel] ?? [],
    [responsesByModel, virtualModel]
  );

  const { data: providersPage } = useModelsListProviders(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { staleTime: 60_000 } }
  );
  const servedNames = useMemo(
    () => servedNamesByEntity(providersPage?.data ?? []),
    [providersPage]
  );

  const onMessageComplete = useCallback(
    (info: AssistantMessageCompletion) =>
      setResponsesByModel((prev) => ({
        ...prev,
        [virtualModel]: [
          ...(prev[virtualModel] ?? []),
          { id: info.assistantMessageId, responseModel: info.responseModel },
        ],
      })),
    [virtualModel]
  );
  const clear = () => setResponsesByModel((prev) => ({ ...prev, [virtualModel]: [] }));

  const tallies = useMemo(
    () => tallyRoutes(route, responses, servedNames),
    [route, responses, servedNames]
  );

  return (
    <Panel
      slotHeading="Test routing"
      slotIcon={<ChatMulti />}
      elevation="high"
      density="compact"
      data-testid="routing-test-panel"
    >
      <Stack gap="density-xl">
        <Flex align="center" justify="between" gap="density-md" wrap="wrap">
          <Flex align="center" gap="density-sm" wrap="wrap" className="min-w-0">
            <Anchor asChild>
              <Link to={getVirtualModelDetailRoute(workspace, virtualModel)} className="truncate">
                {virtualModel}
              </Link>
            </Anchor>
            <Text kind="body/regular/sm" className="text-secondary">
              {routingStrategyTitle(route.configType)} · {routeSettingText(route)}
            </Text>
          </Flex>
          <ParamsPopover value={inferenceParams} onChange={setInferenceParams} />
        </Flex>

        <Grid cols={{ base: 1, xl: 3 }} gap="density-xl">
          <Block className="h-[560px] min-h-0 xl:col-span-2">
            <ModelChat
              key={`${workspace}/${virtualModel}`}
              model={virtualModel}
              workspace={workspace}
              assistantName={virtualModel}
              promptData={{ inference_params: inferenceParams }}
              onMessageComplete={onMessageComplete}
            />
          </Block>

          <Stack
            gap="density-lg"
            className="min-w-0 self-start rounded-lg border border-base p-density-lg"
            aria-label="Routing split"
          >
            <Flex align="center" justify="between" gap="density-md">
              <Text kind="label/semibold/md">Routing split</Text>
              <Button kind="tertiary" size="small" onClick={clear} disabled={!responses.length}>
                Clear
              </Button>
            </Flex>
            <Text kind="body/regular/sm" className="text-secondary" role="status">
              {responses.length
                ? `${responses.length} ${responses.length === 1 ? 'response' : 'responses'} observed.`
                : 'Send a message to see which model answers it.'}
            </Text>
            <Stack gap="density-md" role="list" aria-label="Responses by model">
              {tallies.map((tally) => (
                <TallyRow key={tally.role} tally={tally} />
              ))}
            </Stack>
            {tallies.some((tally) => tally.role === 'unknown') && (
              <Text kind="body/regular/xs" className="text-secondary">
                &quot;Other&quot; responses reported a model that is neither of this route&apos;s
                models, or none at all.
              </Text>
            )}

            <Divider />

            <Stack gap="density-xs">
              <Text kind="label/semibold/sm">How this route decides</Text>
              <Text kind="body/regular/sm" className="text-secondary">
                {routingExplanation(route)}
              </Text>
            </Stack>
          </Stack>
        </Grid>
      </Stack>
    </Panel>
  );
};
