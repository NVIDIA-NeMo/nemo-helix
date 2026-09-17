// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { usePlatformSdk } from '@agent-hardener/api/platform';
import { CancelJobButton } from '@agent-hardener/components/CancelJobButton';
import { HardenPanel } from '@agent-hardener/components/HardenPanel';
import {
  pendingInterview,
  pendingReview,
  type InterviewAnswer,
  type SuiteRow,
} from '@agent-hardener/components/hitlTypes';
import { InterviewPanel } from '@agent-hardener/components/InterviewPanel';
import { ReviewPanel } from '@agent-hardener/components/ReviewPanel';
import { MessageFeed } from '@agent-hardener/components/swarm/MessageFeed';
import { NodeDetail } from '@agent-hardener/components/swarm/NodeDetail';
import { SwarmGraph } from '@agent-hardener/components/swarm/SwarmGraph';
import { deriveSwarmState, NODES } from '@agent-hardener/components/swarm/swarmModel';
import { useSwarmEvents } from '@agent-hardener/components/swarm/useSwarmEvents';
import { useMitigations } from '@agent-hardener/components/useMitigations';
import { useAgentHardenerGetRun } from '@agent-hardener/generated/api';
import { useBreadcrumbs, useWorkspace } from '@agent-hardener/host';
import { getAgentHardenerRunListRoute } from '@agent-hardener/paths';
import { ACCENT, tint } from '@agent-hardener/theme';
import { AccessibleTitle, JOB_POLLING_INTERVAL_MS, getJobRefetchInterval } from '@nemo/common';
import {
  Badge,
  Banner,
  Card,
  Flex,
  Grid,
  PageHeader,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import { FC, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router';

// Fixed panel sizes, as styles rather than `h-[560px]`: Studio's Tailwind only
// scans web/packages/**, so an arbitrary-value class in a plugin has no CSS.
const PANEL_HEIGHT = 560;
const HARDEN_MIN_HEIGHT = 420;

const PILL_COLOR: Record<'good' | 'active', string> = {
  good: ACCENT.green,
  active: ACCENT.teal,
};

const Pill: FC<{ label: string; tone?: 'good' | 'active' }> = ({ label, tone }) => {
  const color = tone ? PILL_COLOR[tone] : undefined;
  return (
    <span
      className={`rounded-full border px-3 py-1 text-xs uppercase tracking-wide ${color ? '' : 'border-base text-subtle'}`}
      style={color ? { color, borderColor: tint(color, 40) } : undefined}
    >
      {label}
    </span>
  );
};

export const AgentHardenerRunDetailsRoute: FC = () => {
  const workspace = useWorkspace();
  const { agentHardenerRunName = '' } = useParams<{ agentHardenerRunName: string }>();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const { data: run } = useAgentHardenerGetRun(workspace, agentHardenerRunName, {
    query: {
      enabled: Boolean(agentHardenerRunName),
      refetchInterval: (query) =>
        query.state.data?.status === 'running' ? JOB_POLLING_INTERVAL_MS : false,
    },
  });

  // Live swarm state, folded from the run's event stream. Stop polling once the run is terminal —
  // otherwise a finished run is polled for as long as the tab stays open.
  const events = useSwarmEvents(
    workspace,
    agentHardenerRunName,
    Boolean(run?.status) && run?.status !== 'running'
  );
  const swarm = useMemo(() => deriveSwarmState(events), [events]);

  // The platform job behind this run drives the interview/review HITL over its status_details.
  const jobName = run?.job_id ?? '';
  const { useJobsGetJob, useJobsUpdateJobStatusDetails } = usePlatformSdk();
  const { data: job } = useJobsGetJob(workspace, jobName, {
    query: {
      enabled: Boolean(jobName),
      refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
    },
  });
  const patch = useJobsUpdateJobStatusDetails();
  const details = job?.status_details as Record<string, unknown> | undefined;
  const interview = pendingInterview(details);
  const review = pendingReview(details);

  // Hardening output (before/after policy + guardrails + selectable defenses), fetched from the job results
  // API once the run ends.
  const {
    mitigations,
    recommendations,
    defenses,
    isLoading: mitigationsLoading,
    hasMitigations,
  } = useMitigations(workspace, jobName, job?.status);

  useBreadcrumbs({
    items: [
      { href: getAgentHardenerRunListRoute(workspace), slotLabel: 'Agent Hardener' },
      { slotLabel: agentHardenerRunName },
    ],
  });

  const respond = (data: Record<string, unknown>) =>
    patch.mutate({ workspace, name: jobName, data });
  const onInterview = (answers: InterviewAnswer[]) =>
    interview && respond({ interview_response: { round: interview.round, answers } });
  const onReview = (suite: SuiteRow[]) =>
    review && respond({ review_response: { round: review.round, suite } });

  const selectedNode = NODES.find((n) => n.id === selectedId) ?? null;

  // Benign generation (and its interview) is a manifest activity, so a normal war-game run is pure mission
  // control. The Interview tab is a fallback — it appears only if a run genuinely pauses for input.
  const hitlPending = Boolean(interview || review);
  const [tab, setTab] = useState('swarm');
  // Owned here (not in HardenPanel) so an in-flight sanity check survives tab switches — the Harden
  // tab's content unmounts when another tab is active.
  const [sanityJob, setSanityJob] = useState<string>();
  const [composedGuardrails, setComposedGuardrails] = useState<string>();
  useEffect(() => {
    if (hitlPending) setTab('interview');
  }, [hitlPending]);
  // Jump to the Harden tab the moment its data lands — but never steal focus from a pending
  // interview, and only jump once (this fires only when `hasMitigations` flips to true, so a user
  // who navigates back to Swarm afterward isn't forced back).
  useEffect(() => {
    if (hasMitigations && !hitlPending) setTab('mitigations');
  }, [hasMitigations, hitlPending]);

  const swarmView = (
    <Grid cols={{ base: 1, xl: 2 }} gap="density-xl">
      <Card className="p-2" style={{ height: PANEL_HEIGHT }}>
        <SwarmGraph swarm={swarm} selectedId={selectedId} onSelect={setSelectedId} />
      </Card>
      <Stack gap="density-xl" className="min-h-0" style={{ height: PANEL_HEIGHT }}>
        {/* Agent view sits a little smaller than the live feed (5:6). */}
        <Card className="min-h-0 overflow-auto p-4" style={{ flex: 5 }}>
          <NodeDetail node={selectedNode} swarm={swarm} />
        </Card>
        {/* Deliberately not a KUI Card: Card wraps children in a grid `.nv-card-content` that grows to
            fit, and bounding it needs a descendant selector. Studio's Tailwind never emits an
            arbitrary-variant class for a plugin, so this panel is a plain bounded flex column instead —
            which is what lets the feed scroll internally rather than stretching the page. */}
        <div
          className="flex min-h-0 flex-col rounded-md border border-base bg-surface-raised p-4"
          style={{ flex: 6 }}
        >
          <Text kind="body/semibold/md" className="mb-2 shrink-0">
            Live Agent Feed
          </Text>
          <div className="min-h-0 flex-1">
            <MessageFeed events={events} />
          </div>
        </div>
      </Stack>
    </Grid>
  );

  // A failed run surfaces its classified cause (from the run record, falling back to the platform job's
  // error_details or the summary) as an inline error banner.
  const jobErrorMessage = (job?.error_details as { message?: unknown } | undefined)?.message;
  const failureMessage =
    run?.error_message ||
    (typeof jobErrorMessage === 'string' ? jobErrorMessage : '') ||
    run?.summary ||
    'War-game run failed.';

  return (
    <AccessibleTitle title={`Agent Hardener — ${agentHardenerRunName}`}>
      <Stack className="min-h-full" gap="density-xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={run?.agent ? `Hardening ${run.agent}` : 'War-Game Run'}
          slotDescription={run?.summary}
          slotActions={
            <Flex gap="density-sm" align="center">
              {swarm.round > 0 && <Pill label={`Round ${swarm.round}`} />}
              {swarm.phase && (
                <Pill label={swarm.phase} tone={swarm.finalPass ? 'good' : 'active'} />
              )}
              {run?.job_id && (
                <CancelJobButton
                  workspace={workspace}
                  jobName={run.job_id}
                  jobStatus={job?.status}
                  compact
                />
              )}
            </Flex>
          }
        />

        {run?.status === 'failed' && (
          <Banner kind="inline" status="error">
            <Stack gap="density-xs">
              <Text kind="body/semibold/md">{failureMessage}</Text>
              {run.error_remediation ? (
                <Text kind="body/regular/sm">{run.error_remediation}</Text>
              ) : null}
            </Stack>
          </Banner>
        )}

        {hitlPending || hasMitigations ? (
          <TabsRoot value={tab} onValueChange={setTab}>
            <TabsList>
              <TabsTrigger value="swarm">Swarm</TabsTrigger>
              {hitlPending ? (
                <TabsTrigger value="interview">
                  <Flex gap="density-xs" align="center">
                    Interview
                    <Badge color="yellow">Action required</Badge>
                  </Flex>
                </TabsTrigger>
              ) : null}
              {hasMitigations ? (
                <TabsTrigger value="mitigations">
                  <Flex gap="density-xs" align="center">
                    Harden
                    {(defenses.length || recommendations.length) > 0 && (
                      <Badge color="green">{defenses.length || recommendations.length}</Badge>
                    )}
                  </Flex>
                </TabsTrigger>
              ) : null}
            </TabsList>

            <TabsContent value="swarm" className="p-0 pt-4">
              {swarmView}
            </TabsContent>

            {hitlPending ? (
              <TabsContent value="interview" className="p-0 pt-4">
                <Card className="p-6" style={{ minHeight: HARDEN_MIN_HEIGHT }}>
                  {interview ? (
                    <InterviewPanel
                      // Remount per round: the panel seeds its answers on mount, so without this a
                      // round that arrives while the panel stays mounted reuses the previous answers.
                      key={interview.round}
                      prompt={interview}
                      loading={patch.isPending}
                      onSubmit={onInterview}
                    />
                  ) : review ? (
                    <ReviewPanel
                      key={review.round}
                      suite={review.suite}
                      loading={patch.isPending}
                      onSubmit={onReview}
                    />
                  ) : null}
                </Card>
              </TabsContent>
            ) : null}

            {hasMitigations ? (
              <TabsContent value="mitigations" className="p-0 pt-4">
                <HardenPanel
                  mitigations={mitigations}
                  defenses={defenses}
                  isLoading={mitigationsLoading}
                  workspace={workspace}
                  runName={agentHardenerRunName}
                  agentName={run?.agent}
                  manifestId={run?.manifest_id}
                  hitlogFileset={run?.hitlog_fileset}
                  sanityJob={sanityJob}
                  onSanityJobChange={setSanityJob}
                  composedGuardrails={composedGuardrails}
                  onComposedGuardrailsChange={setComposedGuardrails}
                />
              </TabsContent>
            ) : null}
          </TabsRoot>
        ) : (
          swarmView
        )}
      </Stack>
    </AccessibleTitle>
  );
};
