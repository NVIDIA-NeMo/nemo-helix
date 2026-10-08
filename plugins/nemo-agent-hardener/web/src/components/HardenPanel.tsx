// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { usePlatformSdk } from '@agent-hardener/api/platform';
import { ConfigDiff } from '@agent-hardener/components/ConfigDiff';
import { SanityCheckReport } from '@agent-hardener/components/SanityCheckReport';
import {
  cleanAttackPrompt,
  shortProbe,
  type DefensePair,
  type Mitigations,
} from '@agent-hardener/components/useMitigations';
import {
  useComposeDefense,
  useLatestSanityCheckJob,
  useSanityCheckComposedGuardrails,
  useSanityCheckResult,
  useSubmitSanityCheck,
} from '@agent-hardener/components/useSanityCheck';
import { useAgentHardenerApplyMitigation, useAgentHardenerGetManifest } from '@agent-hardener/generated/api';
import { useNotify, useToast } from '@agent-hardener/host';
import { ACCENT, tint } from '@agent-hardener/theme';
import { AccordionSection, ConfirmationModal, getJobRefetchInterval } from '@nemo/common';
import {
  AccordionRoot,
  Badge,
  Button,
  Card,
  Grid,
  Flex,
  Spinner,
  Stack,
  Switch,
  Text,
} from '@nvidia/foundations-react-core';
import { FC, useEffect, useState } from 'react';

interface HardenPanelProps {
  mitigations?: Mitigations;
  defenses: DefensePair[];
  isLoading: boolean;
  workspace: string;
  runName: string;
  agentName?: string;
  manifestId?: string;
  hitlogFileset?: string;
  // Sanity-check state is owned by the parent route so an in-flight check (and its report) survives
  // switching away from and back to the Harden tab — the tab content unmounts on switch.
  sanityJob?: string;
  onSanityJobChange: (job: string | undefined) => void;
  composedGuardrails?: string;
  onComposedGuardrailsChange: (guardrails: string | undefined) => void;
}

interface DefenseGroup {
  tool: string;
  items: DefensePair[];
}

// Group defenses by the tool they guard (policy + tool-less land in their own buckets), preserving order.
const groupDefenses = (defenses: DefensePair[]): DefenseGroup[] => {
  const order: string[] = [];
  const byTool = new Map<string, DefensePair[]>();
  for (const defense of defenses) {
    const tool =
      defense.target_tool || (defense.kind === 'policy' ? 'OpenShell sandbox policy' : 'Other');
    if (!byTool.has(tool)) {
      byTool.set(tool, []);
      order.push(tool);
    }
    byTool.get(tool)?.push(defense);
  }
  return order.map((tool) => ({ tool, items: byTool.get(tool) ?? [] }));
};

/**
 * Whether "Apply to Agent" may be clicked.
 *
 * A project-source manifest has no registered agent to adopt onto; its run carries the *manifest*
 * name in `agent`, so an apply targets whatever agent happens to share that name. Until the
 * manifest query has actually succeeded, `isProjectSource` reads `false` by default — enabling on
 * that default would open a window, while the query is loading or failed, to apply a mitigation to
 * an unrelated same-named agent.
 *
 * Mirrors the API's own rule (`_reject_project_source`): a run with no manifest behind it — one
 * launched from `run --config` — has no project-source claim to verify, and the API allows it.
 * Gating those on a query that never runs would grey the button out permanently.
 */
export const canApplyMitigation = (params: {
  hasComposedGuardrails: boolean;
  hasManifestId: boolean;
  manifestQuerySucceeded: boolean;
  isProjectSource: boolean;
}): boolean => {
  if (!params.hasComposedGuardrails) return false;
  if (!params.hasManifestId) return true;
  return params.manifestQuerySucceeded && !params.isProjectSource;
};

// One defense: a scannable row (toggle · shield · rule · the attack it counters) that expands to the full
// attack → mitigation story. Collapsed by default so 15 of these stay glanceable.
const DefenseRow: FC<{ defense: DefensePair; checked: boolean; onToggle: () => void }> = ({
  defense,
  checked,
  onToggle,
}) => {
  const [open, setOpen] = useState(false);
  const attack = defense.attack;
  return (
    <div
      className={`border-t border-base border-l-2 transition-opacity ${checked ? '' : 'opacity-55'}`}
      style={{ borderLeftColor: checked ? tint(ACCENT.green, 70) : 'transparent' }}
    >
      <Flex align="center" gap="density-sm" className="px-3 py-2">
        <Switch checked={checked} onCheckedChange={onToggle} aria-label={`Include ${defense.id}`} />
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <span className="w-3 shrink-0 text-subtle">{open ? '▾' : '▸'}</span>
          <Badge color={defense.kind === 'guardrail' ? 'green' : 'purple'}>
            {defense.kind === 'guardrail' ? 'Guardrail' : 'Policy'}
          </Badge>
          <Text kind="body/semibold/sm" className="truncate">
            {defense.summary}
          </Text>
        </button>
        {attack?.probe ? (
          <div className="shrink-0">
            <Badge color="yellow">{shortProbe(attack.probe)}</Badge>
          </div>
        ) : null}
      </Flex>
      {open ? (
        <Grid cols={{ base: 1, lg: 2 }} gap="density-sm" className="px-3 pb-3 pt-1">
          <Stack
            gap="density-xxs"
            className="rounded border p-3"
            style={{ borderColor: tint(ACCENT.red, 30), backgroundColor: tint(ACCENT.red, 5) }}
          >
            <Text kind="body/semibold/xs" style={{ color: ACCENT.red }}>
              ATTACK{attack?.probe ? ` · ${shortProbe(attack.probe)}` : ''}
            </Text>
            {attack?.goal ? <Text kind="body/regular/sm">{attack.goal}</Text> : null}
            {attack?.prompt_excerpt ? (
              <Text kind="body/regular/xs" className="whitespace-pre-wrap text-subtle">
                {cleanAttackPrompt(attack.prompt_excerpt)}
              </Text>
            ) : (
              <Text kind="body/regular/xs" className="text-subtle">
                No linked attack recorded for this defense.
              </Text>
            )}
          </Stack>
          <Stack
            gap="density-xxs"
            className="rounded border p-3"
            style={{ borderColor: tint(ACCENT.green, 30), backgroundColor: tint(ACCENT.green, 5) }}
          >
            <Text kind="body/semibold/xs" style={{ color: ACCENT.green }}>
              MITIGATION{defense.target_tool ? ` · ${defense.target_tool}` : ''}
            </Text>
            <Text kind="body/regular/sm">{defense.summary}</Text>
            {defense.config_fragment ? (
              <pre className="max-h-56 overflow-auto rounded bg-surface-overlay p-2 text-xs text-primary">
                {defense.config_fragment}
              </pre>
            ) : null}
          </Stack>
        </Grid>
      ) : null}
    </div>
  );
};

// Post-run harden flow: review the attack→defense pairs, keep the ones to apply, sanity-check the selection by
// replaying the recorded attacks + benign suite against it (no new mitigations), then apply to the agent.
export const HardenPanel: FC<HardenPanelProps> = ({
  mitigations,
  defenses,
  isLoading,
  workspace,
  runName,
  agentName,
  manifestId,
  hitlogFileset,
  sanityJob,
  onSanityJobChange,
  composedGuardrails,
  onComposedGuardrailsChange,
}) => {
  const toast = useToast();
  const notify = useNotify();
  const [selected, setSelected] = useState<Set<string>>(() => new Set(defenses.map((d) => d.id)));
  useEffect(() => {
    setSelected(new Set(defenses.map((d) => d.id)));
  }, [defenses]);

  const compose = useComposeDefense(workspace, runName);
  const { submit, isPending: submitting } = useSubmitSanityCheck(workspace);
  // The live job (just submitted) wins; else re-attach the most recent persisted sanity check for this run so
  // the scorecard survives a reload / re-visit.
  const persistedJob = useLatestSanityCheckJob(workspace, runName);
  const effectiveSanityJob = sanityJob ?? persistedJob;
  // The sanity job's status is what tells us a check died without writing a report — without it a
  // failed job leaves this panel spinning forever.
  const { useJobsGetJob } = usePlatformSdk();
  const { data: sanityJobDetail } = useJobsGetJob(workspace, effectiveSanityJob ?? '', {
    query: {
      enabled: Boolean(effectiveSanityJob),
      refetchInterval: (query) => getJobRefetchInterval(query.state.data?.status),
    },
  });
  const sanityStatus = sanityJobDetail?.status;
  const {
    report,
    isLoading: reportLoading,
    missingReport,
  } = useSanityCheckResult(workspace, effectiveSanityJob, sanityStatus);
  // After a reload the in-memory composed guardrail set is gone; recover it from the sanity run so Apply still works.
  const recoveredComposedGuardrails = useSanityCheckComposedGuardrails(
    workspace,
    effectiveSanityJob,
    sanityStatus
  );
  const effectiveComposedGuardrails = composedGuardrails ?? recoveredComposedGuardrails;
  const [preview, setPreview] = useState<{ guardrails?: string }>();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const applyMitigation = useAgentHardenerApplyMitigation();
  // A bring-your-own manifest has no registry entity to adopt onto: its run carries the *manifest*
  // name in `agent`, so an apply would target whatever agent happens to share that name.
  const manifestQuery = useAgentHardenerGetManifest(workspace, manifestId ?? '', {
    query: { enabled: Boolean(manifestId) },
  });
  const isProjectSource = manifestQuery.data?.source_type === 'project';

  if (isLoading) {
    return (
      <Flex align="center" gap="density-sm" className="p-6">
        <Spinner size="small" aria-label="Loading recommendations" />
        <Text kind="body/regular/md" className="text-subtle">
          Loading recommendations…
        </Text>
      </Flex>
    );
  }

  if (defenses.length === 0 && !mitigations?.guardrails && !mitigations?.policy) {
    return (
      <Card className="p-6">
        <Text kind="body/regular/md" className="text-subtle">
          No mitigations were produced for this run.
        </Text>
      </Card>
    );
  }

  const selectedIds = [...selected];
  const total = defenses.length;
  const groups = groupDefenses(defenses);

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleGroup = (items: DefensePair[]) =>
    setSelected((prev) => {
      const next = new Set(prev);
      const allOn = items.every((i) => next.has(i.id));
      for (const i of items) {
        if (allOn) next.delete(i.id);
        else next.add(i.id);
      }
      return next;
    });

  const composeSelection = async (): Promise<{
    guardrails_toml?: string | null;
    policy_yaml?: string | null;
  } | null> => {
    if (!mitigations) return null;
    try {
      return await compose.mutateAsync({ mitigations, selectedDefenseIds: selectedIds });
    } catch {
      toast.error('Failed to compose the selected defenses.');
      return null;
    }
  };

  const previewComposed = async () => {
    const composed = await composeSelection();
    if (composed) setPreview({ guardrails: composed.guardrails_toml ?? undefined });
  };

  const runSanityCheck = async () => {
    if (!hitlogFileset) return toast.error('This run has no recorded attacks to replay.');
    if (!manifestId) return toast.error('This run has no manifest to validate against.');
    const composed = await composeSelection();
    if (!composed) return;
    onComposedGuardrailsChange(composed.guardrails_toml ?? undefined);
    try {
      const job = await submit({
        manifest_id: manifestId,
        driver: 'service',
        validate_only: true,
        replay_hitlog_fileset: hitlogFileset,
        source_run: runName,
        ...(composed.guardrails_toml ? { defense_guardrails: composed.guardrails_toml } : {}),
        ...(composed.policy_yaml ? { defense_policy: composed.policy_yaml } : {}),
      });
      onSanityJobChange(job);
      toast.success('Sanity check started — replaying attacks against your selection…');
    } catch {
      toast.error('Failed to start the sanity check.');
    }
  };

  const applyGuardrails = async (): Promise<boolean> => {
    if (!effectiveComposedGuardrails) return false;
    try {
      await applyMitigation.mutateAsync({
        workspace,
        name: runName,
        data: { guardrails_toml: effectiveComposedGuardrails },
      });
      return true;
    } catch {
      return false;
    }
  };

  const busy = compose.isPending || submitting;
  const coverage = total ? Math.round((selectedIds.length / total) * 100) : 0;

  return (
    <Stack gap="density-xl">
      {/* Overview */}
      <Card className="p-4">
        <Stack gap="density-sm">
          <Text kind="body/semibold/lg">Review &amp; apply defenses</Text>
          <Text kind="body/regular/sm" className="text-subtle">
            {total} defense{total === 1 ? '' : 's'} generated across {groups.length} tool
            {groups.length === 1 ? '' : 's'} from this run&apos;s attacks. Keep the ones you want,
            sanity-check the selection, then apply to the agent.
          </Text>
          {/* Segmented coverage meter: one cell per defense, filled for the selected count. */}
          <div className="flex h-2 w-full gap-0.5" aria-label={`${coverage}% of defenses selected`}>
            {defenses.map((d, i) => (
              <div
                key={d.id}
                className="flex-1 rounded-sm bg-surface-raised transition-colors"
                style={i < selectedIds.length ? { backgroundColor: ACCENT.green } : undefined}
              />
            ))}
          </div>
          <Flex justify="between" align="center">
            <Text kind="body/regular/sm" className="text-primary">
              {selectedIds.length} of {total} selected
            </Text>
            <Flex gap="density-sm">
              <Button
                kind="secondary"
                size="small"
                onClick={() => setSelected(new Set(defenses.map((d) => d.id)))}
              >
                All
              </Button>
              <Button kind="secondary" size="small" onClick={() => setSelected(new Set())}>
                None
              </Button>
            </Flex>
          </Flex>
        </Stack>
      </Card>

      {/* Grouped, expandable defense rows */}
      <Card className="p-0">
        {groups.map((group, gi) => {
          const allOn = group.items.every((i) => selected.has(i.id));
          return (
            <div key={group.tool} className={gi > 0 ? 'border-t border-base' : ''}>
              <Flex align="center" justify="between" className="bg-surface-sunken px-3 py-2">
                <Flex align="center" gap="density-sm">
                  <Text kind="body/semibold/sm" className="font-mono text-primary">
                    {group.tool}
                  </Text>
                  <Badge color="gray">{group.items.length}</Badge>
                </Flex>
                <Switch
                  checked={allOn}
                  onCheckedChange={() => toggleGroup(group.items)}
                  aria-label={`Toggle all ${group.tool}`}
                />
              </Flex>
              <div>
                {group.items.map((defense) => (
                  <DefenseRow
                    key={defense.id}
                    defense={defense}
                    checked={selected.has(defense.id)}
                    onToggle={() => toggle(defense.id)}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </Card>

      {/* Action bar */}
      <Flex
        justify="between"
        align="center"
        className="sticky bottom-0 border-t border-base bg-surface-base py-3"
      >
        <Text kind="body/regular/sm" className="text-primary">
          {selectedIds.length} of {total} selected
        </Text>
        <Flex gap="density-sm" align="center">
          {!hitlogFileset ? (
            <Text kind="body/regular/xs" className="text-subtle">
              No recorded attacks to replay
            </Text>
          ) : null}
          <Button kind="secondary" onClick={previewComposed} disabled={busy}>
            Preview composed
          </Button>
          <Button kind="primary" onClick={runSanityCheck} disabled={busy || !hitlogFileset}>
            ▶ Run sanity check
          </Button>
        </Flex>
      </Flex>

      {preview?.guardrails && mitigations?.guardrails ? (
        <AccordionRoot multiple defaultValue={['preview']}>
          <AccordionSection value="preview" title="Composed guardrails (your selection)">
            <ConfigDiff
              before={mitigations.guardrails.before}
              after={preview.guardrails}
              language="toml"
            />
          </AccordionSection>
        </AccordionRoot>
      ) : null}

      {effectiveSanityJob ? (
        <Stack gap="density-md">
          <Text kind="body/semibold/lg">Sanity check</Text>
          {report ? (
            <>
              <SanityCheckReport report={report} />
              <Flex justify="end" align="center" gap="density-sm" className="min-w-0">
                {isProjectSource ? (
                  <Text kind="body/regular/sm" className="text-subtle">
                    This manifest brings its own image, so there is no registered agent to update —
                    copy the guardrails into your project instead.
                  </Text>
                ) : null}
                <Button
                  kind="primary"
                  size="small"
                  onClick={() => setConfirmOpen(true)}
                  disabled={
                    !canApplyMitigation({
                      hasComposedGuardrails: Boolean(effectiveComposedGuardrails),
                      hasManifestId: Boolean(manifestId),
                      manifestQuerySucceeded: manifestQuery.isSuccess,
                      isProjectSource,
                    })
                  }
                >
                  Apply to Agent
                </Button>
              </Flex>
            </>
          ) : missingReport ? (
            <Flex align="center" gap="density-sm" className="p-4">
              <Text kind="body/regular/md" className="text-feedback-danger">
                The sanity check finished without producing a report — it may have failed or been
                cancelled. Try running it again.
              </Text>
            </Flex>
          ) : (
            <Flex align="center" gap="density-sm" className="p-4">
              <Spinner size="small" aria-label="Running sanity check" />
              <Text kind="body/regular/md" className="text-subtle">
                {reportLoading
                  ? 'Replaying attacks + benign requests against your selection…'
                  : 'Starting sanity check…'}
              </Text>
            </Flex>
          )}
        </Stack>
      ) : null}

      <ConfirmationModal
        onNotify={notify}
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        onConfirm={applyGuardrails}
        title={agentName ? `Apply selected defenses to ${agentName}?` : 'Apply selected defenses?'}
        description="This records your selected guardrails on the agent as Relay components. Redeploy the agent afterward to activate them."
        submitButtonText="Apply"
        successText="Applied. Redeploy the agent to activate the guardrails."
        errorText="Could not apply the selected defenses to the agent."
      />
    </Stack>
  );
};
