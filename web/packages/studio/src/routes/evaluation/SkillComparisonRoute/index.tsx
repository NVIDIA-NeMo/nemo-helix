// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import {
  Badge,
  Button,
  Flex,
  Modal,
  PageHeader,
  Stack,
  Table,
  Tabs,
  Text,
} from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import {
  ArmEvidence,
  MetricComparison,
  MetricEvidence,
  TokenComparison,
  TrialHeatmap,
} from '@studio/routes/evaluation/SkillComparisonRoute/ComparisonResults';
import {
  ARM_NAMES,
  comparisonWarnings,
  readReport,
  type ComparisonReport,
} from '@studio/routes/evaluation/SkillComparisonRoute/report';
import { useRef, useState } from 'react';

export const SkillComparisonRoute = () => {
  const workspace = useWorkspaceFromPath();
  const [imported, setImported] = useState<{ report: ComparisonReport; filename: string } | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [metric, setMetric] = useState<string | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const request = useRef(0);
  useBreadcrumbs({ items: [{ slotLabel: 'Skill Comparisons' }] });

  const importFile = async (file: File) => {
    const current = ++request.current;
    setLoading(true);
    setError(null);
    try {
      const report = await readReport(file);
      if (request.current !== current) return;
      setImported({ report, filename: file.name });
      setMetric(null);
    } catch (reason) {
      if (request.current === current)
        setError(
          reason instanceof Error ? reason.message : 'Could not read the comparison report.'
        );
    } finally {
      if (request.current === current) setLoading(false);
    }
  };
  const report = imported?.report;
  const warnings = report ? comparisonWarnings(report) : [];
  const incomplete =
    !!report &&
    (!!report.error ||
      warnings.length > 0 ||
      Object.values(report.arms).some(
        (arm) =>
          arm.evidence_error ||
          arm.replay_error ||
          !arm.replay ||
          arm.replay.summary.error_count > 0
      ));

  return (
    <AccessibleTitle title="Skill Comparisons">
      <div className="h-full overflow-auto">
        <Stack gap="density-xl" padding="density-2xl" className="mx-auto min-w-0 w-full max-w-7xl">
          <PageHeader
            className="p-0"
            slotHeading={
              <Flex gap="density-md" align="center" className="flex-wrap">
                Skill Comparisons{' '}
                {report && (
                  <Badge color={incomplete ? 'yellow' : 'gray'}>
                    {incomplete ? 'Needs Review' : 'Imported Report'}
                  </Badge>
                )}
              </Flex>
            }
            slotDescription="Compare baseline and with-skill evaluations, then inspect the measurements and evidence behind each score."
            slotActions={
              <Button
                color="brand"
                kind={report ? 'primary' : 'secondary'}
                onClick={() => picker.current?.click()}
              >
                {loading ? 'Reading Report…' : report ? 'Replace Report' : 'Import Report'}
              </Button>
            }
          />
          <input
            ref={picker}
            type="file"
            accept=".json,application/json"
            aria-label="Comparison report"
            className="sr-only"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = '';
              if (file) void importFile(file);
            }}
          />
          <Text kind="body/regular/sm" className="text-secondary">
            Import skill-eval-summary.json from the Harbor Evals skill example. The report stays in
            this browser tab; importing it does not upload data or start an evaluation.
          </Text>
          {error && (
            <div role="alert">
              <ErrorPanel title="Could Not Import Report" errorMessage={error} />
            </div>
          )}
          {loading && <Text role="status">Reading comparison report…</Text>}
          {!report && !loading && !error && (
            <EntityEmptyState
              entity="skillComparisons"
              variant="first-use"
              onCreate={() => picker.current?.click()}
            />
          )}
          {report && (
            <>
              <KeyValueGrid
                items={[
                  { key: 'file', label: 'Report', value: imported.filename },
                  {
                    key: 'skill',
                    label: 'Skill Revision',
                    value: report.skill_ref,
                    wrapValue: true,
                  },
                  {
                    key: 'taskset',
                    label: 'Taskset Revision',
                    value: report.taskset_ref,
                    wrapValue: true,
                  },
                  { key: 'context', label: 'Source Context', value: report.context },
                  { key: 'workspace', label: 'Source Workspace', value: report.workspace },
                ]}
              />
              {warnings.length > 0 && (
                <DetailPanel title="Comparison Needs Review">
                  <Stack gap="density-sm">
                    {warnings.map((warning) => (
                      <Text key={warning} role="status">
                        {warning}
                      </Text>
                    ))}
                  </Stack>
                </DetailPanel>
              )}
              {report.error && (
                <Text role="status" className="text-error">
                  Scoring error: {report.error}
                </Text>
              )}
              <div className="grid grid-cols-1 gap-density-lg md:grid-cols-2">
                {ARM_NAMES.map((name) => (
                  <DetailPanel key={name} title={name === 'baseline' ? 'Baseline' : 'With Skill'}>
                    <Stack gap="density-sm">
                      <Text kind="title/lg" className="tabular-nums">
                        {report.arms[name].reward.toFixed(2)}
                      </Text>
                      <Text>Live Harbor reward · {report.arms[name].run_id}</Text>
                      <Text kind="body/regular/sm" className="text-secondary">
                        Skill activation:{' '}
                        {report.arms[name].activation == null
                          ? 'Unmeasured'
                          : report.arms[name].activation
                            ? 'Confirmed'
                            : 'Not Activated'}
                      </Text>
                    </Stack>
                  </DetailPanel>
                ))}
              </div>
              <Text kind="body/regular/sm" className="text-secondary">
                Each arm is one live trial. Differences describe these observations; the report does
                not establish statistical significance. Live rewards, replay metrics, and package
                checks are separate measurements.
              </Text>
              <Tabs
                className="min-w-0 w-full"
                key={imported.filename + report.arms.baseline.run_id}
                defaultValue="overview"
                items={[
                  {
                    value: 'overview',
                    children: 'Overview',
                    slotContent: (
                      <Stack gap="density-xl" className="pt-4">
                        <MetricComparison report={report} onInspect={setMetric} />
                        <TokenComparison report={report} />
                      </Stack>
                    ),
                  },
                  {
                    value: 'trials',
                    children: 'Task Results',
                    slotContent: (
                      <div className="pt-4">
                        <TrialHeatmap report={report} />
                      </div>
                    ),
                  },
                  {
                    value: 'package',
                    children: 'Package Checks',
                    slotContent: (
                      <Stack gap="density-lg" className="pt-4">
                        <Text kind="body/regular/sm" className="text-secondary">
                          These checks run once on the pinned skill archive. They do not measure the
                          difference between trial arms.
                        </Text>
                        {report.package_metrics ? (
                          Object.entries(report.package_metrics).map(([name, result]) => (
                            <DetailPanel key={name} title={name}>
                              <Stack gap="density-md">
                                <Table
                                  aria-label={`${name} outputs`}
                                  columns={['Output', 'Value']}
                                  rows={result.outputs.map((output) => ({
                                    id: output.name,
                                    cells: [
                                      output.name,
                                      {
                                        children: (
                                          <pre className="whitespace-pre-wrap break-all text-sm">
                                            {JSON.stringify(output.value, null, 2)}
                                          </pre>
                                        ),
                                      },
                                    ],
                                  }))}
                                />
                                {result.diagnostics.map((diagnostic, index) => (
                                  <Text key={`${index}-${diagnostic.message}`}>
                                    {diagnostic.message}
                                  </Text>
                                ))}
                              </Stack>
                            </DetailPanel>
                          ))
                        ) : (
                          <Text>No package checks were retained.</Text>
                        )}
                      </Stack>
                    ),
                  },
                  {
                    value: 'evidence',
                    children: 'Evidence',
                    slotContent: (
                      <Stack gap="density-lg" className="pt-4">
                        <Text className="text-secondary" kind="body/regular/sm">
                          Report context: {report.context}. Job links open this Helix deployment in
                          workspace {report.workspace}.
                          {workspace !== report.workspace &&
                            ` You are currently browsing workspace ${workspace}.`}{' '}
                          Local bundle paths identify artifacts retained by the CLI; importing the
                          report does not import those bundles.
                        </Text>
                        {ARM_NAMES.map((name) => (
                          <ArmEvidence
                            key={name}
                            name={name}
                            arm={report.arms[name]}
                            workspace={report.workspace}
                          />
                        ))}
                        <DetailPanel title="Original Scoring Records" defaultCollapsed>
                          <pre className="whitespace-pre-wrap break-all text-sm max-h-[32rem] overflow-auto">
                            {JSON.stringify(report, null, 2)}
                          </pre>
                        </DetailPanel>
                      </Stack>
                    ),
                  },
                ]}
              />
              <Modal
                open={metric !== null}
                onOpenChange={(open) => {
                  if (!open) setMetric(null);
                }}
                slotHeading={metric ?? 'Metric Evidence'}
                className="max-w-4xl w-[90vw]"
              >
                {metric && (
                  <div className="max-h-[65vh] overflow-auto">
                    <MetricEvidence report={report} metric={metric} />
                  </div>
                )}
              </Modal>
            </>
          )}
        </Stack>
      </div>
    </AccessibleTitle>
  );
};
