// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Shield } from '@nemo/icons/icons';
import { AccordionRoot, Text } from '@nvidia/foundations-react-core';
import {
  piiColumnGroups,
  piiHeadline,
  piiReplayNote,
  plural,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import { DatasetLegend } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/DatasetLegend';
import {
  AttributeInferenceBars,
  MembershipInferencePie,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/PrivacyCharts';
import { WindowMembershipCharts } from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/WindowMembershipCharts';
import {
  MetricCard,
  MetricGroup,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/MetricCard';
import { PiiReplacementDetails } from '@studio/routes/SafeSynthesizerJobReportRoute/components/PiiReplacementDetails';
import {
  SummaryLines,
  TimeSeriesNotes,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/TimeSeriesNotes';
import {
  METRIC_COPY,
  METRIC_GROUPS,
  PII_COPY,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type {
  EvaluationReportData,
  PIIReplacement,
  PIIReplayMetric,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { windowMembershipSummary } from '@studio/routes/SafeSynthesizerJobReportRoute/timeSeriesSummary';
import { type FC, useMemo } from 'react';

const Subheading: FC<{ children: string }> = ({ children }) => (
  <Text kind="body/regular/sm" className="text-secondary">
    {children}
  </Text>
);

const PiiReplacementCard: FC<{ pii: PIIReplacement; replay: PIIReplayMetric | null }> = ({
  pii,
  replay,
}) => {
  const groups = useMemo(() => piiColumnGroups(pii, replay), [pii, replay]);
  const replayNote = piiReplayNote(replay);
  return (
    <MetricCard
      id="pii-replay"
      copy={{
        name: PII_COPY.title,
        description: pii.replaced ? PII_COPY.descriptionReplaced : PII_COPY.descriptionOff,
        tooltip: '',
      }}
      stat={piiHeadline(pii, groups, replayNote)}
    >
      <PiiReplacementDetails pii={pii} groups={groups} replayNote={replayNote} />
    </MetricCard>
  );
};

/** The library's Protection Metrics panel, or only the PII card when no privacy score was computed. */
export const DataPrivacyMetrics: FC<{ data: EvaluationReportData; withPrivacyScore: boolean }> = ({
  data,
  withPrivacyScore,
}) => {
  const { metrics, pii_replacement: pii } = data;
  const mia = metrics.membership_inference_protection;
  const aia = metrics.attribute_inference_protection;
  const windowMia = metrics.window_membership_inference_protection;
  const piiCard = pii ? <PiiReplacementCard pii={pii} replay={metrics.pii_replay} /> : null;

  if (!withPrivacyScore) {
    return piiCard ? <AccordionRoot multiple>{piiCard}</AccordionRoot> : null;
  }

  return (
    <MetricGroup
      title={METRIC_GROUPS.protection.title}
      description={METRIC_GROUPS.protection.description}
      icon={<Shield />}
    >
      <MetricCard
        id="mia"
        copy={METRIC_COPY.membership_inference_protection}
        score={mia?.score ?? null}
      >
        {mia?.outcomes.length ? (
          <>
            <Subheading>Breakdown of protection levels across simulated attacks.</Subheading>
            {mia.excluded_columns.length ? (
              <Subheading>
                {`${plural(mia.excluded_columns.length, 'column')} excluded from this metric: ${mia.excluded_columns.join(', ')}`}
              </Subheading>
            ) : null}
            <MembershipInferencePie metric={mia} />
          </>
        ) : null}
      </MetricCard>
      {windowMia ? (
        <MetricCard
          id="window-group-mia"
          copy={METRIC_COPY.window_membership_inference_protection}
          score={windowMia.score}
        >
          <TimeSeriesNotes score={windowMia.score} details={{}} />
          <SummaryLines lines={windowMembershipSummary(windowMia.details)} />
          <DatasetLegend
            datasets={
              windowMia.details.attack
                ? ['training', 'synthetic', 'holdout']
                : ['training', 'synthetic']
            }
          />
          <WindowMembershipCharts details={windowMia.details} />
        </MetricCard>
      ) : null}
      <MetricCard
        id="aia"
        copy={METRIC_COPY.attribute_inference_protection}
        score={aia?.score ?? null}
      >
        {aia?.columns.length ? (
          <>
            <Subheading>Breakdown of protection across all evaluated columns.</Subheading>
            <AttributeInferenceBars metric={aia} />
          </>
        ) : null}
      </MetricCard>
      {piiCard}
    </MetricGroup>
  );
};
