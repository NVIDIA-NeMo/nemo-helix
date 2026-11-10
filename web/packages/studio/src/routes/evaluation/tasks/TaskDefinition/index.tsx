// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { MarkdownContent } from '@nemo/common/src/components/MarkdownContent';
import type {
  EvaluatorTaskDefinition,
  HarborTaskDefinition,
  Task,
} from '@nemo/sdk/generated/evals/schema';
import { Anchor, CodeSnippet, Stack, Text } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import {
  configSection,
  formatValue,
  getFilesetRoute,
  parseFilesRef,
} from '@studio/routes/evaluation/tasksets/helpers';
import type { FC } from 'react';
import { Link } from 'react-router';

const Json: FC<{ value: unknown }> = ({ value }) => (
  <CodeSnippet kind="block" language="json" value={JSON.stringify(value, null, 2)} />
);

const Prose: FC<{ text?: string; empty: string }> = ({ text, empty }) =>
  text ? (
    <Text className="whitespace-pre-wrap">{text}</Text>
  ) : (
    <Text className="text-subtle">{empty}</Text>
  );

/** A `task.toml` section as label/value rows. */
const ConfigPairs: FC<{ section: Record<string, unknown>; empty: string }> = ({
  section,
  empty,
}) => {
  const entries = Object.entries(section);
  if (entries.length === 0) return <Text className="text-subtle">{empty}</Text>;
  return (
    <Stack gap="3">
      {entries.map(([key, value]) => (
        <KVPair
          key={key}
          label={key}
          value={formatValue(value)}
          size="medium"
          orientation="horizontal"
        />
      ))}
    </Stack>
  );
};

const HarborPanels: FC<{ spec: HarborTaskDefinition }> = ({ spec }) => (
  <>
    <DetailPanel title="Instruction">
      {spec.instruction ? (
        <MarkdownContent content={spec.instruction} disableImages />
      ) : (
        <Prose empty="No instruction." />
      )}
    </DetailPanel>
    <DetailPanel title="Environment">
      <ConfigPairs section={configSection(spec.config, 'environment')} empty="Harbor defaults." />
    </DetailPanel>
    <DetailPanel title="Tests">
      <Stack gap="density-lg">
        <ConfigPairs section={configSection(spec.config, 'verifier')} empty="Harbor defaults." />
        <Text className="text-subtle">
          The test scripts ship in the task archive at{' '}
          <Anchor asChild>
            <Link
              to={getFilesetRoute(parseFilesRef(spec.source.fileset_ref))}
              className="break-all"
            >
              {spec.source.fileset_ref}
            </Link>
          </Anchor>
          .
        </Text>
      </Stack>
    </DetailPanel>
  </>
);

const InlinePanels: FC<{ spec: EvaluatorTaskDefinition }> = ({ spec }) => (
  <>
    <DetailPanel title="Intent">
      <Prose text={spec.intent} empty="No intent." />
    </DetailPanel>
    <DetailPanel title="Inputs">
      <Json value={spec.inputs ?? {}} />
    </DetailPanel>
    <DetailPanel
      title="Reference"
      slotAction={
        <Text kind="body/regular/sm" className="text-secondary">
          Grader only
        </Text>
      }
    >
      {spec.reference ? <Json value={spec.reference} /> : <Prose empty="None." />}
    </DetailPanel>
  </>
);

/** A task's content as stacked panels, laid out by the runner that executes it. */
export const TaskDefinition: FC<{ task: Task }> = ({ task: { spec } }) => (
  <Stack gap="density-2xl">
    {spec.kind === 'harbor' ? <HarborPanels spec={spec} /> : <InlinePanels spec={spec} />}
  </Stack>
);
