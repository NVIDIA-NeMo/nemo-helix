// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import type {
  EvaluatorTaskDefinition,
  HarborTaskDefinition,
} from '@nemo/sdk/generated/evals/schema';
import { Anchor, CodeSnippet, SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import {
  configSection,
  formatValue,
  getFilesetRoute,
  parseFilesRef,
  parseTaskRef,
  RUNNER_LABELS,
} from '@studio/routes/evaluation/tasksets/helpers';
import type { TasksetTaskRow } from '@studio/routes/evaluation/tasksets/types';
import { useResolvedTasks } from '@studio/routes/evaluation/tasksets/useResolvedTasks';
import { type FC, type ReactNode, useMemo } from 'react';
import { Link } from 'react-router';

const Section: FC<{ heading: string; children: ReactNode }> = ({ heading, children }) => (
  <Stack gap="density-sm">
    <Text kind="label/bold/md">{heading}</Text>
    {children}
  </Stack>
);

const Json: FC<{ value: unknown }> = ({ value }) => (
  <CodeSnippet kind="block" language="json" value={JSON.stringify(value, null, 2)} />
);

const ConfigGrid: FC<{ section: Record<string, unknown>; empty: string }> = ({
  section,
  empty,
}) => {
  const entries = Object.entries(section);
  if (entries.length === 0) return <Text className="text-subtle">{empty}</Text>;
  return (
    <KeyValueGrid
      items={entries.map(([key, value]) => ({ key, label: key, value: formatValue(value) }))}
    />
  );
};

const HarborDetails: FC<{ spec: HarborTaskDefinition }> = ({ spec }) => {
  const archive = parseFilesRef(spec.source.fileset_ref);
  return (
    <>
      <Section heading="Instruction">
        <Text className="whitespace-pre-wrap">{spec.instruction || 'No instruction.'}</Text>
      </Section>
      <Section heading="Environment">
        <ConfigGrid section={configSection(spec.config, 'environment')} empty="Harbor defaults." />
      </Section>
      <Section heading="Tests">
        <ConfigGrid section={configSection(spec.config, 'verifier')} empty="Harbor defaults." />
        <Text className="text-subtle">
          The test scripts ship in the task archive at{' '}
          <Anchor asChild>
            <Link to={getFilesetRoute(archive)} className="break-all">
              {spec.source.fileset_ref}
            </Link>
          </Anchor>
          .
        </Text>
      </Section>
    </>
  );
};

const InlineDetails: FC<{ spec: EvaluatorTaskDefinition }> = ({ spec }) => (
  <>
    <Section heading="Intent">
      <Text className="whitespace-pre-wrap">{spec.intent}</Text>
    </Section>
    <Section heading="Inputs">
      <Json value={spec.inputs ?? {}} />
    </Section>
    <Section heading="Reference">
      {spec.reference ? (
        <Json value={spec.reference} />
      ) : (
        <Text className="text-subtle">None.</Text>
      )}
    </Section>
  </>
);

interface TaskDetailDrawerProps {
  /** The member ref to show; the drawer is closed when unset. */
  taskRef?: string;
  /** Workspace a bare ref resolves against: the taskset's. */
  workspace: string;
  onClose: () => void;
}

/** A member task's definition, rendered by the runner that executes it. Resolves its own task so
 *  a `?task=` deep link opens whatever page of the table is showing. */
export const TaskDetailDrawer: FC<TaskDetailDrawerProps> = ({ taskRef, workspace, onClose }) => {
  const requested = useMemo<TasksetTaskRow[]>(
    () => (taskRef ? [{ id: taskRef, ref: taskRef, ...parseTaskRef(taskRef, workspace) }] : []),
    [taskRef, workspace]
  );
  const { rows, isFetching, error } = useResolvedTasks(requested);
  const row = rows[0];
  const task = row?.task;

  const renderBody = () => {
    if (!row) return null;
    if (isFetching && !task) return <Text className="text-subtle">Loading task...</Text>;
    if (error) return <Text className="text-feedback-danger">{getErrorMessage(error)}</Text>;
    if (!task) {
      return (
        <Text className="text-subtle">
          {row.ref} could not be loaded. The task may have been deleted.
        </Text>
      );
    }
    return (
      <Stack gap="density-xl">
        <KeyValueGrid
          items={[
            { key: 'runner', label: 'Runner', value: RUNNER_LABELS[task.spec.kind] },
            { key: 'revision', label: 'Revision', value: `r${task.revision}` },
            { key: 'ref', label: 'Task Ref', value: row.ref, wrapValue: true },
          ]}
        />
        {task.spec.kind === 'harbor' ? (
          <HarborDetails spec={task.spec} />
        ) : (
          <InlineDetails spec={task.spec} />
        )}
      </Stack>
    );
  };

  return (
    <SidePanel
      open={Boolean(row)}
      onOpenChange={(open) => !open && onClose()}
      side="right"
      bordered
      modal
      className="w-[520px]"
      slotHeading={row?.name}
    >
      {renderBody()}
    </SidePanel>
  );
};
