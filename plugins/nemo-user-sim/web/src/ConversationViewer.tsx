// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Anchor,
  Button,
  Card,
  Flex,
  Modal,
  Stack,
  Tag,
  Text,
  TextArea,
} from '@nvidia/foundations-react-core';
import { useState, type ReactNode } from 'react';
import { useAnnotate, useFeedback } from './api';
import { useHost } from './host';
import { getIntakeSessionRoute } from './paths';
import { axisLabel, otherFailures, pickText, words, type Row } from './report';
import { Transcript } from './Transcript';
import type { ChatMessage, Probe, Transcripts } from './types';

/** Who did what in a simulated conversation, so a reader knows the agent's side is real. */
const RealAndSimulated = ({
  agent,
  persona,
  probePrompt,
  ownTools,
}: {
  agent: string;
  persona: string;
  probePrompt: boolean;
  ownTools: boolean;
}) => (
  <Text kind="body/regular/xs" className="text-subtle">
    {probePrompt
      ? `Real: ${agent}'s model, replies and tool calls, under the probe's system prompt shown below. `
      : `Real: ${agent}, with its own prompt, model and tool calls. `}
    {ownTools
      ? `Its own tools' results are real too. Simulated: ${persona}.`
      : `Simulated: ${persona} and the tool results.`}
  </Text>
);

const Field = ({ label, children }: { label: string; children: ReactNode }) => (
  <Stack gap="0">
    <Text kind="body/semibold/xs" className="text-subtle">
      {label}
    </Text>
    {children}
  </Stack>
);

/** What the conversation tested and how it came out, above the transcript. */
const WhatWasTested = ({
  row,
  probe,
  messages,
}: {
  row: Row;
  probe: Probe | undefined;
  messages: ChatMessage[] | undefined;
}) => {
  const asked = messages?.find((message) => message.role === 'user')?.content;
  const failed = otherFailures(row);
  const [why, setWhy] = useState(false);
  return (
    <Card density="compact">
      <Stack gap="density-sm">
        <Text kind="body/semibold/sm">
          What this conversation tested{probe ? `: ${probe.label}` : ''}
        </Text>
        {asked ? (
          <Field label="The user asked">
            <Text kind="body/regular/sm">{asked}</Text>
          </Field>
        ) : null}
        {row.tested ? (
          <>
            {row.tested.expected ? (
              <Field label="What the agent should do">
                <Text kind="body/regular/sm">{row.tested.expected}</Text>
              </Field>
            ) : null}
            <Flex gap="density-md" wrap="wrap">
              {row.tested.should_call ? (
                <Field label="Should call">
                  <Tag kind="solid" color="green" density="compact" readOnly>
                    {row.tested.should_call}
                  </Tag>
                </Field>
              ) : null}
              {row.tested.should_not_call ? (
                <Field label="Shouldn't call">
                  <Tag kind="solid" color="red" density="compact" readOnly>
                    {row.tested.should_not_call}
                  </Tag>
                </Field>
              ) : null}
            </Flex>
          </>
        ) : probe ? (
          <Field label="What this probe looks for">
            <Text kind="body/regular/sm">{probe.tests}</Text>
          </Field>
        ) : null}
        <Field label="Result">
          {row.passed === null ? (
            <Text kind="body/regular/sm">Not scored.</Text>
          ) : row.passed && !row.picked ? (
            <Text kind="body/regular/sm">Passed every check.</Text>
          ) : (
            <Stack gap="density-xs">
              <Text kind="body/regular/sm">
                <strong>
                  {[
                    ...(row.picked ? [pickText(row.picked)] : []),
                    ...(failed.length ? [`Failed: ${failed.map(axisLabel).join(', ')}`] : []),
                  ].join('. ')}
                  .
                </strong>
              </Text>
              {failed.length ? (
                <div>
                  <Button kind="tertiary" size="small" onClick={() => setWhy(!why)}>
                    {why ? 'Hide why' : 'Why'}
                  </Button>
                </div>
              ) : null}
              {why
                ? failed.map((axis) => (
                    <Text key={axis} kind="body/regular/sm">
                      <strong>{axisLabel(axis)}:</strong> {row.why[axis] ?? ''}
                    </Text>
                  ))
                : null}
            </Stack>
          )}
        </Field>
      </Stack>
    </Card>
  );
};

/**
 * One conversation at a time, with previous and next, and the review a training export reads: a thumbs
 * up or down and a note, saved on the conversation's Intake session as Studio's trace view saves them.
 */
export const ConversationViewer = ({
  rows,
  index,
  transcripts,
  outcome,
  agent,
  model,
  probes,
  onIndex,
  onClose,
}: {
  rows: Row[];
  index: number | null;
  transcripts: Transcripts | null | undefined;
  outcome: (row: Row) => string;
  /** The agent or model the run tested. */
  agent: string;
  model: string | undefined;
  probes: Map<string, Probe>;
  onIndex: (index: number) => void;
  onClose: () => void;
}) => {
  const { workspaceId, navigation, notifications } = useHost();
  const row = index === null ? null : rows[index];
  const reviewable = row?.status === 'ok';
  const feedback = useFeedback(reviewable ? row.trajectory_id : null);
  const annotate = useAnnotate();
  const [note, setNote] = useState('');
  if (!row || index === null) return null;
  const messages = transcripts?.[row.trajectory_id];
  const persona = row.persona_name ?? 'Simulated user';
  const intakeRoute = getIntakeSessionRoute(workspaceId, row.trajectory_id);

  const save = (annotation: Parameters<typeof annotate.mutate>[0], done: string) =>
    annotate.mutate(annotation, {
      onSuccess: () => notifications.notify(done, 'success'),
      onError: (error) => notifications.notify(`Couldn't save. ${error.message}`, 'error'),
    });

  return (
    <Modal
      open
      onOpenChange={(next) => !next && onClose()}
      className="w-[90vw] max-w-[1000px]"
      slotHeading={
        <Flex align="center" justify="between" className="w-full" gap="density-md">
          <Stack gap="0">
            <Text kind="body/semibold/md">{row.persona_name}</Text>
            <Text kind="body/regular/xs" className="text-subtle">
              {row.persona_age_bin} · {row.user_interaction_style} ·{' '}
              {probes.get(row.probe_type ?? '')?.label ?? words(row.probe_type)} · {outcome(row)}
            </Text>
          </Stack>
          <Flex align="center" gap="density-sm">
            <Button
              kind="secondary"
              size="small"
              disabled={index === 0}
              onClick={() => onIndex(index - 1)}
            >
              ‹
            </Button>
            <Text kind="body/regular/sm" className="text-subtle">
              {index + 1} of {rows.length}
            </Text>
            <Button
              kind="secondary"
              size="small"
              disabled={index === rows.length - 1}
              onClick={() => onIndex(index + 1)}
            >
              ›
            </Button>
          </Flex>
        </Flex>
      }
      slotFooter={
        reviewable ? (
          <Stack gap="density-sm" className="w-full">
            <Flex align="center" gap="density-sm">
              <Text kind="body/regular/sm">Use for training?</Text>
              <Button
                kind={feedback.data === 'positive' ? 'primary' : 'secondary'}
                size="small"
                onClick={() =>
                  save(
                    { session_id: row.trajectory_id, kind: 'feedback', value: 'positive' },
                    'Marked for training.'
                  )
                }
              >
                👍 Yes
              </Button>
              <Button
                kind={feedback.data === 'negative' ? 'primary' : 'secondary'}
                size="small"
                onClick={() =>
                  save(
                    { session_id: row.trajectory_id, kind: 'feedback', value: 'negative' },
                    'Left out of training.'
                  )
                }
              >
                👎 No
              </Button>
              <div className="flex-1" />
              <Anchor
                href={intakeRoute}
                onClick={(event) => {
                  event.preventDefault();
                  navigation.navigate(intakeRoute);
                }}
              >
                Open trace in Intake
              </Anchor>
            </Flex>
            <Flex gap="density-sm" align="end">
              <TextArea
                className="flex-1"
                rows={2}
                placeholder="Add a note for whoever reviews this next"
                value={note}
                onChange={(event) => setNote(event.target.value)}
              />
              <Button
                kind="secondary"
                disabled={!note.trim() || annotate.isPending}
                onClick={() => {
                  save(
                    { session_id: row.trajectory_id, kind: 'note', text: note.trim() },
                    'Note saved.'
                  );
                  setNote('');
                }}
              >
                Save note
              </Button>
            </Flex>
          </Stack>
        ) : null
      }
    >
      <div className="max-h-[70vh] overflow-auto">
        {messages ? (
          <Stack gap="density-md">
            <RealAndSimulated
              agent={model && model !== agent ? `${agent} (${model})` : agent}
              persona={persona}
              probePrompt={messages.some((message) => message.role === 'system')}
              ownTools={(row.own_tool_calls ?? []).length > 0}
            />
            <WhatWasTested key={row.trajectory_id} row={row} probe={probes.get(row.probe_type ?? '')} messages={messages} />
            <Transcript
              messages={messages}
              persona={persona}
              agent={agent}
              ownCalls={row.own_tool_calls ?? []}
            />
          </Stack>
        ) : (
          <Text kind="body/regular/sm" className="text-subtle">
            {row.status === 'running'
              ? 'This conversation is still running.'
              : 'No transcript was recorded for this conversation.'}
          </Text>
        )}
      </div>
    </Modal>
  );
};
