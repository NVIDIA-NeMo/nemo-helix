// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormModal } from '@nemo/common/src/components/FormModal';
import { Checkbox, CodeSnippet, Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import type {
  AgentSandbox,
  NetworkRequest,
} from '@studio/routes/agents/AgentDetailRoute/network/types';
import {
  agentYamlFragment,
  binaryName,
  isFinished,
  requestTarget,
} from '@studio/routes/agents/AgentDetailRoute/network/utils';
import { type FC, useState } from 'react';

export type DecisionKind = 'approve' | 'reject' | 'remember';

export interface PendingDecision {
  kind: DecisionKind;
  sandbox: AgentSandbox;
  request: NetworkRequest;
}

interface DecideRequestModalProps {
  agentName: string;
  decision: PendingDecision;
  loading: boolean;
  errorText?: string | null;
  onClose: () => void;
  onApprove: (remember: boolean) => void;
  onReject: (options: { reason: string; forget: boolean }) => void;
}

const TITLE: Record<DecisionKind, (target: string) => string> = {
  approve: (target) => `Allow ${target}?`,
  reject: (target) => `Keep ${target} blocked?`,
  remember: (target) => `Remember ${target} for this agent?`,
};

const SUBMIT: Record<DecisionKind, string> = {
  approve: 'Allow',
  reject: 'Keep blocked',
  remember: 'Remember',
};

/**
 * Confirms a decision on one network request and offers the matching change to the agent:
 * remembering the rule on approval, dropping a remembered rule on rejection.
 */
export const DecideRequestModal: FC<DecideRequestModalProps> = ({
  agentName,
  decision,
  loading,
  errorText,
  onClose,
  onApprove,
  onReject,
}) => {
  const { kind, sandbox, request } = decision;
  const [remember, setRemember] = useState(true);
  const [forget, setForget] = useState(request.remembered);
  const [reason, setReason] = useState('');
  const target = requestTarget(request);
  const workload = `${sandbox.kind} ${sandbox.name}`;
  const process = binaryName(request.binary) || 'A process';
  const showsRule = kind === 'remember' || (kind === 'approve' && remember);
  const finished = isFinished(sandbox);

  return (
    <FormModal
      open
      title={TITLE[kind](target)}
      submitButtonText={SUBMIT[kind]}
      loading={loading}
      disabled={loading}
      errorText={errorText}
      onClose={onClose}
      onSubmit={(event) => {
        event.preventDefault();
        if (kind === 'reject') {
          onReject({ reason, forget });
        } else {
          onApprove(kind === 'remember' || remember);
        }
      }}
      attributes={kind === 'reject' ? { SubmitButton: { color: 'danger' } } : undefined}
    >
      <Stack gap="density-lg">
        {kind === 'approve' && (
          <Text kind="body/regular/sm">
            {process} in {workload} was blocked from {target}.{' '}
            {finished
              ? 'That sandbox has finished, so allowing it only helps later ones: remember the rule and run the agent again.'
              : 'Allowing it lets this sandbox make the connection from now on.'}
          </Text>
        )}
        {kind === 'remember' && (
          <Text kind="body/regular/sm">
            {workload} is already allowed to reach {target}. Remembering it adds the rule to{' '}
            {agentName}, so new deployments and runs start with it.
          </Text>
        )}
        {kind === 'reject' && (
          <>
            <Text kind="body/regular/sm">
              {process} in {workload} stays blocked from {target}. The gateway does not ask again
              for a rejected destination in this sandbox.
            </Text>
            <TextInput
              value={reason}
              placeholder="Reason (optional)"
              aria-label="Reason"
              onChange={(event) => setReason(event.target.value)}
            />
            {request.remembered && (
              <Checkbox
                checked={forget}
                onCheckedChange={(checked) => setForget(checked === true)}
                slotLabel={`Also remove this rule from ${agentName}, so new sandboxes are blocked too`}
              />
            )}
          </>
        )}
        {kind === 'approve' && (
          <Checkbox
            checked={remember}
            onCheckedChange={(checked) => setRemember(checked === true)}
            slotLabel={
              <Stack gap="density-xxs">
                <Text kind="body/bold/sm">Remember for {agentName}</Text>
                <Text kind="body/regular/xs" className="text-secondary">
                  New deployments and runs of this agent start with the rule, so they are not
                  blocked again.
                </Text>
              </Stack>
            }
          />
        )}
        {showsRule && (
          <Stack gap="density-xs">
            <Text kind="body/regular/xs" className="text-secondary">
              Saved on the agent in Helix. To keep it in the agent's source too, add this to
              agent.yaml:
            </Text>
            <CodeSnippet
              value={agentYamlFragment(request.rule_name, request.rule)}
              language="yaml"
              kind="block"
            />
          </Stack>
        )}
      </Stack>
    </FormModal>
  );
};
