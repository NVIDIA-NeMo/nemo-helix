// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CopyButton } from '@nemo/common/src/components/CopyButton';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  Badge,
  Banner,
  Button,
  CodeSnippet,
  Flex,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import {
  DecideRequestModal,
  type PendingDecision,
} from '@studio/routes/agents/AgentDetailRoute/network/DecideRequestModal';
import { NetworkRequestRow } from '@studio/routes/agents/AgentDetailRoute/network/NetworkRequestRow';
import {
  useAgentNetworkPolicy,
  useDecideNetworkRequest,
} from '@studio/routes/agents/AgentDetailRoute/network/queries';
import type { AgentSandbox } from '@studio/routes/agents/AgentDetailRoute/network/types';
import {
  pendingCount,
  requestTarget,
  ruleBinaries,
  ruleEndpoints,
  sortRequests,
  sortSandboxes,
} from '@studio/routes/agents/AgentDetailRoute/network/utils';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { type FC, useState } from 'react';
import { Link } from 'react-router';

interface NetworkAccessTabProps {
  workspace: string;
  agentName: string;
  /** Polling stops while the tab is not shown. */
  active: boolean;
}

const KIND_LABEL: Record<AgentSandbox['kind'], string> = {
  deployment: 'Deployment',
  job: 'Run',
  sandbox: 'Sandbox',
};

const SandboxHeader: FC<{ workspace: string; sandbox: AgentSandbox; isFirst: boolean }> = ({
  workspace,
  sandbox,
  isFirst,
}) => (
  <Flex
    align="center"
    gap="2"
    className={`bg-surface-sunken px-4 py-2 ${isFirst ? '' : 'border-t border-base'}`}
  >
    <Badge kind="outline" color="gray" size="small">
      {KIND_LABEL[sandbox.kind]}
    </Badge>
    {sandbox.kind === 'job' ? (
      <Link to={getWorkspaceJobDetailRoute(workspace, sandbox.name)}>
        <Text kind="body/semibold/sm">{sandbox.name}</Text>
      </Link>
    ) : (
      <Text kind="body/semibold/sm">{sandbox.name}</Text>
    )}
    <Text kind="body/regular/xs" className="text-secondary">
      sandbox {sandbox.sandbox} · {sandbox.phase}
    </Text>
  </Flex>
);

/**
 * The network access of an agent's OpenShell sandboxes: what each running sandbox asked
 * for, and the rules the agent carries into new ones.
 */
export const NetworkAccessTab: FC<NetworkAccessTabProps> = ({ workspace, agentName, active }) => {
  const toast = useToast();
  const {
    data: policy,
    isPending,
    error,
  } = useAgentNetworkPolicy(workspace, agentName, {
    enabled: active,
  });
  const {
    mutate,
    isPending: isDeciding,
    error: decideError,
    reset,
  } = useDecideNetworkRequest(workspace, agentName);
  const [decision, setDecision] = useState<PendingDecision | null>(null);

  const closeModal = () => {
    setDecision(null);
    reset();
  };

  const sandboxes = sortSandboxes(policy?.sandboxes ?? []);
  const rules = Object.entries(policy?.rules ?? {});
  const pending = pendingCount(policy);

  return (
    <Stack gap="5" className="w-full">
      {error && (
        <Banner kind="inline" status="error">
          Could not load network access: {error.message}
        </Banner>
      )}
      {policy?.gateway_error && (
        <Banner kind="inline" status="warning">
          Could not reach the OpenShell gateway, so running sandboxes are not shown:{' '}
          {policy.gateway_error}
        </Banner>
      )}

      <DetailPanel
        title="Requests from running sandboxes"
        flush
        slotAction={
          pending > 0 ? (
            <Badge kind="solid" color="yellow" size="small">
              {pending} pending
            </Badge>
          ) : null
        }
      >
        {sandboxes.length === 0 ? (
          <Text kind="body/regular/sm" className="p-4 text-secondary">
            {isPending
              ? 'Loading…'
              : 'No sandbox is running this agent. When a deployment or run of it is blocked from a host, the request appears here.'}
          </Text>
        ) : (
          <Stack gap="0">
            {sandboxes.map((sandbox, index) => (
              <Stack gap="0" key={sandbox.sandbox}>
                <SandboxHeader workspace={workspace} sandbox={sandbox} isFirst={index === 0} />
                {sandbox.error && (
                  <Text
                    kind="body/regular/xs"
                    className="border-t border-base px-4 py-3 text-feedback-danger"
                  >
                    Could not read this sandbox's requests: {sandbox.error}
                  </Text>
                )}
                {!sandbox.error && sandbox.requests.length === 0 && (
                  <Text
                    kind="body/regular/xs"
                    className="border-t border-base px-4 py-3 text-secondary"
                  >
                    Nothing blocked.
                  </Text>
                )}
                {sortRequests(sandbox.requests).map((request) => (
                  <NetworkRequestRow
                    key={request.id}
                    request={request}
                    onDecide={(kind) => setDecision({ kind, sandbox, request })}
                  />
                ))}
              </Stack>
            ))}
          </Stack>
        )}
      </DetailPanel>

      <DetailPanel
        title="Remembered for this agent"
        flush
        slotAction={
          policy?.agent_yaml ? (
            <CopyButton kind="tertiary" size="small" text={policy.agent_yaml}>
              Copy for agent.yaml
            </CopyButton>
          ) : null
        }
      >
        <Stack gap="0">
          <Text kind="body/regular/xs" className="px-4 py-3 text-secondary">
            New deployments and runs of this agent start with these rules. Sandboxes that are
            already running keep what they have.
          </Text>
          {rules.length === 0 ? (
            <Text kind="body/regular/sm" className="border-t border-base px-4 py-3 text-secondary">
              Nothing remembered yet. Allow a request and choose to remember it.
            </Text>
          ) : (
            <>
              {rules.map(([name, rule]) => (
                <Flex key={name} align="center" gap="3" className="border-t border-base px-4 py-3">
                  <Stack gap="0" className="min-w-0 flex-1">
                    <Text kind="body/semibold/sm" className="font-mono">
                      {ruleEndpoints(rule).join(', ') || name}
                    </Text>
                    <Text kind="body/regular/xs" className="truncate text-secondary">
                      {ruleBinaries(rule).join(', ') || 'any process'} · {name}
                    </Text>
                  </Stack>
                  <Button
                    kind="tertiary"
                    size="small"
                    color="danger"
                    disabled={isDeciding}
                    onClick={() =>
                      mutate(
                        { action: 'forget', rule: name },
                        {
                          onSuccess: () => toast.success(`Removed ${name} from ${agentName}`),
                          onError: (failure) => toast.error(failure.message),
                        }
                      )
                    }
                  >
                    Remove
                  </Button>
                </Flex>
              ))}
              <div className="border-t border-base px-4 py-3">
                <CodeSnippet value={policy?.agent_yaml ?? ''} language="yaml" kind="block" />
              </div>
            </>
          )}
        </Stack>
      </DetailPanel>

      {decision && (
        <DecideRequestModal
          // A new request gets fresh checkbox defaults rather than the last one's.
          key={`${decision.kind}:${decision.request.id}`}
          agentName={agentName}
          decision={decision}
          loading={isDeciding}
          errorText={decideError?.message}
          onClose={closeModal}
          onApprove={(remember) =>
            mutate(
              {
                action: 'approve',
                sandbox: decision.sandbox.sandbox,
                requestId: decision.request.id,
                remember,
              },
              {
                onSuccess: () => {
                  const target = requestTarget(decision.request);
                  toast.success(
                    remember
                      ? `Allowed ${target} and remembered it for ${agentName}`
                      : `Allowed ${target}`
                  );
                  closeModal();
                },
              }
            )
          }
          onReject={({ reason, forget }) =>
            mutate(
              {
                action: 'reject',
                sandbox: decision.sandbox.sandbox,
                requestId: decision.request.id,
                reason,
                forget,
              },
              {
                onSuccess: () => {
                  toast.success(`${requestTarget(decision.request)} stays blocked`);
                  closeModal();
                },
              }
            )
          }
        />
      )}
    </Stack>
  );
};
