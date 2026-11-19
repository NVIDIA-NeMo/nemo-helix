// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRelativeTimeSince } from '@nemo/common/src/components/RelativeTime';
import { Badge, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { DecisionKind } from '@studio/routes/agents/AgentDetailRoute/network/DecideRequestModal';
import type { NetworkRequest } from '@studio/routes/agents/AgentDetailRoute/network/types';
import { binaryName, requestTarget } from '@studio/routes/agents/AgentDetailRoute/network/utils';
import type { ComponentProps, FC } from 'react';

interface NetworkRequestRowProps {
  request: NetworkRequest;
  onDecide: (kind: DecisionKind) => void;
}

const STATUS_BADGE: Record<
  string,
  { label: string; color: ComponentProps<typeof Badge>['color'] }
> = {
  pending: { label: 'Pending', color: 'yellow' },
  approved: { label: 'Allowed', color: 'green' },
  rejected: { label: 'Blocked', color: 'red' },
};

const attempts = (count: number) => `${count} ${count === 1 ? 'attempt' : 'attempts'}`;

/** One requested connection, with the decisions that still apply to it. */
export const NetworkRequestRow: FC<NetworkRequestRowProps> = ({ request, onDecide }) => {
  const badge = STATUS_BADGE[request.status] ?? { label: request.status, color: 'gray' };
  const seen = request.status === 'pending' ? request.last_seen : request.decided;
  const note = request.status === 'rejected' ? request.rejection_reason : request.rationale;
  const seenAgo = useRelativeTimeSince(seen ?? new Date().toISOString(), false);

  return (
    <Flex
      align="start"
      gap="3"
      className="border-t border-base px-4 py-3"
      data-testid="network-request-row"
    >
      <Stack gap="0" className="min-w-0 flex-1">
        <Text kind="body/semibold/sm" className="font-mono">
          {requestTarget(request)}
        </Text>
        <Flex gap="1" className="text-secondary" wrap="wrap">
          <Text kind="body/regular/xs" title={request.binary}>
            {binaryName(request.binary) || 'unknown process'}
          </Text>
          <Text kind="body/regular/xs">· {attempts(request.hit_count)}</Text>
          {seen && (
            <Text kind="body/regular/xs" title={seen}>
              · {request.status === 'pending' ? 'last tried' : 'decided'} {seenAgo}
            </Text>
          )}
        </Flex>
        {note && (
          <Text kind="body/regular/xs" className="text-secondary">
            {note}
          </Text>
        )}
        {request.security_notes && (
          <Text kind="body/regular/xs" className="text-feedback-warning">
            {request.security_notes}
          </Text>
        )}
      </Stack>
      <Flex align="center" gap="2" className="shrink-0">
        {request.remembered && (
          <Badge kind="outline" color="gray" size="small">
            Remembered
          </Badge>
        )}
        <Badge kind="solid" color={badge.color} size="small">
          {badge.label}
        </Badge>
        {request.status === 'pending' && (
          <Flex gap="1">
            <Button kind="secondary" size="small" onClick={() => onDecide('approve')}>
              Allow
            </Button>
            <Button kind="tertiary" size="small" color="danger" onClick={() => onDecide('reject')}>
              Keep blocked
            </Button>
          </Flex>
        )}
        {request.status === 'approved' && !request.remembered && (
          <Button kind="tertiary" size="small" onClick={() => onDecide('remember')}>
            Remember for agent
          </Button>
        )}
      </Flex>
    </Flex>
  );
};
