// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentHardenerManifest } from '@agent-hardener/generated/schema';
import { AccordionSection } from '@nemo/common';
import { AccordionRoot, Button, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

interface TargetPanelProps {
  manifest: AgentHardenerManifest;
  /** Re-resolve against the agent as it is now. Omitted when the manifest names no agent. */
  onRefresh?: () => void;
  refreshing?: boolean;
  /** Open the environment-variable editor. */
  onEditEnv?: () => void;
}

const Row: FC<{ label: string; children: ReactNode }> = ({ label, children }) => (
  <Flex gap="density-md" className="items-baseline">
    <Text kind="body/regular/sm" className="shrink-0 text-fg-secondary" style={{ width: '8rem' }}>
      {label}
    </Text>
    <Text kind="body/regular/sm" className="break-all">
      {children}
    </Text>
  </Flex>
);

const Empty: FC<{ children: ReactNode }> = ({ children }) => (
  <span className="text-fg-secondary">{children}</span>
);

/**
 * What the war-game will actually attack.
 *
 * A manifest is a frozen target: `init` resolves the agent once and stores the result, so this is
 * the manifest every run uses — not a preview that gets regenerated. That is what makes showing it
 * worth doing, and what makes Refresh a deliberate action rather than a reload.
 */
export const TargetPanel: FC<TargetPanelProps> = ({
  manifest,
  onRefresh,
  refreshing,
  onEditEnv,
}) => {
  const egress = manifest.egress ?? [];
  const secrets = manifest.secrets ?? [];
  const env = Object.entries(manifest.env ?? {});

  return (
    <Panel>
      <Stack gap="density-lg" padding="density-lg">
        <Flex className="items-center justify-between">
          <Text kind="body/semibold/md">Target</Text>
          {onRefresh ? (
            <Button kind="secondary" size="small" disabled={refreshing} onClick={onRefresh}>
              {refreshing ? 'Refreshing' : 'Refresh Target'}
            </Button>
          ) : null}
        </Flex>

        <Text kind="body/regular/sm" className="text-fg-secondary">
          Resolved from the agent when this manifest was created. Editing the agent afterwards does
          not change it — refresh to take those changes.
        </Text>

        <Stack gap="density-sm">
          <Row label="Agent">
            {manifest.agent || <Empty>unknown</Empty>}
          </Row>
          <Row label="Victim Port">{manifest.port || <Empty>not set</Empty>}</Row>
          <Row label="Egress">
            {egress.length ? (
              egress.join(', ')
            ) : (
              <Empty>none — the victim&apos;s outbound calls are blocked</Empty>
            )}
          </Row>
          <Row label="Secrets">{secrets.length ? secrets.join(', ') : <Empty>none</Empty>}</Row>
          <Row label="Environment">
            <Flex gap="density-md" className="items-baseline">
              <span>
                {env.length ? (
                  env.map(([key, value]) => `${key}=${value}`).join(', ')
                ) : (
                  <Empty>none</Empty>
                )}
              </span>
              {onEditEnv ? (
                <Button kind="tertiary" size="small" onClick={onEditEnv}>
                  Edit
                </Button>
              ) : null}
            </Flex>
          </Row>
        </Stack>

        {manifest.manifest_yaml ? (
          <AccordionRoot>
            <AccordionSection value="manifest-yaml" title="agent-hardener.yaml">
              <pre className="overflow-auto px-density-xs text-sm text-fg-secondary whitespace-pre-wrap" style={{ maxHeight: '24rem' }}>
                {manifest.manifest_yaml}
              </pre>
            </AccordionSection>
          </AccordionRoot>
        ) : null}
      </Stack>
    </Panel>
  );
};
