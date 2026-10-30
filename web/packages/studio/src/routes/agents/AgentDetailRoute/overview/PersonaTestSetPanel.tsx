// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { Button, Stack, Text } from '@nvidia/foundations-react-core';
import { PersonaTestSetModal } from '@studio/components/agents/PersonaTestSet/PersonaTestSetModal';
import { personaTestSetAgent } from '@studio/components/agents/PersonaTestSet/personaTestSetRequest';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { type FC, useState } from 'react';

interface PersonaTestSetPanelProps {
  workspace: string;
  agentName: string;
  agent: Agent;
}

/** Entry point for drafting a Nemotron persona test set against this agent. */
export const PersonaTestSetPanel: FC<PersonaTestSetPanelProps> = ({
  workspace,
  agentName,
  agent,
}) => {
  const [open, setOpen] = useState(false);
  return (
    <DetailPanel title="Persona test set">
      <Stack gap="3">
        <Text kind="body/regular/sm" className="text-secondary">
          Draft test questions written by diverse synthetic personas, then evaluate the agent on
          them.
        </Text>
        <Button kind="secondary" className="w-fit" onClick={() => setOpen(true)}>
          Generate persona test set
        </Button>
      </Stack>
      <PersonaTestSetModal
        open={open}
        onClose={() => setOpen(false)}
        workspace={workspace}
        agent={personaTestSetAgent(agentName, agent)}
      />
    </DetailPanel>
  );
};
