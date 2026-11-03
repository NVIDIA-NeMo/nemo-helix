// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import { NEMOTRON_PERSONAS_LOCALES } from '@studio/components/agents/PersonaTestSet/nemotronPersonas';
import {
  buildPersonaTestSetRequest,
  personaTestSetAgent,
} from '@studio/components/agents/PersonaTestSet/personaTestSetRequest';
import type { NemotronPersonasLocale } from '@studio/components/agents/PersonaTestSet/types';
import { useNemotronPersonasSetup } from '@studio/components/agents/PersonaTestSet/useNemotronPersonasSetup';
import { useEffect, useMemo, type FC } from 'react';
import { useForm, useWatch } from 'react-hook-form';

const DEFAULT_ROWS = '50';
const MAX_ROWS = 5000;

const LOCALE_ITEMS = NEMOTRON_PERSONAS_LOCALES.map((locale) => ({
  value: locale,
  children: locale,
}));

interface FormValues {
  agent: string;
  locale: NemotronPersonasLocale;
  rows: string;
}

export interface PersonaTestSetPanelProps {
  workspace: string;
  /** Fired on every change: the drafted job when the form is complete and the dataset is ready, else null. */
  onValidConfig: (jobRequest: DataDesignerJobRequest | null) => void;
}

const parseRows = (rows: string): number | undefined => {
  const value = Number(rows);
  return Number.isInteger(value) && value >= 1 && value <= MAX_ROWS ? value : undefined;
};

/** Picks the agent, population and row count for a Nemotron persona test set. */
export const PersonaTestSetPanel: FC<PersonaTestSetPanelProps> = ({ workspace, onValidConfig }) => {
  const { control } = useForm<FormValues>({
    defaultValues: { agent: '', locale: 'en_US', rows: DEFAULT_ROWS },
  });
  const [agentName, locale, rows] = useWatch({ control, name: ['agent', 'locale', 'rows'] });
  const personas = useNemotronPersonasSetup(workspace, locale);
  const { data, isLoading } = useAgentsListAgents(workspace, undefined, {
    query: { enabled: !!workspace },
  });
  const agents = useMemo(() => (data?.data ?? []).filter((agent) => !!agent.name), [data]);
  const agentItems = useMemo(
    () => agents.map((agent) => ({ value: agent.name as string, children: agent.name as string })),
    [agents]
  );
  const rowCount = parseRows(rows);
  const agent = agents.find((candidate) => candidate.name === agentName);

  useEffect(() => {
    onValidConfig(
      agent && rowCount && personas.status === 'ready'
        ? buildPersonaTestSetRequest({
            agent: personaTestSetAgent(agentName, agent),
            locale,
            rows: rowCount,
          })
        : null
    );
  }, [agent, agentName, locale, rowCount, personas.status, onValidConfig]);

  return (
    <>
      <Stack gap="density-lg">
        <ControlledSelect
          useControllerProps={{ name: 'agent', control }}
          formFieldProps={{
            slotLabel: 'Agent',
            slotInfo: 'Personas write messages this agent is meant to handle.',
          }}
          items={agentItems}
          placeholder={isLoading ? 'Loading agents…' : 'Select an agent'}
          disabled={isLoading}
        />
        <Flex gap="density-lg">
          <ControlledSelect
            useControllerProps={{ name: 'locale', control }}
            formFieldProps={{
              className: 'flex-1',
              slotLabel: 'Population',
              slotInfo: 'The Nemotron Personas locale people are sampled from.',
            }}
            items={LOCALE_ITEMS}
          />
          <ControlledTextInput
            type="number"
            useControllerProps={{ name: 'rows', control }}
            formFieldProps={{
              className: 'flex-1',
              slotLabel: 'Rows',
              status: rowCount ? undefined : 'error',
              slotError: rowCount ? undefined : `A whole number from 1 to ${MAX_ROWS}.`,
            }}
          />
        </Flex>
        {personas.banner}
      </Stack>
      {personas.secretModal}
    </>
  );
};
