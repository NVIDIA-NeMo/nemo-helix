// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WorkspaceAgentSelect } from '@nemo/common/src/components/AgentSelect';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import type { Agent } from '@nemo/sdk/generated/agents/schema';
import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { Flex, FormField, Stack } from '@nvidia/foundations-react-core';
import { NEMOTRON_PERSONAS_LOCALES } from '@studio/components/agents/PersonaTestSet/nemotronPersonas';
import {
  buildPersonaTestSetRequest,
  personaTestSetAgent,
} from '@studio/components/agents/PersonaTestSet/personaTestSetRequest';
import type { NemotronPersonasLocale } from '@studio/components/agents/PersonaTestSet/types';
import { useNemotronPersonasSetup } from '@studio/components/agents/PersonaTestSet/useNemotronPersonasSetup';
import { useEffect, useState, type FC } from 'react';
import { useForm, useWatch } from 'react-hook-form';

const DEFAULT_ROWS = '50';
const MAX_ROWS = 5000;

const LOCALE_ITEMS = NEMOTRON_PERSONAS_LOCALES.map((locale) => ({
  value: locale,
  children: locale,
}));

interface FormValues {
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
    defaultValues: { locale: 'en_US', rows: DEFAULT_ROWS },
  });
  const [locale, rows] = useWatch({ control, name: ['locale', 'rows'] });
  const [agent, setAgent] = useState<Agent>();
  const personas = useNemotronPersonasSetup(workspace, locale);
  const rowCount = parseRows(rows);

  useEffect(() => {
    onValidConfig(
      agent && rowCount && personas.status === 'ready'
        ? buildPersonaTestSetRequest({
            agent: personaTestSetAgent(agent.name ?? '', agent),
            locale,
            rows: rowCount,
          })
        : null
    );
  }, [agent, locale, rowCount, personas.status, onValidConfig]);

  return (
    <>
      <Stack gap="density-lg">
        <FormField
          slotLabel="Agent"
          slotInfo="Personas write messages this agent is meant to handle."
        >
          <WorkspaceAgentSelect
            workspace={workspace}
            value={agent?.name ?? null}
            onValueChange={(_, picked) => setAgent(picked)}
            aria-label="Agent"
            fullWidth
          />
        </FormField>
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
