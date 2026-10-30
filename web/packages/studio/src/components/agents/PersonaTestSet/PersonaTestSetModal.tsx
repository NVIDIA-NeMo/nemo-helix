// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import { NEMOTRON_PERSONAS_LOCALES } from '@studio/components/agents/PersonaTestSet/nemotronPersonas';
import { buildPersonaTestSetRequest } from '@studio/components/agents/PersonaTestSet/personaTestSetRequest';
import type {
  NemotronPersonasLocale,
  PersonaTestSetAgent,
} from '@studio/components/agents/PersonaTestSet/types';
import { useNemotronPersonasSetup } from '@studio/components/agents/PersonaTestSet/useNemotronPersonasSetup';
import type { DataDesignerGeneratedState } from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import { getDataDesignerJobBuildRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';

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

interface PersonaTestSetModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  agent: PersonaTestSetAgent;
}

const parseRows = (rows: string): number | undefined => {
  const value = Number(rows);
  return Number.isInteger(value) && value >= 1 && value <= MAX_ROWS ? value : undefined;
};

/**
 * Starts a Nemotron persona test set for an agent: picks the population and row count, makes
 * sure the platform can read that population, then opens the drafted job in the Data Designer
 * builder for review.
 */
export const PersonaTestSetModal: FC<PersonaTestSetModalProps> = ({
  open,
  onClose,
  workspace,
  agent,
}) => {
  const navigate = useNavigate();
  const { control, handleSubmit } = useForm<FormValues>({
    defaultValues: { locale: 'en_US', rows: DEFAULT_ROWS },
  });
  const [locale, rows] = useWatch({ control, name: ['locale', 'rows'] });
  const personas = useNemotronPersonasSetup(workspace, locale);
  const rowCount = parseRows(rows);

  const submit = handleSubmit(() => {
    if (!rowCount || personas.status !== 'ready') return;
    const state: DataDesignerGeneratedState = {
      generatedJobRequest: buildPersonaTestSetRequest({ agent, locale, rows: rowCount }),
    };
    onClose();
    navigate(getDataDesignerJobBuildRoute(workspace), { state });
  });

  return (
    <>
      <FormModal
        open={open}
        onClose={onClose}
        className="w-[640px] max-w-[90vw]"
        title="Generate persona test set"
        instruction="Each row samples a person from Nemotron Personas and has them write a message to this agent. You review and run the job in Data Designer, then evaluate the agent, which answers each message itself."
        submitButtonText="Review in Data Designer"
        onSubmit={(event) => void submit(event)}
        submitDisabled={personas.status !== 'ready' || !rowCount}
      >
        <Stack gap="density-lg">
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
      </FormModal>
      {personas.secretModal}
    </>
  );
};
