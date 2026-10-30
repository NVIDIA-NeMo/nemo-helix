// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledRadioGroup } from '@nemo/common/src/components/form/ControlledRadioGroup';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { FormModal } from '@nemo/common/src/components/FormModal';
import { DEFAULT_LARGE_PAGE_SIZE } from '@nemo/common/src/constants/api';
import { useSecretsListSecrets } from '@nemo/sdk/generated/platform/secrets';
import { Banner, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { NEMOTRON_PERSONAS_LOCALES } from '@studio/components/agents/PersonaTestSet/nemotronPersonas';
import { buildPersonaTestSetRequest } from '@studio/components/agents/PersonaTestSet/personaTestSetRequest';
import type {
  NemotronPersonasLocale,
  PersonaAnswerStyle,
  PersonaTestSetAgent,
} from '@studio/components/agents/PersonaTestSet/types';
import { useNemotronPersonasFileset } from '@studio/components/agents/PersonaTestSet/useNemotronPersonasFileset';
import type { DataDesignerGeneratedState } from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import { getDataDesignerJobBuildRoute } from '@studio/routes/utils';
import { type FC, useMemo } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';

const DEFAULT_ROWS = '50';
const MAX_ROWS = 5000;

const LOCALE_ITEMS = NEMOTRON_PERSONAS_LOCALES.map((locale) => ({
  value: locale,
  children: locale,
}));

const ANSWER_STYLE_ITEMS = [
  { value: 'rubric', children: 'Rubric written by an LLM' },
  { value: 'arithmetic', children: 'Computed arithmetic' },
];

interface FormValues {
  locale: NemotronPersonasLocale;
  rows: string;
  answerStyle: PersonaAnswerStyle;
  apiKeySecret: string;
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
 * Starts a Nemotron persona test set for an agent: picks the population and answer style, makes
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
    defaultValues: { locale: 'en_US', rows: DEFAULT_ROWS, answerStyle: 'rubric', apiKeySecret: '' },
  });
  const [locale, rows, apiKeySecret] = useWatch({
    control,
    name: ['locale', 'rows', 'apiKeySecret'],
  });
  const personas = useNemotronPersonasFileset(locale);
  const { data: secretsPage } = useSecretsListSecrets(
    workspace,
    { page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled: open && personas.status === 'missing' } }
  );
  const secretItems = useMemo(
    () =>
      (secretsPage?.data ?? []).map((secret) => ({
        value: `${workspace}/${secret.name}`,
        children: secret.name,
      })),
    [secretsPage?.data, workspace]
  );

  const rowCount = parseRows(rows);

  const submit = handleSubmit(({ answerStyle }) => {
    if (!rowCount) return;
    const state: DataDesignerGeneratedState = {
      generatedJobRequest: buildPersonaTestSetRequest({
        agent,
        locale,
        rows: rowCount,
        answerStyle,
      }),
    };
    onClose();
    navigate(getDataDesignerJobBuildRoute(workspace), { state });
  });

  return (
    <FormModal
      open={open}
      onClose={onClose}
      className="w-[640px] max-w-[90vw]"
      title="Generate persona test set"
      instruction="Each row samples a person from Nemotron Personas and has them write a message to this agent, with the answer it should give. You review and run the job in Data Designer, then evaluate the agent on its output."
      submitButtonText="Review in Data Designer"
      onSubmit={(event) => void submit(event)}
      submitDisabled={personas.status !== 'ready' || !rowCount}
    >
      <Stack gap="density-lg">
        <Flex gap="density-lg">
          <ControlledSelect
            className="flex-1"
            useControllerProps={{ name: 'locale', control }}
            formFieldProps={{
              slotLabel: 'Population',
              slotInfo: 'The Nemotron Personas locale people are sampled from.',
            }}
            items={LOCALE_ITEMS}
          />
          <ControlledTextInput
            className="flex-1"
            type="number"
            useControllerProps={{ name: 'rows', control }}
            formFieldProps={{
              slotLabel: 'Rows',
              status: rowCount ? undefined : 'error',
              slotError: rowCount ? undefined : `A whole number from 1 to ${MAX_ROWS}.`,
            }}
          />
        </Flex>
        <ControlledRadioGroup
          useControllerProps={{ name: 'answerStyle', control }}
          formFieldProps={{
            slotLabel: 'Expected answers',
            slotInfo:
              'Computed arithmetic samples the numbers and calculates the exact result, for calculator-style agents.',
          }}
          items={ANSWER_STYLE_ITEMS}
        />
        {personas.status === 'error' ? (
          <Banner kind="inline" status="error">
            <Stack gap="density-xs">
              <Text kind="body/bold/sm">Could not check the persona dataset</Text>
              <Text kind="body/regular/sm">{personas.checkError}</Text>
            </Stack>
          </Banner>
        ) : null}
        {personas.status === 'missing' ? (
          <Banner kind="inline" status="warning">
            <Stack gap="density-md">
              <Text kind="body/bold/sm">Nemotron Personas for {locale} isn't set up</Text>
              <Text kind="body/regular/sm">
                Register <code>{personas.filesetRef}</code> once for the whole platform. It reads
                the dataset from NGC with the API key in the secret you pick.
              </Text>
              <Flex gap="density-md" align="end">
                <ControlledSelect
                  className="flex-1"
                  useControllerProps={{ name: 'apiKeySecret', control }}
                  formFieldProps={{ slotLabel: 'NGC API key secret' }}
                  placeholder={
                    secretItems.length ? 'Select a secret' : 'No secrets in this workspace'
                  }
                  items={secretItems}
                />
                <Button
                  type="button"
                  kind="secondary"
                  disabled={!apiKeySecret || personas.isCreating}
                  onClick={() => void personas.create(apiKeySecret).catch(() => undefined)}
                >
                  {personas.isCreating ? 'Registering…' : 'Register dataset'}
                </Button>
              </Flex>
              {personas.createError ? (
                <Text kind="body/regular/sm" className="text-feedback-danger">
                  {personas.createError}
                </Text>
              ) : null}
            </Stack>
          </Banner>
        ) : null}
      </Stack>
    </FormModal>
  );
};
