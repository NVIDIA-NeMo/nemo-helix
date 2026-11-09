// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Block, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { NemotronPersonasLocale } from '@studio/components/agents/PersonaTestSet/types';
import { useNemotronPersonasFileset } from '@studio/components/agents/PersonaTestSet/useNemotronPersonasFileset';
import { CreateSecretModal } from '@studio/routes/SecretsListRoute/CreateSecretModal';
import { SecretSearchableSelect } from '@studio/routes/SecretsListRoute/SecretSearchableSelect';
import { type ReactNode, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';

/**
 * Checks that the person sampler can read `locale`. `banner` explains why it can't and registers
 * the dataset, and is empty once the dataset is ready. `secretModal` creates the NGC key secret;
 * render it outside any `<form>`, since it holds its own.
 */
export const useNemotronPersonasSetup = (workspace: string, locale: NemotronPersonasLocale) => {
  const personas = useNemotronPersonasFileset(locale);
  const [isSecretModalOpen, setSecretModalOpen] = useState(false);
  const { control, setValue } = useForm({ defaultValues: { apiKeySecret: '' } });
  const apiKeySecret = useWatch({ control, name: 'apiKeySecret' });

  const secretModal = isSecretModalOpen ? (
    <CreateSecretModal
      workspace={workspace}
      open
      onClose={() => setSecretModalOpen(false)}
      onSecretCreated={(secretName) => {
        setValue('apiKeySecret', secretName);
        setSecretModalOpen(false);
      }}
    />
  ) : null;

  const errorBanner = (
    <Banner kind="inline" status="error" className="shrink-0">
      <Stack gap="density-xs">
        <Text kind="body/bold/sm">Could not check the persona dataset</Text>
        <Text kind="body/regular/sm">{personas.checkError}</Text>
      </Stack>
    </Banner>
  );

  const missingBanner = (
    <Banner kind="inline" status="warning" className="shrink-0">
      <Stack gap="density-md">
        <Text kind="body/bold/sm">Nemotron Personas for {locale} isn't set up</Text>
        <Text kind="body/regular/sm">
          Register <code>{personas.filesetRef}</code> once for the whole platform. It reads the
          dataset from NGC with the API key in the secret you pick.
        </Text>
        <Flex gap="density-md" align="end">
          <Block className="flex-1">
            <SecretSearchableSelect
              workspace={workspace}
              queryEnabled
              useControllerProps={{ name: 'apiKeySecret', control }}
              onRequestNewSecret={() => setSecretModalOpen(true)}
              triggerPlaceholder="Select a secret"
              formFieldProps={{ slotLabel: 'NGC API key secret' }}
            />
          </Block>
          <Button
            type="button"
            kind="secondary"
            disabled={!apiKeySecret || personas.isCreating}
            onClick={() =>
              void personas.create(`${workspace}/${apiKeySecret}`).catch(() => undefined)
            }
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
  );

  const banners: Partial<Record<typeof personas.status, ReactNode>> = {
    error: errorBanner,
    missing: missingBanner,
  };
  return { status: personas.status, banner: banners[personas.status] ?? null, secretModal };
};
