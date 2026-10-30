// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, isNotFoundError } from '@nemo/common/src/api/common/utils';
import {
  getFilesRetrieveFilesetQueryKey,
  useFilesCreateFileset,
  useFilesRetrieveFileset,
} from '@nemo/sdk/generated/platform/files';
import {
  NEMOTRON_PERSONAS_WORKSPACE,
  nemotronPersonasFilesetName,
  nemotronPersonasFilesetRequest,
} from '@studio/components/agents/PersonaTestSet/nemotronPersonas';
import type { NemotronPersonasLocale } from '@studio/components/agents/PersonaTestSet/types';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';

export type NemotronPersonasFilesetStatus = 'checking' | 'ready' | 'missing' | 'error';

/**
 * Whether the person sampler can read `locale`, and a way to register its NGC-backed fileset when
 * it cannot. There is no Faker fallback: a persona test set without Nemotron Personas is not one.
 */
export const useNemotronPersonasFileset = (locale: NemotronPersonasLocale) => {
  const queryClient = useQueryClient();
  const name = nemotronPersonasFilesetName(locale);
  const { isPending, error } = useFilesRetrieveFileset(NEMOTRON_PERSONAS_WORKSPACE, name, {
    query: { retry: (count, err) => !isNotFoundError(err) && count < 2 },
  });
  const createFileset = useFilesCreateFileset();
  const { mutateAsync } = createFileset;

  const status: NemotronPersonasFilesetStatus =
    [isPending && 'checking', error && isNotFoundError(error) && 'missing', error && 'error'].find(
      (value): value is NemotronPersonasFilesetStatus => !!value
    ) ?? 'ready';

  const create = useCallback(
    async (apiKeySecret: string) => {
      await mutateAsync({
        workspace: NEMOTRON_PERSONAS_WORKSPACE,
        data: nemotronPersonasFilesetRequest(locale, apiKeySecret),
      });
      await queryClient.invalidateQueries({
        queryKey: getFilesRetrieveFilesetQueryKey(NEMOTRON_PERSONAS_WORKSPACE, name),
      });
    },
    [locale, mutateAsync, name, queryClient]
  );

  return {
    filesetRef: `${NEMOTRON_PERSONAS_WORKSPACE}/${name}`,
    status,
    checkError: status === 'error' && error ? getErrorMessage(error) : undefined,
    create,
    isCreating: createFileset.isPending,
    createError: createFileset.error ? getErrorMessage(createFileset.error) : undefined,
  };
};
