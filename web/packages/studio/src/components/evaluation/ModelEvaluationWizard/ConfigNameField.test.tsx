// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
} from '@studio/components/evaluation/EvalWizard/types';
import { ConfigNameField } from '@studio/components/evaluation/ModelEvaluationWizard/ConfigNameField';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

const NameField: FC = () => {
  const form = useForm<EvaluationFormValues>({
    defaultValues: { ...EVALUATION_FORM_DEFAULTS, name: 'judge-validation-v1' },
  });
  return (
    <FormProvider {...form}>
      <ConfigNameField />
    </FormProvider>
  );
};

describe('ConfigNameField', () => {
  it('names a taken name as a fileset, not as an existing configuration', async () => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`, () =>
        HttpResponse.json({ data: [{ name: 'judge-validation-v1', purpose: 'dataset' }] })
      )
    );

    renderRoute(undefined, {
      history: '/workspaces/default/evaluations/new',
      routes: [{ path: '/workspaces/:workspace/evaluations/new', element: <NameField /> }],
    });

    expect(
      await screen.findByText(
        'A fileset named judge-validation-v1 already exists in the default workspace'
      )
    ).toBeInTheDocument();
    expect(screen.queryByText(/configuration named/)).not.toBeInTheDocument();
  });
});
