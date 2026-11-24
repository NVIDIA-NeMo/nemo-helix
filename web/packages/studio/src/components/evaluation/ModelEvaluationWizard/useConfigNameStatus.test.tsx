// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema/FilesetPurpose';
import {
  EVALUATION_FORM_DEFAULTS,
  type EvaluationFormValues,
} from '@studio/components/evaluation/EvalWizard/types';
import { ConfigNameField } from '@studio/components/evaluation/ModelEvaluationWizard/ConfigNameField';
import { configNameTakenMessage } from '@studio/components/evaluation/ModelEvaluationWizard/useConfigNameStatus';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { renderRoute } from '@studio/tests/util/render';
import { act, screen } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import type { FC } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

describe('configNameTakenMessage', () => {
  it.each([
    [FilesetPurpose.dataset, 'a dataset'],
    [FilesetPurpose.model, 'a model'],
    [FilesetPurpose.generic, 'a fileset'],
    [undefined, 'a fileset'],
  ])('names a %s conflict as %s, never as a configuration', (purpose, owner) => {
    const message = configNameTakenMessage(purpose, 'default');
    expect(message).toBe(
      `This name is already used by ${owner} in the default workspace. Choose a different configuration name.`
    );
    expect(message).not.toMatch(/configuration named/);
  });
});

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
  afterEach(() => {
    vi.useRealTimers();
  });

  it('names the dataset holding the name and checks it only once', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let requests = 0;
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`, () => {
        requests += 1;
        return HttpResponse.json({
          data: [{ name: 'judge-validation-v1', purpose: FilesetPurpose.dataset }],
        });
      })
    );

    renderRoute(undefined, {
      history: '/workspaces/default/evaluations/new',
      routes: [{ path: '/workspaces/:workspace/evaluations/new', element: <NameField /> }],
    });

    expect(
      await screen.findByText(configNameTakenMessage(FilesetPurpose.dataset, 'default'))
    ).toBeInTheDocument();
    // Several debounce windows: a refetch loop would have fired again by now.
    await act(() => vi.advanceTimersByTimeAsync(DEFAULT_DEBOUNCE_MS * 5));
    expect(requests).toBe(1);
  });
});
