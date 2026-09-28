// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { NewDataDesignerJobForm } from '@studio/components/NewDataDesignerJobForm';
import { DEFAULT_BUILD_MODEL_NAME, DEFAULT_MODEL_NAME } from '@studio/constants/constants';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { mockUseNavigate, mockUseParams } from '@studio/tests/util/mockUseParams';
import { render, screen } from '@studio/tests/util/render';
import { fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('react-oidc-context', () => ({
  useAuth: () => ({ user: { access_token: 'test-token' } }),
}));

vi.mock('@studio/components/NewDataDesignerJobForm/usePreview', () => ({
  usePreview: () => ({ previewLogs: null, isPreviewing: false, runPreview: vi.fn() }),
}));

vi.mock('@studio/components/NewDataDesignerJobForm/JobRequestGenerator', async () => {
  const { Controller } = await import('react-hook-form');
  return {
    JobRequestGenerator: ({
      setJsonContent,
      control,
      descriptionName,
    }: {
      setJsonContent: (value: string) => void;
      control: Parameters<typeof Controller>[0]['control'];
      descriptionName: string;
      [key: string]: unknown;
    }) => (
      <div>
        <Controller
          name={descriptionName as 'description'}
          control={control}
          render={({ field }) => <textarea aria-label="Data description" {...field} />}
        />
        <button
          type="button"
          onClick={() =>
            setJsonContent(
              JSON.stringify({
                name: 'generated-job',
                spec: {
                  num_records: 50,
                  config: { columns: [], model_configs: [] },
                },
              } satisfies DataDesignerJobRequest)
            )
          }
        >
          Simulate Generate
        </button>
      </div>
    ),
  };
});

const WORKSPACE = 'test-workspace';

function setupProvidersMock() {
  server.use(
    http.get(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`, () =>
      HttpResponse.json({
        data: [
          {
            workspace: WORKSPACE,
            name: 'test-provider',
            served_models: [
              {
                model_entity_id: `${WORKSPACE}/${DEFAULT_BUILD_MODEL_NAME}`,
                served_model_name: DEFAULT_MODEL_NAME,
              },
            ],
          },
        ],
        pagination: {
          page: 1,
          page_size: 100,
          current_page_size: 1,
          total_pages: 1,
          total_results: 1,
        },
      })
    )
  );
}

function setupCreateJobMock(onRequest: (body: DataDesignerJobRequest) => void) {
  server.use(
    http.post(
      `${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create`,
      async ({ request }) => {
        onRequest((await request.json()) as DataDesignerJobRequest);
        return HttpResponse.json({
          name: 'created-job',
          workspace: WORKSPACE,
          status: 'created',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
      }
    )
  );
}

describe('NewDataDesignerJobForm', () => {
  beforeEach(() => {
    mockUseParams({ workspace: WORKSPACE });
    mockUseNavigate();
    setupProvidersMock();
  });

  async function fillRequiredFields(user: ReturnType<typeof userEvent.setup>) {
    await screen.findByRole('button', { name: 'Create Job' });
    await user.type(screen.getByLabelText('Data description'), 'At least ten characters');
    await user.click(screen.getByRole('button', { name: 'Simulate Generate' }));
  }

  describe('rows field takes priority over generated config num_records', () => {
    it('uses the form rows value when it differs from the generated config num_records', async () => {
      const user = userEvent.setup();
      const submitted: DataDesignerJobRequest[] = [];
      setupCreateJobMock((body) => submitted.push(body));

      render(<NewDataDesignerJobForm />);
      await fillRequiredFields(user);

      // Generated config set num_records=50, which also synced rows field to 50.
      // User overrides rows to 25.
      const rowsInput = screen.getByRole('spinbutton', {
        name: /records to generate/i,
      });
      fireEvent.change(rowsInput, { target: { value: '25' } });

      await user.click(screen.getByRole('button', { name: 'Create Job' }));

      await waitFor(() => {
        expect(submitted).toHaveLength(1);
        expect(submitted[0].spec?.num_records).toBe(25);
      });
    });

    it('uses the synced rows value when user does not override it after generation', async () => {
      const user = userEvent.setup();
      const submitted: DataDesignerJobRequest[] = [];
      setupCreateJobMock((body) => submitted.push(body));

      render(<NewDataDesignerJobForm />);
      await fillRequiredFields(user);

      // Generated config set num_records=50 and synced rows field to 50.
      // User does not change rows.
      expect(screen.getByRole('spinbutton', { name: /records to generate/i })).toHaveValue(50);

      await user.click(screen.getByRole('button', { name: 'Create Job' }));

      await waitFor(() => {
        expect(submitted).toHaveLength(1);
        expect(submitted[0].spec?.num_records).toBe(50);
      });
    });
  });

  describe('name and description form fields override the generated config', () => {
    it('applies form name over the generated config name when provided', async () => {
      const user = userEvent.setup();
      const submitted: DataDesignerJobRequest[] = [];
      setupCreateJobMock((body) => submitted.push(body));

      render(<NewDataDesignerJobForm />);
      await screen.findByRole('button', { name: 'Create Job' });
      await user.type(screen.getByRole('textbox', { name: /fileset name/i }), 'my-custom-name');
      await user.type(screen.getByLabelText('Data description'), 'At least ten characters');
      await user.click(screen.getByRole('button', { name: 'Simulate Generate' }));

      await user.click(screen.getByRole('button', { name: 'Create Job' }));

      await waitFor(() => {
        expect(submitted).toHaveLength(1);
        expect(submitted[0].name).toBe('my-custom-name');
      });
    });

    it('applies form job description over generated config description when provided', async () => {
      const user = userEvent.setup();
      const submitted: DataDesignerJobRequest[] = [];
      setupCreateJobMock((body) => submitted.push(body));

      render(<NewDataDesignerJobForm />);
      await screen.findByRole('button', { name: 'Create Job' });
      await user.type(screen.getByLabelText('Data description'), 'At least ten characters');
      await user.type(screen.getByRole('textbox', { name: /^description/i }), 'My job description');
      await user.click(screen.getByRole('button', { name: 'Simulate Generate' }));

      await user.click(screen.getByRole('button', { name: 'Create Job' }));

      await waitFor(() => {
        expect(submitted).toHaveLength(1);
        expect(submitted[0].description).toBe('My job description');
      });
    });

    it('preserves the generated config name when form name is empty', async () => {
      const user = userEvent.setup();
      const submitted: DataDesignerJobRequest[] = [];
      setupCreateJobMock((body) => submitted.push(body));

      render(<NewDataDesignerJobForm />);
      await fillRequiredFields(user);
      await user.click(screen.getByRole('button', { name: 'Create Job' }));

      await waitFor(() => {
        expect(submitted).toHaveLength(1);
        // The generated config name 'generated-job' should be used (sanitized)
        expect(submitted[0].name).toBe('generated-job');
      });
    });
  });
});
