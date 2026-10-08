// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { CustomizationOverview } from '@studio/components/CustomizationOverview';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import {
  customizationJob1,
  grpoCustomizationJob,
} from '@studio/mocks/customizer/customization-jobs';
import { server } from '@studio/mocks/node';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { TestProviders } from '@studio/tests/util/TestProviders';
import type { CustomizationJob } from '@studio/util/customizationBackend';
import { getBaseModel } from '@studio/util/customizations';
import { render, screen, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const renderOverview = async (job: CustomizationJob = customizationJob1) => {
  render(
    <TestProviders>
      <CustomizationOverview customizationJobName={job.name!} workspace={job.workspace} />
    </TestProviders>
  );
  await waitForElementToBeRemoved(() => screen.queryByText('Loading...'), {
    timeout: XL_SELECTOR_TIMEOUT,
  });
};

describe('CustomizationOverview', () => {
  it.each([
    [HelixJobStatus.created, 'Training metrics are not available yet', 'after training starts'],
    [HelixJobStatus.pending, 'Training metrics are not available yet', 'after training starts'],
    [HelixJobStatus.active, 'Training metrics are not available yet', 'after training starts'],
    [HelixJobStatus.error, 'Training metrics are unavailable', 'job failed'],
    [HelixJobStatus.cancelled, 'Training metrics are unavailable', 'job was stopped'],
    [HelixJobStatus.completed, 'No training metrics were reported', 'job completed'],
    [HelixJobStatus.paused, 'Training metrics are not available yet', 'job is paused'],
  ] as const)('explains an empty loss chart for a %s job', async (status, title, description) => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name`, () =>
        HttpResponse.json({
          ...customizationJob1,
          status,
          status_details: { metrics: { train_loss: [], val_loss: [] } },
        })
      )
    );
    await renderOverview();
    expect(screen.getByText(title)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(description))).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByText('No training data available')).not.toBeInTheDocument();
  });

  it('describes empty GRPO chart series as reward metrics', async () => {
    server.use(
      http.get(`${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs/:name`, () =>
        HttpResponse.json({
          ...grpoCustomizationJob,
          status: HelixJobStatus.active,
          status_details: {},
        })
      )
    );
    await renderOverview(grpoCustomizationJob);
    expect(screen.getByText(/Reward metrics are not available yet/)).toBeInTheDocument();
  });

  it('summarizes the training results as stat tiles', async () => {
    await renderOverview();

    expect(await screen.findByText('Final Training Loss')).toBeInTheDocument();
    expect(screen.getByText('Final Validation Loss')).toBeInTheDocument();
    expect(screen.getAllByText('0.9000')).toHaveLength(2);

    expect(screen.getByText('Steps Completed')).toBeInTheDocument();
    expect(screen.getByText('10 / 10')).toBeInTheDocument();
    expect(screen.getByText('Epochs Completed')).toBeInTheDocument();
    expect(screen.getByText('1 / 1')).toBeInTheDocument();
  });

  it('surfaces the live per-step telemetry as diagnostics tiles', async () => {
    await renderOverview();

    expect(await screen.findByText('Learning Rate')).toBeInTheDocument();
    expect(screen.getByText((5e-6).toExponential(2))).toBeInTheDocument();
    expect(screen.getByText('Gradient Norm')).toBeInTheDocument();
    expect(screen.getByText('1.2345')).toBeInTheDocument();

    expect(screen.getByText('Train/Val Gap')).toBeInTheDocument();
    expect(screen.getByText('0.0000')).toBeInTheDocument();

    expect(screen.getByText('Duration')).toBeInTheDocument();
    expect(screen.getByText('00:01:12')).toBeInTheDocument();
    expect(screen.queryByText('Phase')).not.toBeInTheDocument();
  });

  it('renders the run configuration', async () => {
    await renderOverview();

    expect(await screen.findByText('Run configuration')).toBeInTheDocument();

    expect(screen.getByText('Customization ID')).toBeInTheDocument();
    expect(screen.getByText(customizationJob1.id!)).toBeInTheDocument();

    expect(screen.getByText('Output Model')).toBeInTheDocument();
    expect(screen.getByText(customizationJob1.spec?.output?.name ?? '-')).toBeInTheDocument();

    expect(screen.getByText('Base Model')).toBeInTheDocument();
    expect(screen.getByText(getBaseModel(customizationJob1) || '-')).toBeInTheDocument();

    expect(screen.getByText('Description')).toBeInTheDocument();
    expect(screen.getByText(customizationJob1.description!)).toBeInTheDocument();

    expect(screen.getByText('Created')).toBeInTheDocument();
    expect(
      screen.getByText(
        customizationJob1.created_at ? formatAbsoluteTimestamp(customizationJob1.created_at) : '-'
      )
    ).toBeInTheDocument();

    expect(screen.getByText('Latest Checkpoint')).toBeInTheDocument();
    expect(screen.getByText('default/output-fileset/checkpoints/step-10')).toBeInTheDocument();

    expect(screen.getByRole('button', { name: 'View Job Configuration' })).toBeInTheDocument();
  });

  it('renders the training loss chart panel', async () => {
    await renderOverview();

    expect(await screen.findByText('Training loss')).toBeInTheDocument();
  });

  it('no longer renders status logs — they moved to their own tab', async () => {
    await renderOverview();

    expect(await screen.findByText('Run configuration')).toBeInTheDocument();
    expect(screen.queryByText('Status Logs')).not.toBeInTheDocument();
  });
});

describe('CustomizationOverview — GRPO', () => {
  it('reads on reward instead of loss', async () => {
    await renderOverview(grpoCustomizationJob);

    expect(await screen.findByText('Final Mean Reward')).toBeInTheDocument();
    expect(screen.getByText('0.6170')).toBeInTheDocument();
    expect(screen.getByText('+0.4370 from start')).toBeInTheDocument();

    expect(screen.getByText('Validation Reward')).toBeInTheDocument();
    expect(screen.getByText('0.5800')).toBeInTheDocument();
    // The eval step survives alongside the delta — "at the last eval interval" is the point.
    expect(screen.getByText('held-out prompts, step 500')).toBeInTheDocument();

    expect(screen.getByText('Truncation Rate')).toBeInTheDocument();
    expect(screen.getByText('4.1%')).toBeInTheDocument();

    // The median of the six reported steps, so the one slow step in the run cannot set the pace.
    expect(screen.getByText('Median Step Time')).toBeInTheDocument();
    expect(screen.getByText('34.2s')).toBeInTheDocument();
  });

  it('drops the loss tiles and the loss chart', async () => {
    await renderOverview(grpoCustomizationJob);

    expect(await screen.findByText('Reward')).toBeInTheDocument();
    // Both curves reach the chart — the series would otherwise fall back to the empty frame.
    expect(screen.getByRole('button', { name: 'Training reward' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Validation reward' })).toBeInTheDocument();
    expect(screen.queryByText(/Reward metrics are/)).not.toBeInTheDocument();

    expect(screen.queryByText('Training loss')).not.toBeInTheDocument();
    expect(screen.queryByText('Final Training Loss')).not.toBeInTheDocument();
    expect(screen.queryByText('Final Validation Loss')).not.toBeInTheDocument();
    // Loss-derived, so it says nothing about a GRPO run.
    expect(screen.queryByText('Train/Val Gap')).not.toBeInTheDocument();

    // Step count isn't duplicated here — it's already in the page header, and the chart's own x
    // axis is the step count — but epoch and duration/phase have no other home on the page.
    expect(screen.queryByText('Steps Completed')).not.toBeInTheDocument();
    expect(screen.getByText('Epochs Completed')).toBeInTheDocument();
    expect(screen.getByText('Duration')).toBeInTheDocument();
    expect(screen.queryByText('Phase')).not.toBeInTheDocument();
  });

  it('keeps the training health diagnostics collapsed until asked for', async () => {
    await renderOverview(grpoCustomizationJob);

    expect(await screen.findByText('Training health')).toBeInTheDocument();
    expect(screen.queryByText('train_gen_kl_error')).not.toBeInTheDocument();

    await userEvent.click(screen.getByText('Training health'));

    // Once as the tile label, once as the chart's own raw-key header.
    const genKlKeys = await screen.findAllByText('train_gen_kl_error');
    expect(genKlKeys).toHaveLength(2);
    // The header copy is styled as secondary, distinguishing it from the bold chart title.
    expect(genKlKeys[1]).toHaveClass('text-secondary');
    expect(screen.getByText('5.4e-4')).toBeInTheDocument();
    expect(screen.getAllByText('train_approx_entropy')).toHaveLength(2);
    expect(screen.getByText('falling')).toBeInTheDocument();

    // A ratio centred on 1 rendered as its deviation — 1.004 is 0.4% off-policy, not 100%.
    expect(screen.getByText('train_token_mult_prob_error')).toBeInTheDocument();
    expect(screen.getByText('0.4%')).toBeInTheDocument();

    // The sequence-level twin of that drift, read against 1 rather than as a deviation from it.
    expect(screen.getByText('train_sampling_importance_ratio')).toBeInTheDocument();
    expect(screen.getByText('1.002')).toBeInTheDocument();

    expect(screen.getByText('Generation KL')).toBeInTheDocument();
    expect(screen.getByText('Policy entropy')).toBeInTheDocument();
    expect(screen.getByText('Mean generated tokens per response')).toBeInTheDocument();
    // The chart has no tile of its own, so this raw key appears exactly once.
    expect(screen.getByText('train_gen_tokens_per_sample/mean')).toBeInTheDocument();
    expect(screen.getByText('Training step time')).toBeInTheDocument();
    expect(screen.queryByText('No training metrics to compare')).not.toBeInTheDocument();

    // Drift keeps its tile but gets no chart, and `kl_penalty` — a flat zero under the default
    // `ref_policy_kl_penalty=0` — is reported by the run but rendered nowhere.
    expect(screen.queryByText('Rollout / training logprob drift')).not.toBeInTheDocument();
    expect(screen.queryByText('Reference KL penalty')).not.toBeInTheDocument();
  });

  it('still renders the shared run configuration', async () => {
    await renderOverview(grpoCustomizationJob);

    expect(await screen.findByText('Run configuration')).toBeInTheDocument();
    expect(screen.getByText(grpoCustomizationJob.spec.output.name)).toBeInTheDocument();
    expect(
      screen.getByText('default/grpo-output-fileset/checkpoints/step-500')
    ).toBeInTheDocument();

    // `getBaseModel` returned '' for every RL job, so this row rendered blank.
    expect(screen.getByText('qwen/qwen2.5-7b-instruct')).toBeInTheDocument();
  });

  it('adds the GRPO run configuration rows', async () => {
    await renderOverview(grpoCustomizationJob);

    expect(await screen.findByText('Environment')).toBeInTheDocument();
    expect(screen.getByText('default/math-verifier-env')).toBeInTheDocument();

    expect(screen.getByText('Prompt Dataset')).toBeInTheDocument();

    expect(screen.getByText('Training Backend')).toBeInTheDocument();
    expect(screen.getByText('DTensor · Full weights')).toBeInTheDocument();

    expect(screen.getByText('Parallelism')).toBeInTheDocument();
    expect(screen.getByText('TP 4 · PP 1 · CP 1')).toBeInTheDocument();

    expect(screen.getByText('Generation')).toBeInTheDocument();
    expect(screen.getByText('vLLM, colocated · TP 4')).toBeInTheDocument();

    expect(screen.getByText('Sequence Packing')).toBeInTheDocument();
    expect(screen.getByText('Disabled')).toBeInTheDocument();
  });

  it('leaves the GRPO rows off a non-GRPO job', async () => {
    await renderOverview();

    expect(await screen.findByText('Run configuration')).toBeInTheDocument();
    expect(screen.queryByText('Environment')).not.toBeInTheDocument();
    expect(screen.queryByText('Training Backend')).not.toBeInTheDocument();
  });
});
