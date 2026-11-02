// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TrainingMemoryEstimate } from '@nemo/sdk/generated/platform/schema';
import { TrainingMemoryWarning } from '@studio/components/NewCustomizationForm/TrainingMemoryWarning';
import { render, screen } from '@studio/tests/util/render';

const estimate = (over: Partial<TrainingMemoryEstimate> = {}): TrainingMemoryEstimate => ({
  gpu_count: 1,
  gpu_memory_gb: 80,
  gpu_memory_assumed: true,
  data_parallel_size: 1,
  model_states_gb: 104.7,
  activations_gb: 6.06,
  overhead_gb: 15.38,
  total_gb: 126.14,
  utilization: 1.5768,
  warning_threshold: 0.85,
  status: 'exceeds_capacity',
  limiting_factor: 'model_states',
  ...over,
});

describe('TrainingMemoryWarning', () => {
  it('renders nothing without an estimate', () => {
    render(<TrainingMemoryWarning estimate={undefined} usesLora={false} />);
    expect(screen.queryByTestId('nv-banner-root')).not.toBeInTheDocument();
  });

  it('renders nothing when the configuration fits', () => {
    render(<TrainingMemoryWarning estimate={estimate({ status: 'ok' })} usesLora={false} />);
    expect(screen.queryByTestId('nv-banner-root')).not.toBeInTheDocument();
  });

  it('warns that the job will likely run out of memory and states the assumed GPU size', () => {
    render(<TrainingMemoryWarning estimate={estimate()} usesLora={false} />);

    expect(screen.getByText('Likely to run out of GPU memory')).toBeInTheDocument();
    expect(
      screen.getByText(/126\.1 GB per GPU, more than the 80\.0 GB assumed per GPU/)
    ).toBeInTheDocument();
    expect(screen.getByText(/you can still start the job/)).toBeInTheDocument();
  });

  it('shows the utilization when near capacity', () => {
    render(
      <TrainingMemoryWarning
        estimate={estimate({
          status: 'near_capacity',
          total_gb: 72.8,
          utilization: 0.91,
          gpu_memory_gb: 141,
          gpu_memory_assumed: false,
        })}
        usesLora={false}
      />
    );

    expect(screen.getByText('Close to the GPU memory limit')).toBeInTheDocument();
    expect(screen.getByText(/72\.8 GB per GPU, 91% of 141\.0 GB per GPU/)).toBeInTheDocument();
  });

  it('suggests LoRA for an all-weights run limited by model states', () => {
    render(<TrainingMemoryWarning estimate={estimate()} usesLora={false} />);
    expect(screen.getByText(/use LoRA instead of all-weights fine-tuning/)).toBeInTheDocument();
  });

  it('does not suggest LoRA to a run already using it', () => {
    render(<TrainingMemoryWarning estimate={estimate()} usesLora />);
    expect(screen.queryByText(/use LoRA/)).not.toBeInTheDocument();
    expect(screen.getByText(/raise the tensor parallel size/)).toBeInTheDocument();
  });

  it('suggests smaller batches or sequences when activations dominate', () => {
    render(
      <TrainingMemoryWarning
        estimate={estimate({ limiting_factor: 'activations' })}
        usesLora={false}
      />
    );
    expect(
      screen.getByText(/lower the micro batch size, or lower the max sequence length/)
    ).toBeInTheDocument();
  });
});
