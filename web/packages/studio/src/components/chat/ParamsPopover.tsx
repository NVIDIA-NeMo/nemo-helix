// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Button,
  Flex,
  Popover,
  Slider,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from '@nvidia/foundations-react-core';
import { InfoCircle, Retry, Sliders } from '@nvidia/react-gui-icons/icons';
import { DEFAULT_INFERENCE_PARAMS, type InferenceParams } from '@studio/components/chat/params';
import { type FC } from 'react';

interface ParamsPopoverProps {
  value: InferenceParams;
  onChange: (next: InferenceParams) => void;
}

const SLIDERS: Array<{
  key: keyof InferenceParams;
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
  default: number;
}> = [
  {
    key: 'temperature',
    label: 'Temperature',
    min: 0,
    max: 2,
    step: 0.05,
    hint: 'Controls output randomness. Higher values produce more creative, varied responses; lower values are more deterministic.',
    default: DEFAULT_INFERENCE_PARAMS.temperature,
  },
  {
    key: 'max_tokens',
    label: 'Max tokens',
    min: 32,
    max: 4096,
    step: 32,
    hint: 'Maximum number of tokens the model will generate in a single response.',
    default: DEFAULT_INFERENCE_PARAMS.max_tokens,
  },
];

export const ParamsPopover: FC<ParamsPopoverProps> = ({ value, onChange }) => {
  const update = (key: keyof InferenceParams, v: number) => {
    onChange({ ...value, [key]: v });
  };

  const clamp = (s: (typeof SLIDERS)[number], v: number) => Math.min(Math.max(v, s.min), s.max);

  return (
    <Popover
      slotContent={
        <Stack gap="density-lg" className="w-[380px] p-4">
          <Text kind="label/bold/lg">Inference parameters</Text>
          {SLIDERS.map((s) => {
            const current = value[s.key] as number;
            return (
              <Flex key={s.key} align="center" gap="density-md">
                <Flex align="center" gap="density-xs" className="w-[120px] shrink-0">
                  <Text kind="label/regular/md" className="truncate">
                    {s.label}
                  </Text>
                  <Tooltip slotContent={s.hint} side="top">
                    <InfoCircle size="12" className="shrink-0 text-fg-subdued" />
                  </Tooltip>
                </Flex>
                <div className="min-w-0 flex-1">
                  <Slider
                    value={current}
                    onValueChange={(v) => update(s.key, clamp(s, v))}
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    aria-label={s.label}
                  />
                </div>
                <TextInput
                  type="number"
                  value={String(current)}
                  min={String(s.min)}
                  max={String(s.max)}
                  step={String(s.step)}
                  aria-label={`${s.label} value`}
                  className="w-[80px] shrink-0"
                  attributes={{ Input: { className: 'text-center' } }}
                  onValueChange={(v) => {
                    if (v === '') return;
                    const n = parseFloat(v);
                    if (!Number.isNaN(n)) update(s.key, clamp(s, n));
                  }}
                />
                <Button
                  kind="tertiary"
                  size="small"
                  aria-label={`Reset ${s.label}`}
                  title={`Reset to ${s.default}`}
                  onClick={() => update(s.key, s.default)}
                  className="shrink-0"
                  type="button"
                >
                  <Retry size="14" />
                </Button>
              </Flex>
            );
          })}
        </Stack>
      }
    >
      <Button kind="secondary" aria-label="Inference parameters" title="Inference parameters">
        <Sliders size="14" />
      </Button>
    </Popover>
  );
};
