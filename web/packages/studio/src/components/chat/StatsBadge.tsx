// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartPerformance, Numbers, Timer } from '@nemo/icons/icons';
import type { FC } from 'react';

export interface ChatMetrics {
  totalMs: number;
  completionTokens: number;
  tokensPerSec: number;
}

interface StatsBadgeProps {
  metrics: ChatMetrics;
  emphasis?: boolean;
  /** Colour of the metrics. Defaults to subdued grey; 'brand' renders NVIDIA green. */
  tone?: 'subdued' | 'brand';
  className?: string;
}

export const StatsBadge: FC<StatsBadgeProps> = ({
  metrics,
  emphasis,
  tone = 'subdued',
  className,
}) => {
  const seconds = (metrics.totalMs / 1000).toFixed(1);
  const tokensPerSec = Math.max(0, Math.round(metrics.tokensPerSec));
  const iconSize = emphasis ? '14' : '12';
  const toneClass =
    tone === 'brand' ? 'text-[var(--color-brand,#76b900)]' : 'text-fg-subdued opacity-60';
  return (
    <div
      className={`inline-flex items-center gap-3 font-mono ${toneClass} ${emphasis ? 'text-sm font-bold' : 'text-xs'} ${className ?? ''}`}
    >
      <span className="inline-flex items-center gap-1" title="Total time">
        <Timer size={iconSize} />
        {seconds}s
      </span>
      <span className="inline-flex items-center gap-1" title="Tokens per second">
        <ChartPerformance size={iconSize} />
        {tokensPerSec} t/s
      </span>
      <span className="inline-flex items-center gap-1" title="Completion tokens">
        <Numbers size={iconSize} />
        {Math.round(metrics.completionTokens)} tok
      </span>
    </div>
  );
};
