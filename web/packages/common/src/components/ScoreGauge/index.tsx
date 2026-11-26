// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ARC_START, ARC_SWEEP, CENTER } from '@nemo/common/src/components/ScoreGauge/consts';
import {
  arcPath,
  gradientSegments,
  point,
  scoreColor,
} from '@nemo/common/src/components/ScoreGauge/utils';

export * from '@nemo/common/src/components/ScoreGauge/consts';
export * from '@nemo/common/src/components/ScoreGauge/utils';

type ScoreGaugeSize = 'lg' | 'sm';

interface ScoreGaugeProps {
  /** Raw score on a 0-10 scale. Falsy / non-finite renders as unavailable. */
  score?: number | null;
  size?: ScoreGaugeSize;
  /** Scales the gauge to fit its parent while keeping a square aspect ratio. */
  scaleToFit?: boolean;
}

export const ScoreGauge = ({ score, size = 'lg', scaleToFit = false }: ScoreGaugeProps) => {
  const available = typeof score === 'number' && Number.isFinite(score) && score > 0;
  const clamped = available ? Math.min(Math.max(score, 0), 10) : 0;
  const percentage = clamped * 10;
  const display = available ? clamped.toFixed(1) : '—';

  const dimension = size === 'lg' ? 220 : 56;
  const strokeWidth = size === 'lg' ? 12 : 6;

  return (
    <div
      role="img"
      aria-label={available ? `Score: ${display} out of 10` : 'Score unavailable'}
      /* eslint-disable-next-line no-restricted-syntax */
      style={
        scaleToFit
          ? {
              width: '100%',
              height: '100%',
              maxWidth: dimension,
              maxHeight: dimension,
              aspectRatio: 1,
            }
          : { width: dimension, height: dimension }
      }
    >
      <svg
        width={scaleToFit ? '100%' : dimension}
        height={scaleToFit ? '100%' : dimension}
        viewBox="0 0 100 100"
        /* eslint-disable-next-line no-restricted-syntax */
        style={{ display: 'block' }}
      >
        {size === 'lg' && available ? (
          gradientSegments.map((segment, index) => (
            <path
              key={index}
              d={segment.d}
              fill="none"
              stroke={segment.stroke}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              data-testid="gauge-gradient-segment"
            />
          ))
        ) : (
          <>
            <path
              d={arcPath(0, 100)}
              fill="none"
              stroke="var(--background-color-accent-gray)"
              strokeWidth={strokeWidth}
              strokeLinecap="round"
            />
            {available && (
              <path
                d={arcPath(0, percentage)}
                fill="none"
                stroke={scoreColor(clamped)}
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                data-testid="gauge-progress"
              />
            )}
          </>
        )}

        {size === 'lg' && available && (
          <g data-testid="gauge-marker">
            <circle
              cx={point(ARC_START + (ARC_SWEEP * percentage) / 100).x}
              cy={point(ARC_START + (ARC_SWEEP * percentage) / 100).y}
              r={strokeWidth / 2 + 2}
              fill="none"
              stroke="rgba(255, 255, 255, 0.5)"
              strokeWidth={2}
            />
            <circle
              cx={point(ARC_START + (ARC_SWEEP * percentage) / 100).x}
              cy={point(ARC_START + (ARC_SWEEP * percentage) / 100).y}
              r={strokeWidth / 2}
              fill="#ffffff"
              stroke="rgba(0, 0, 0, 0.35)"
              strokeWidth={1}
            />
          </g>
        )}

        <text
          x={CENTER}
          y={size === 'lg' ? 48 : 52}
          textAnchor="middle"
          dominantBaseline="central"
          data-testid="gauge-display"
          /* eslint-disable-next-line no-restricted-syntax */
          style={{
            fontSize: size === 'lg' ? 26 : 24,
            fontWeight: 700,
            fill: available ? 'currentColor' : 'var(--background-color-accent-gray)',
          }}
        >
          {display}
        </text>
        {size === 'lg' && (
          <text
            x={CENTER}
            y={64}
            textAnchor="middle"
            dominantBaseline="central"
            /* eslint-disable-next-line no-restricted-syntax */
            style={{ fontSize: 9, fontWeight: 500, fill: 'currentColor', fillOpacity: 0.6 }}
          >
            of 10
          </text>
        )}
      </svg>
    </div>
  );
};
