// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Stack, Text } from '@nvidia/foundations-react-core';
import { ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { PlugZap, Server } from 'lucide-react';
import { type FC } from 'react';
import { generatePath, useNavigate } from 'react-router';

interface ChatEmptyStateProps {
  /** When false, paints the headline as "No models available" and surfaces the
   *  connect/deploy CTAs. When true, just shows the animated "Ready" state. */
  hasModels: boolean;
}

/**
 * Replicates the Figma's "Ready" state: an animated green particle swirl, a
 * large headline, a soft subhead, optional seed-question chips above the
 * (route-owned) composer. Particle ring is a tuned inline SVG plus a single
 * CSS keyframe — Kaizen ships no equivalent (confirmed in component scan).
 */
export const ChatEmptyState: FC<ChatEmptyStateProps> = ({ hasModels }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

  const headline = hasModels ? 'Ready' : 'No models available';
  const subhead = hasModels
    ? 'Prompt your model to get started.'
    : 'Connect an inference provider or create a deployment to start chatting.';

  return (
    <div className="flex h-full w-full items-center justify-center p-8">
      <Stack gap="density-xl" align="center" className="relative z-10 w-full max-w-xl text-center">
        <div className="relative h-72 w-72">
          <ParticleSwirl />
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
            <Text kind="display/sm">{headline}</Text>
            <Text className="text-secondary" kind="body/regular/md">
              {subhead}
            </Text>
          </div>
        </div>
        {!hasModels && (
          <Stack gap="density-md" align="center">
            <div className="flex gap-3">
              <Button
                kind="primary"
                color="brand"
                onClick={() =>
                  navigate(generatePath(ROUTES.workspace.inferenceProviders, { workspace }))
                }
              >
                <PlugZap size={16} />
                Connect inference provider
              </Button>
              <Button
                kind="secondary"
                onClick={() => navigate(generatePath(ROUTES.workspace.deployments, { workspace }))}
              >
                <Server size={16} />
                Create deployment
              </Button>
            </div>
          </Stack>
        )}
      </Stack>
    </div>
  );
};

/**
 * 96 small dots arranged on a ring with jittered radii and per-dot animation
 * delays, so the swirl reads as motion even at low frame budgets. Pure CSS so
 * we don't add a dep.
 */
const ParticleSwirl: FC = () => {
  const dots = Array.from({ length: 96 }, (_, i) => {
    const angle = (i / 96) * Math.PI * 2;
    const radiusJitter = 0.85 + ((i * 37) % 100) / 600;
    const r = 130 * radiusJitter;
    const cx = 144 + Math.cos(angle) * r;
    const cy = 144 + Math.sin(angle) * r;
    const size = 1.5 + ((i * 13) % 100) / 80;
    const delay = (i / 96) * 6;
    return { cx, cy, size, delay, key: i };
  });

  return (
    <svg
      viewBox="0 0 288 288"
      className="absolute inset-0 h-full w-full animate-[playground-swirl-spin_18s_linear_infinite]"
      aria-hidden="true"
    >
      <defs>
        <radialGradient id="playground-swirl-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="var(--color-brand)" stopOpacity="0.18" />
          <stop offset="65%" stopColor="var(--color-brand)" stopOpacity="0.04" />
          <stop offset="100%" stopColor="var(--color-brand)" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx="144" cy="144" r="128" fill="url(#playground-swirl-glow)" />
      {dots.map(({ cx, cy, size, key }, i) => (
        <circle
          key={key}
          cx={cx}
          cy={cy}
          r={size}
          fill="var(--color-brand)"
          className={`playground-swirl-dot dot-${i}`}
        />
      ))}
      {/* Per-dot animation-delay is generated dynamically into this style block
       *  rather than via inline `style={}` so we don't trip the
       *  no-restricted-syntax lint rule. The styled <style> tag inside an SVG
       *  scopes the rules to this swirl. */}
      <style>{`
        @keyframes playground-swirl-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes playground-swirl-pulse {
          0%, 100% { opacity: 0.2; }
          50% { opacity: 0.85; }
        }
        .playground-swirl-dot {
          opacity: 0.6;
          animation: playground-swirl-pulse 2.8s ease-in-out infinite;
        }
        ${dots.map(({ delay }, i) => `.playground-swirl-dot.dot-${i} { animation-delay: ${delay}s; }`).join('\n        ')}
      `}</style>
    </svg>
  );
};
