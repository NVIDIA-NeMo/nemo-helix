// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { destroy, initialize, type NebulaState } from '@nemo/common/src/components/Nebula/animate';
import type { NebulaProps } from '@nemo/common/src/components/Nebula/types';
import { useCallback, useEffect, useRef } from 'react';

export type {
  NebulaAmbientVariant,
  NebulaColor,
  NebulaCommonProps,
  NebulaProps,
  NebulaShape,
  NebulaSphereVariant,
  NebulaVariant,
} from '@nemo/common/src/components/Nebula/types';

export const Nebula = ({
  variant = 'sphere',
  color = 'brand',
  shape = 'triangle',
  className = '',
  ...props
}: NebulaProps) => {
  const sphereProps =
    variant === 'sphere' ? (props as { x?: number; y?: number; sphereSize?: number }) : null;
  const x = sphereProps?.x;
  const y = sphereProps?.y;
  const sphereSize = sphereProps?.sphereSize ?? 20;

  const nebulaState = useRef<NebulaState>({
    xOffset: x,
    yOffset: y,
    lastYOffset: y ?? 0,
    lastXOffset: x ?? 0,
    variant,
    initialized: false,
    sphereSize,
    particles: [],
    lastSphereSize: 0,
    center: { x: 0, y: 0 },
    canvas: null,
    ctx: null,
    appearance: { color, shape },
    animationFrameId: null,
  });

  useEffect(() => {
    if (!nebulaState.current) return;
    if (typeof x === 'number') {
      nebulaState.current.xOffset = x;
    }
    if (typeof y === 'number') {
      nebulaState.current.yOffset = y;
    }
    nebulaState.current.sphereSize = sphereSize;
  }, [x, y, sphereSize]);

  useEffect(() => {
    if (!nebulaState.current) return;
    const nextAppearance = { color, shape };
    nebulaState.current.appearance = nextAppearance;
    nebulaState.current.particles.forEach((particle) => {
      particle.setAppearance(nextAppearance);
    });
  }, [color, shape]);

  const handleRef = useCallback(
    (canvasElement: HTMLCanvasElement | null) => {
      if (canvasElement) {
        initialize(nebulaState, canvasElement, variant);
      } else {
        destroy(nebulaState);
      }
    },
    [variant]
  );

  return (
    <div
      className={`relative size-full min-h-[200px] min-w-[200px] ${className}`}
      data-testid="nv-nebula"
    >
      <canvas
        ref={handleRef}
        className="pointer-events-none absolute inset-[0]"
        aria-hidden="true"
        tabIndex={-1}
      />
    </div>
  );
};
