// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AmbientParticle } from '@nemo/common/src/components/Nebula/particles/AmbientParticle';
import { type Particle } from '@nemo/common/src/components/Nebula/particles/Base';
import { SphereParticle } from '@nemo/common/src/components/Nebula/particles/SphereParticle';
import type {
  NebulaAppearance,
  NebulaCenter,
  NebulaVariant,
} from '@nemo/common/src/components/Nebula/types';
import type { RefObject } from 'react';

export interface NebulaState {
  xOffset?: number;
  yOffset?: number;
  lastYOffset: number;
  lastXOffset: number;
  variant: NebulaVariant;
  initialized: boolean;
  sphereSize: number;
  particles: Particle[];
  lastSphereSize: number;
  center: NebulaCenter;
  canvas: HTMLCanvasElement | null;
  ctx: CanvasRenderingContext2D | null;
  appearance: NebulaAppearance;
  animationFrameId: number | null;
}

const CONNECTION_OPACITY = 0.15;
const SIZE_CHANGE_SENSITIVITY = 0.02;
const PARTICLE_COUNT = 250;

const lerp = (start: number, end: number, t: number): number => start + (end - start) * t;

const easeInLerp = (start: number, end: number, t: number): number => {
  const easedT = t * t * t;
  return lerp(start, end, easedT);
};

const resizeCanvas = (nebulaState: RefObject<NebulaState>): void => {
  if (!nebulaState.current?.ctx || !nebulaState.current.canvas) return;
  const { canvas } = nebulaState.current;
  const parent = canvas.parentElement;
  if (!parent) return;
  canvas.width = parent.clientWidth;
  canvas.height = parent.clientHeight;
  nebulaState.current.center = {
    x:
      nebulaState.current.xOffset === undefined
        ? canvas.width / 2
        : nebulaState.current.lastXOffset,
    y:
      nebulaState.current.yOffset === undefined
        ? canvas.height / 2
        : nebulaState.current.lastYOffset,
  };
};

const drawConnections = (
  ctx: CanvasRenderingContext2D,
  i: number,
  particles: Particle[],
  connectionDistance: number
): void => {
  for (let j = i + 1; j < particles.length; j++) {
    const row = particles[i];
    const column = particles[j];
    if (!row || !column) continue;
    const dx = row.x - column.x;
    const dy = row.y - column.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < connectionDistance) {
      const chosenColor = Math.random() < 0.5 ? row.color : column.color;
      ctx.beginPath();
      ctx.moveTo(row.x, row.y);
      ctx.lineTo(column.x, column.y);
      ctx.strokeStyle = chosenColor.replace(/[\d.]+\)$/g, `${CONNECTION_OPACITY})`);
      ctx.lineWidth = 0.4;
      ctx.stroke();
    }
  }
};

export const animate = (nebulaState: RefObject<NebulaState>) => (): void => {
  if (!nebulaState.current?.ctx || !nebulaState.current.canvas) return;
  resizeCanvas(nebulaState);
  const { ctx } = nebulaState.current;
  ctx.clearRect(0, 0, nebulaState.current.canvas.width, nebulaState.current.canvas.height);

  let scaleFactor = 1;
  if (
    nebulaState.current.yOffset !== undefined &&
    Math.abs(nebulaState.current.yOffset - nebulaState.current.lastYOffset) > 0.1
  ) {
    nebulaState.current.lastYOffset = easeInLerp(
      nebulaState.current.lastYOffset,
      nebulaState.current.yOffset,
      0.35
    );
  }
  if (
    nebulaState.current.xOffset !== undefined &&
    Math.abs(nebulaState.current.xOffset - nebulaState.current.lastXOffset) > 0.1
  ) {
    nebulaState.current.lastXOffset = easeInLerp(
      nebulaState.current.lastXOffset,
      nebulaState.current.xOffset,
      0.35
    );
  }
  if (Math.abs(nebulaState.current.sphereSize - nebulaState.current.lastSphereSize) > 0.1) {
    const newSphereSize = lerp(
      nebulaState.current.lastSphereSize,
      nebulaState.current.sphereSize,
      SIZE_CHANGE_SENSITIVITY
    );
    scaleFactor = newSphereSize / nebulaState.current.lastSphereSize;
    nebulaState.current.lastSphereSize = newSphereSize;
  }
  const connectionDistance =
    nebulaState.current.variant === 'sphere' ? nebulaState.current.lastSphereSize * 1.2 : 45;

  nebulaState.current.particles.forEach((particle, i) => {
    drawConnections(ctx, i, nebulaState.current.particles, connectionDistance);
    if (particle instanceof SphereParticle) {
      particle.scaleOrbit(nebulaState.current.center, scaleFactor);
    }
    particle.update(nebulaState.current.lastSphereSize);
    particle.draw(ctx);
  });
  nebulaState.current.animationFrameId = requestAnimationFrame(animate(nebulaState));
};

export const destroy = (nebulaState: RefObject<NebulaState>): void => {
  if (!nebulaState.current) return;

  if (nebulaState.current.animationFrameId !== null) {
    cancelAnimationFrame(nebulaState.current.animationFrameId);
  }

  nebulaState.current.animationFrameId = null;
  nebulaState.current.initialized = false;
  nebulaState.current.particles = [];
  nebulaState.current.canvas = null;
  nebulaState.current.ctx = null;
};

export const initialize = (
  nebulaState: RefObject<NebulaState>,
  canvas: HTMLCanvasElement | null,
  variant: NebulaVariant
): void => {
  if (!canvas || !nebulaState.current || nebulaState.current.initialized) return;
  nebulaState.current.initialized = true;
  nebulaState.current.lastSphereSize = nebulaState.current.sphereSize;
  nebulaState.current.lastYOffset = nebulaState.current.yOffset ?? 0;
  nebulaState.current.lastXOffset = nebulaState.current.xOffset ?? 0;
  const appearance = nebulaState.current.appearance ?? { color: 'brand', shape: 'triangle' };
  nebulaState.current.appearance = appearance;
  const ctx = canvas.getContext('2d', { alpha: true });
  nebulaState.current.canvas = canvas;
  nebulaState.current.ctx = ctx;

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const particle =
      variant === 'sphere'
        ? new SphereParticle(
            nebulaState.current.center,
            nebulaState.current.sphereSize / 10,
            appearance
          )
        : new AmbientParticle(canvas, appearance);
    nebulaState.current.particles.push(particle);
  }
  nebulaState.current.animationFrameId = requestAnimationFrame(animate(nebulaState));
};
