// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BURST_COUNT, BURST_INTERVAL_MS } from '@studio/components/CelebrationOverlay/constants';
import { type FC, useEffect, useRef } from 'react';

const PARTICLES_PER_BURST = 70;
const GRAVITY = 0.05;
const FADE_PER_FRAME = 0.011;
const FRAME_MS = 1000 / 60;

/** Token classes whose computed colour becomes a particle colour, brand green weighted double. */
const SWATCH_CLASSES = [
  'text-brand',
  'text-brand',
  'text-feedback-warning',
  'text-feedback-danger',
  'text-feedback-info',
  'text-accent-blue',
];

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  alpha: number;
  color: string;
}

const burst = (width: number, height: number, colors: string[]): Particle[] => {
  const x = width * (0.1 + Math.random() * 0.8);
  const y = height * (0.15 + Math.random() * 0.45);
  const color = colors[Math.floor(Math.random() * colors.length)] ?? 'currentColor';
  return Array.from({ length: PARTICLES_PER_BURST }, () => {
    const angle = Math.random() * Math.PI * 2;
    const speed = 1.5 + Math.random() * 4.5;
    return {
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      alpha: 1,
      color,
    };
  });
};

/** Staggered firework bursts across the viewport, drawn on one canvas. */
export const Fireworks: FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const swatchesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }

    const ratio = window.devicePixelRatio || 1;
    const width = window.innerWidth;
    const height = window.innerHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.scale(ratio, ratio);

    const colors = Array.from(swatchesRef.current?.children ?? []).map(
      (swatch) => getComputedStyle(swatch).color
    );
    const particles: Particle[] = [];
    const timers = Array.from({ length: BURST_COUNT }, (_, index) =>
      window.setTimeout(() => {
        particles.push(...burst(width, height, colors));
      }, index * BURST_INTERVAL_MS)
    );

    let frame = 0;
    let last = performance.now();
    const draw = (now: number) => {
      const step = Math.min((now - last) / FRAME_MS, 3);
      last = now;
      context.clearRect(0, 0, width, height);
      for (let index = particles.length - 1; index >= 0; index -= 1) {
        const particle = particles[index];
        particle.vy += GRAVITY * step;
        particle.x += particle.vx * step;
        particle.y += particle.vy * step;
        particle.alpha -= FADE_PER_FRAME * step;
        if (particle.alpha <= 0) {
          particles.splice(index, 1);
          continue;
        }
        context.globalAlpha = particle.alpha;
        context.fillStyle = particle.color;
        context.beginPath();
        context.arc(particle.x, particle.y, 2.5, 0, Math.PI * 2);
        context.fill();
      }
      frame = window.requestAnimationFrame(draw);
    };
    frame = window.requestAnimationFrame(draw);

    return () => {
      timers.forEach((timer) => window.clearTimeout(timer));
      window.cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <>
      <div ref={swatchesRef} hidden>
        {SWATCH_CLASSES.map((className, index) => (
          <span key={index} className={className} />
        ))}
      </div>
      <canvas
        ref={canvasRef}
        data-testid="celebration-fireworks"
        aria-hidden
        className="pointer-events-none absolute inset-0 h-full w-full"
      />
    </>
  );
};
