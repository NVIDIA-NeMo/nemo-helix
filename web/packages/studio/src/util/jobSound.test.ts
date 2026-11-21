// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { playJobSound } from '@studio/util/jobSound';

const frequencies: number[] = [];
const resume = vi.fn<() => Promise<void>>();

class FakeAudioContext {
  currentTime = 0;
  destination = {};
  resume = resume;
  createGain = () => ({
    gain: {
      setValueAtTime: vi.fn(),
      linearRampToValueAtTime: vi.fn(),
      exponentialRampToValueAtTime: vi.fn(),
    },
    connect: vi.fn(),
  });
  createOscillator = () => {
    const oscillator = {
      type: '',
      frequency: {
        set value(hz: number) {
          frequencies.push(hz);
        },
      },
      connect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
    };
    return oscillator;
  };
}

describe('playJobSound', () => {
  beforeEach(() => {
    frequencies.length = 0;
    resume.mockReset().mockResolvedValue(undefined);
    vi.stubGlobal('AudioContext', FakeAudioContext);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('plays a rising pair of notes for success', () => {
    playJobSound('success');
    expect(frequencies).toHaveLength(2);
    expect(frequencies[1]).toBeGreaterThan(frequencies[0]);
  });

  it('plays a lower, falling pair of notes for failure', () => {
    playJobSound('failure');
    expect(frequencies).toHaveLength(2);
    expect(frequencies[1]).toBeLessThan(frequencies[0]);
  });

  it('does nothing, without throwing, when the browser has no AudioContext', () => {
    vi.unstubAllGlobals();
    vi.stubGlobal('AudioContext', undefined);
    expect(() => playJobSound('success')).not.toThrow();
    expect(frequencies).toHaveLength(0);
  });

  it('does not throw when the browser refuses to start audio', () => {
    resume.mockRejectedValue(new DOMException('not allowed', 'NotAllowedError'));
    expect(() => playJobSound('success')).not.toThrow();
  });
});
