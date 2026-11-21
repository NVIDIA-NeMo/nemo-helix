// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type JobSound = 'success' | 'failure';

/** A rising pair reads as "done"; a lower, falling pair reads as "something went wrong". */
const NOTES_HZ: Record<JobSound, readonly number[]> = {
  success: [659.25, 880],
  failure: [440, 329.63],
};

const NOTE_SECONDS = 0.18;
const ATTACK_SECONDS = 0.02;
/** Quiet enough not to startle; this plays while the user is away from the screen. */
const PEAK_GAIN = 0.2;
const SILENCE_GAIN = 0.001;

let audioContext: AudioContext | undefined;

const getAudioContext = (): AudioContext | undefined => {
  if (typeof AudioContext === 'undefined') return undefined;
  audioContext ??= new AudioContext();
  return audioContext;
};

/**
 * Plays a short generated chime. Browsers give a web notification no way to name its sound, and
 * some platforms play none, so the page makes its own.
 *
 * Browsers only allow audio on a page the user has already interacted with, so this can be
 * silently blocked (a freshly reloaded tab, for instance). That is not an error: nothing
 * plays, and the notification itself is unaffected.
 */
export const playJobSound = (sound: JobSound): void => {
  try {
    const context = getAudioContext();
    if (!context) return;
    // A context created before user interaction starts suspended; resuming is a no-op once it
    // is running, and stays pending (harmlessly) if the browser still refuses.
    void context.resume().catch(() => undefined);

    const start = context.currentTime;
    NOTES_HZ[sound].forEach((frequency, index) => {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const noteStart = start + index * NOTE_SECONDS;
      const noteEnd = noteStart + NOTE_SECONDS;

      oscillator.type = 'sine';
      oscillator.frequency.value = frequency;
      // A short ramp in and out avoids the audible click of starting at full volume.
      gain.gain.setValueAtTime(0, noteStart);
      gain.gain.linearRampToValueAtTime(PEAK_GAIN, noteStart + ATTACK_SECONDS);
      gain.gain.exponentialRampToValueAtTime(SILENCE_GAIN, noteEnd);

      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(noteStart);
      oscillator.stop(noteEnd);
    });
  } catch {
    // Audio is a nicety; never let it break notifications.
  }
};
