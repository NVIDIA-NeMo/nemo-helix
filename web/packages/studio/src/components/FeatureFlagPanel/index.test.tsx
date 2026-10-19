// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { FeatureFlagPanel } from '@studio/components/FeatureFlagPanel';
import { FEATURE_FLAG_OVERRIDES_KEY } from '@studio/util/localStorage';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const readBlob = () => {
  const raw = window.localStorage.getItem(FEATURE_FLAG_OVERRIDES_KEY);
  return raw ? (JSON.parse(raw) as Record<string, string>) : null;
};

/** KUI Select is a custom listbox, not a native <select>, so drive it by role. */
const chooseOption = async (
  user: ReturnType<typeof userEvent.setup>,
  selectName: string,
  optionName: string
) => {
  const trigger = await screen.findByRole('combobox', { name: selectName });
  await user.click(trigger);

  // All 35 rows keep a listbox mounted and every one offers "true"/"false", so
  // scope to this select's own listbox, which carries the same accessible name.
  const listbox = await screen.findByRole('listbox', { name: selectName, hidden: true });

  await user.click(within(listbox).getByRole('option', { name: optionName }));
};

const openPanel = async () => {
  const user = userEvent.setup();
  render(<FeatureFlagPanel />);
  await user.click(screen.getByRole('button', { name: 'Feature flags' }));
  return user;
};

beforeEach(() => {
  window.localStorage.clear();
});

describe('FeatureFlagPanel', () => {
  it('opens from the header trigger', async () => {
    await openPanel();
    expect(screen.getByText('Feature flags')).toBeInTheDocument();
  });

  // The gate is read only from the build env, so a control for it would be a
  // no-op. Hiding it is what keeps that from being confusing.
  it('does not list the gate flag itself', async () => {
    await openPanel();
    expect(screen.queryByText('flagOverridesEnabled')).not.toBeInTheDocument();
  });

  it('lists real flags with a control each', async () => {
    await openPanel();
    expect(screen.getByText('tourEnabled')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'tourEnabled override' })).toBeInTheDocument();
  });

  it('shows no pending overrides on a clean profile', async () => {
    await openPanel();
    expect(screen.getByText(/0 overrides active/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply and reload' })).toBeDisabled();
  });

  it('reflects an override already in storage', async () => {
    window.localStorage.setItem(
      FEATURE_FLAG_OVERRIDES_KEY,
      JSON.stringify({ tourEnabled: 'false' })
    );

    await openPanel();
    expect(screen.getByText(/1 override active/)).toBeInTheDocument();
  });

  it('filters by flag name', async () => {
    const user = await openPanel();
    await user.type(screen.getByLabelText('Filter flags'), 'tourEnabled');

    expect(screen.getByText('tourEnabled')).toBeInTheDocument();
    expect(screen.queryByText('agentsEnabled')).not.toBeInTheDocument();
  });

  it('reports when no flag matches the filter', async () => {
    const user = await openPanel();
    await user.type(screen.getByLabelText('Filter flags'), 'zzzznope');

    expect(screen.getByText(/No flags match/)).toBeInTheDocument();
  });

  it('writes the blob and reloads on apply', async () => {
    const reload = vi.fn();
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...window.location, reload },
    });

    const user = await openPanel();
    await chooseOption(user, 'tourEnabled override', 'false');
    await user.click(screen.getByRole('button', { name: 'Apply and reload' }));

    expect(readBlob()).toEqual({ tourEnabled: 'false' });
    expect(reload).toHaveBeenCalledOnce();
  });

  // "Default" is not the same as false — it means no entry in the blob, so the
  // flag falls through to whatever the deployment ships.
  it('clearing the last override removes the key entirely', async () => {
    const reload = vi.fn();
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...window.location, reload },
    });
    window.localStorage.setItem(
      FEATURE_FLAG_OVERRIDES_KEY,
      JSON.stringify({ tourEnabled: 'false' })
    );

    const user = await openPanel();
    await user.click(screen.getByRole('button', { name: 'Clear tourEnabled override' }));
    await user.click(screen.getByRole('button', { name: 'Apply and reload' }));

    expect(readBlob()).toBeNull();
  });

  it('reset all clears every pending override', async () => {
    window.localStorage.setItem(
      FEATURE_FLAG_OVERRIDES_KEY,
      JSON.stringify({ tourEnabled: 'false' })
    );

    const user = await openPanel();
    await user.click(screen.getByRole('button', { name: 'Reset all' }));

    expect(screen.getByText(/0 overrides pending/)).toBeInTheDocument();
  });

  it('surfaces a failed save instead of claiming success', async () => {
    const reload = vi.fn();
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...window.location, reload },
    });
    // The failed write is reported through logger.error by design.
    vi.spyOn(logger, 'error').mockImplementation(() => {});

    // jsdom's localStorage does not route through Storage.prototype, so stub
    // the object itself: reads keep working, writes reject.
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        getItem: () => null,
        setItem: () => {
          throw new DOMException('Quota exceeded', 'QuotaExceededError');
        },
        removeItem: () => {
          throw new DOMException('Quota exceeded', 'QuotaExceededError');
        },
        clear: () => {},
      },
    });

    const user = await openPanel();
    await chooseOption(user, 'tourEnabled override', 'false');
    await user.click(screen.getByRole('button', { name: 'Apply and reload' }));

    expect(screen.getByText(/Could not save/)).toBeInTheDocument();
    expect(reload).not.toHaveBeenCalled();
  });
});
