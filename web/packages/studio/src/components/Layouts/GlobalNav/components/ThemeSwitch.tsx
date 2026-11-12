// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Moon, SunHigh } from '@nemo/icons/icons';
import { Button, type Theme, Tooltip, useTheme } from '@nvidia/foundations-react-core';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { usePreference } from '@studio/providers/preferences/usePreference';

/**
 * @remarks
 * A component that will render a CTA to toggle between light and dark themes.
 */
export function ThemeSwitch() {
  const { theme, setTheme } = useTheme();
  const [, setSavedTheme] = usePreference(studioPreferences.theme);

  const changeTheme = (newTheme: Theme) => {
    setTheme(newTheme);
    setSavedTheme(newTheme);
  };

  return (
    <Tooltip slotContent={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`} side="bottom">
      <Button
        aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
        color="neutral"
        onClick={() => changeTheme(theme === 'light' ? 'dark' : 'light')}
        kind="tertiary"
        size="medium"
      >
        {theme === 'light' ? <Moon /> : <SunHigh />}
      </Button>
    </Tooltip>
  );
}
