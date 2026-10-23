// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Moon, SunHigh } from '@nemo/icons/icons';
import { Button, Theme, Tooltip, useTheme } from '@nvidia/foundations-react-core';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { UI_THEME } from '@studio/util/localStorage';

/**
 * @remarks
 * A component that will render a CTA to toggle between light and dark themes.
 */
export function ThemeSwitch() {
  const { theme, setTheme } = useTheme();
  const [, setSavedTheme] = useLocalStorage<Theme>(UI_THEME, 'system');

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
