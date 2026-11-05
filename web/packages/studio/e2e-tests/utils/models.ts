// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Page, expect } from '@playwright/test';

export const expectChatResponseToContain = async (page: Page, text: string) => {
  await expect(page.locator('[data-testid="chat-message-content-text"]').last()).toContainText(
    new RegExp(text, 'i')
  );
};
