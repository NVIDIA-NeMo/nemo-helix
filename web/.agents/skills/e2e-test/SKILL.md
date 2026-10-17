---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: e2e-test
description: End-to-end testing workflow with Playwright. Use when writing, running, or debugging E2E tests for Studio user flows.
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# E2E Testing Workflow

## Test Strategy

- Test critical user journeys end-to-end
- Focus on happy paths and most common user workflows
- Verify integrations between different parts of the application
- Test real user scenarios, not just technical functionality

## Page Object Model

- Use Page Object Model for complex user flows
- Encapsulate page interactions in reusable page classes
- Keep selectors centralized in page objects
- Use meaningful method names that describe user actions
- Return page objects from navigation methods for chaining

## Playwright Best Practices

- Implement proper wait strategies (avoid arbitrary timeouts)
- Use Playwright's auto-waiting and retry mechanisms
- Leverage built-in assertions with `expect(page).toHave...`
- Use `page.locator()` over deprecated `page.$()` methods
- Take screenshots on test failures for debugging

## Selectors and Interactions

- Prefer user-facing selectors (text, labels, roles)
- Use `data-testid` attributes for complex components
- Avoid CSS selectors that depend on styling
- Use Playwright's built-in locator methods
- Chain actions fluently: `page.locator('button').click()`

## Test Organization

- Group tests by user journey or feature area
- Use descriptive test names that explain the user scenario
- Set up test data in beforeEach hooks when needed
- Clean up test data after tests complete
- Use test fixtures for complex setup scenarios

## Configuration

- Set appropriate timeouts for different types of operations
- Use headless mode in CI, headed mode for debugging
- Configure test parallelization appropriately

## Debugging

- Use Playwright's trace viewer for debugging failures
- Record videos of test runs for failure analysis
- Run tests in headed mode during development
- Use `page.pause()` for interactive debugging
- Keep tests stable by avoiding flaky selectors
