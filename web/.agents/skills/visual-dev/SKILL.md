---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: visual-dev
description: Screenshot-driven visual development workflow using Playwright MCP. Use when implementing designs from mockups or Figma, building new UI components, or fixing visual bugs.
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Visual Development Workflow

## When to Use This Workflow

- Implementing designs from mockups, Figma, or design specifications
- Building new UI components that need precise visual matching
- Fixing visual bugs or inconsistencies
- Creating responsive layouts that match design breakpoints
- Iterating on complex visual interactions or animations

## Prerequisites

Playwright MCP must be available for browser automation and screenshots.

## Visual Development Process

### 1. Screenshot-Driven Analysis

- **Take initial screenshots** of current implementation state
- **Compare with design specifications** (mockups, Figma, etc.)
- **Identify visual gaps** — spacing, colors, typography, layout, interactions
- **Document specific differences** before making changes

### 2. Iterative Implementation Loop

```
Take Screenshot → Analyze vs Design → Make Code Changes → Navigate & Test → Repeat
```

### 3. Implementation Strategy

- **Start with structure** — get HTML/component hierarchy correct
- **Apply base styles** — colors, typography, basic spacing
- **Refine layout** — flexbox, grid, positioning details
- **Add interactions** — hover states, focus states, animations
- **Verify accessibility** — focus indicators, contrast, screen readers

## Screenshot & Analysis

- Take full page screenshots for layout verification
- Take element-specific screenshots for component comparison
- Compare multiple viewport sizes (mobile, tablet, desktop)
- Capture different states (hover, focus, loading, error)

## Design Matching Criteria

- **Spacing & Layout** — margins, padding, alignment match design specs
- **Typography** — font families, sizes, weights, line heights correct
- **Colors** — exact color values from design tokens/system
- **Interactive States** — proper hover, focus, active, disabled states
- **Responsive Behavior** — breakpoints and layout shifts as designed

## Iteration Guidelines

- **Take screenshots after each major change** for progress tracking
- **Compare side-by-side** with design specifications
- **Test multiple scenarios** — different content lengths, edge cases
- **Verify across browsers** if cross-browser compatibility is required

## Code Quality During Visual Work

- **Use design tokens** from KUI/design system when available
- **Avoid magic numbers** — use named spacing/sizing variables
- **Test with realistic content** — not just placeholder text

## Completion Criteria

- [ ] Screenshots match design specifications closely
- [ ] All interactive states implemented and tested
- [ ] Accessibility requirements met (focus, contrast, screen readers)
- [ ] Performance impact assessed (no layout thrashing, smooth animations)

## When to Ask for Design Clarification

- Ambiguous spacing or sizing in design specifications
- Missing interactive states not shown in mockups
- Unclear responsive behavior at different breakpoints
- Accessibility considerations not addressed in designs
