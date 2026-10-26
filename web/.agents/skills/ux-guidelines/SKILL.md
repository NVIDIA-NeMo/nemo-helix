---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: ux-guidelines
description: UX design guidelines and standards for NeMo Studio. Use when implementing new UI features, reviewing designs, or making UX decisions.
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# UX Guidelines

Detailed best practices provided by the Studio UI/UX Design team.

## When to use this skill

- For features that change the UI.
- For brand new views or features.
- Whenever you change the user experience.

## Guidelines

### General

- **Always** Use Kaizen UI (KUI) styles, tokens, variables, and components whenever possible.

### Buttons

- Primary call to action (CTA) buttons generally follow the `{Verb} {Noun}` syntax, like `"Create Model"`.
- **Avoid** verbose labels inside buttons. Button labels should be succinct.

### Language and microcopy

- Use title case for things like page or section titles, buttons, and form fields. Use sentence case for other copy like descriptions or paragraphs.
- **Never** use emojis.

### Menus

- **Avoid** using icons in popover menus or `<QuickActionsMenuRoot>` items.

### Forms and Form elements

- In general, form fields should **rarely** include placeholder values. Unless the design explicitly calls for them, do not include placeholders in form fields.
- Form labels should **always** be title case (e.g., "First Name" instead of "First name").
- **Do not** disable submit buttons. Instead of disabling, enable the button and use it to trigger error messages for required fields upon submission.
- If a Form Field is optional, include that language in the label. (e.g., "Description (optional)").
- Forms should follow standard browser behavior and submit when the user presses "enter"

### Modals and Dialogs

- Avoid adding icons to `slotHeading` within `<Modal>`. If you do not use `slotHeading`, still avoid including icons near the title of a Modal.
- If a dialog will perform a destructive action, use `color="danger"` for the primary button.
- If a dialog confirms an action, such as "Delete {item}", display the item's name if possible. Example: "Delete my-project-name?". If the action affects multiple items at once, use a count instead (e.g., "Delete 2 Models?")
