/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Button — a styled link rendered as a CTA button.
 *
 * Replaces MkDocs Material's `{ .md-button }` / `{ .md-button--primary }`
 * attribute list. The converter emits this wrapper when it sees that
 * pattern in the docs source.
 *
 * NOTE: Fern's custom component pipeline uses the automatic JSX runtime;
 * we do not import React.
 *
 * Usage in MDX:
 *   <Button href="/foo">Open the foo guide</Button>
 *   <Button href="/foo" variant="primary">Primary CTA</Button>
 *
 * In MDX you can also pass a markdown link via children if you prefer:
 *   <Button href="/foo">**Bold label**</Button>
 */

import type { ReactNode } from "react";

export interface ButtonProps {
  href: string;
  variant?: "primary" | "secondary";
  children: ReactNode;
}

export function Button({ href, variant = "secondary", children }: ButtonProps) {
  const external = /^https?:\/\//.test(href);
  return (
    <a
      className={`docs-cta-button docs-cta-button--${variant}`}
      href={href}
      {...(external ? { target: "_blank", rel: "noreferrer noopener" } : {})}
    >
      {children}
    </a>
  );
}

export default Button;
