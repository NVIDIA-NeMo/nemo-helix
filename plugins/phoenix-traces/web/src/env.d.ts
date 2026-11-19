// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference types="vite/client" />

// Common's components import stylesheets; typed from source, so tsc sees them.
declare module '*.css';

declare module '*.svg?react' {
  import type { FC, SVGProps } from 'react';
  const Component: FC<SVGProps<SVGSVGElement>>;
  export default Component;
}
