// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import cn from 'classnames';
import type { FC, ReactNode } from 'react';

interface GradientBackgroundProps {
  children: ReactNode;
  className?: string;
}

/**
 * Wrapper component for a gradient background.
 *
 * @example
 * ```tsx
 * <GradientBackground>
 *   <Stack gap="density-3xl" padding="density-2xl" className="relative">
 *     {content}
 *   </Stack>
 * </GradientBackground>
 * ```
 */
export const GradientBackground: FC<GradientBackgroundProps> = ({ children, className }) => {
  return (
    <div className={cn('relative min-h-full h-full', className)}>
      <div
        className={cn(
          'pointer-events-none absolute left-1/2 -translate-x-1/2 top-[-300px] w-full h-[435px] rounded-full opacity-15 blur-[100px]',
          'bg-[linear-gradient(270deg,#7CD7FE_1%,#5FE0FA_14.29%,#3DE9EE_28.57%,#23F1DB_42.86%,#37F6C1_57.14%,#63F89E_71.43%,#91F773_85.71%,#BFF230_100%)]'
        )}
      />
      {children}
    </div>
  );
};
