// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Button, Flex, Stack } from '@nvidia/foundations-react-core';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { type FC, memo } from 'react';

export interface BuilderDetailsPanelProps {
  validationErrors: string[];
  submitError: string | null;
  isOpen: boolean;
  onToggle: () => void;
}
export const BuilderDetailsPanel: FC<BuilderDetailsPanelProps> = memo(function BuilderDetailsPanel({
  validationErrors,
  submitError,
  isOpen,
  onToggle,
}) {
  const hasDetails = validationErrors.length > 0 || !!submitError;
  if (!hasDetails) return null;

  const summary = [
    validationErrors.length > 0 && `${validationErrors.length} validation issue(s)`,
    submitError && 'Job creation error',
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div className="shrink-0 border-b border-base px-density-2xl">
      <Button
        kind="tertiary"
        color="neutral"
        size="small"
        className="py-density-sm"
        onClick={onToggle}
      >
        <Flex align="center" gap="density-xs">
          {isOpen ? <ChevronDown size={14} aria-hidden /> : <ChevronRight size={14} aria-hidden />}
          {summary}
        </Flex>
      </Button>

      {isOpen && (
        <Stack gap="density-sm" className="pb-density-lg">
          {validationErrors.length > 0 && (
            <Banner kind="inline" status="error">
              Please fix the following before continuing:
              <ul className="list-disc pl-density-lg">
                {validationErrors.map((error) => (
                  <li key={error}>{error}</li>
                ))}
              </ul>
            </Banner>
          )}
          {submitError && (
            <Banner kind="inline" status="error">
              There was an error creating this job: {submitError}
            </Banner>
          )}
        </Stack>
      )}
    </div>
  );
});
