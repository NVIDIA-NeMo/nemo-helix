// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExpandableMessage } from '@nemo/common';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';

interface InterviewQA {
  question?: string;
  answer?: string;
  gap?: string;
}

interface BenignInterviewCardProps {
  interview: InterviewQA[];
}

// The interview Q&A captured during the last benign-suite generation — the "why" behind the current suite.
export const BenignInterviewCard: FC<BenignInterviewCardProps> = ({ interview }) => {
  if (interview.length === 0) return null;
  return (
    <Stack gap="density-md">
      {interview.map((qa, index) => (
        <Stack key={index} gap="density-xs" className="rounded-md border border-base p-3">
          <Text kind="body/semibold/sm">{qa.question || qa.gap || `Question ${index + 1}`}</Text>
          <ExpandableMessage message={qa.answer || '(no answer)'} characterLimit={220} />
        </Stack>
      ))}
    </Stack>
  );
};
