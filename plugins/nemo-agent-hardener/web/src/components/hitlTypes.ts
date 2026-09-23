// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shapes the war-game job publishes on the platform job's `status_details` for human-in-the-loop.
// The job's StatusDetailsChannel writes `interview`/`review` (round-stamped); the UI answers by
// PATCHing `interview_response`/`review_response` with the matching round.

export interface InterviewOption {
  label: string;
  description: string;
  recommended?: boolean;
}

export interface InterviewQuestion {
  gap: string;
  question: string;
  options?: InterviewOption[];
}

export interface InterviewAnswer {
  gap: string;
  question: string;
  answer: string;
}

export interface SuiteRow {
  tool: string;
  payload: string;
  label?: string;
  rationale?: string;
  persona?: string;
}

export interface InterviewPrompt {
  round: number;
  questions: InterviewQuestion[];
}

export interface ReviewPrompt {
  round: number;
  suite: SuiteRow[];
}

// Read the current interview/review prompt out of a job's free-form status_details, skipping any
// round the operator has already responded to.
export const pendingInterview = (
  details: Record<string, unknown> | undefined
): InterviewPrompt | null => {
  const interview = details?.interview as InterviewPrompt | undefined;
  const response = details?.interview_response as { round?: number } | undefined;
  if (!interview || typeof interview.round !== 'number') return null;
  if (response?.round === interview.round) return null;
  return interview;
};

export const pendingReview = (
  details: Record<string, unknown> | undefined
): ReviewPrompt | null => {
  const review = details?.review as ReviewPrompt | undefined;
  const response = details?.review_response as { round?: number } | undefined;
  if (!review || typeof review.round !== 'number') return null;
  if (response?.round === review.round) return null;
  return review;
};
