// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Anchor, Skeleton, Stack, Text } from '@nvidia/foundations-react-core';
import { type ComponentProps, FC, useState } from 'react';

const DEFAULT_CHARACTER_LIMIT = 300;

interface Props {
  /**
   * The message to display.
   */
  message?: string;
  /**
   * The message to display when there is an error. If this is provided, this takes priority over the message prop.
   */
  errorMessage?: string;
  /**
   * If true, a skeleton will be displayed.
   */
  loading?: boolean;
  /**
   * Character limit before showing "Show more" button. Defaults to 300.
   */
  characterLimit?: number;
  attributes?: {
    Anchor?: ComponentProps<typeof Anchor>;
    Text?: ComponentProps<typeof Text>;
  };
}

export const ExpandableMessage: FC<Props> = ({
  message,
  loading,
  errorMessage,
  characterLimit = DEFAULT_CHARACTER_LIMIT,
  attributes,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const contentToDisplay = errorMessage || message || '';
  const exceedsLimit = contentToDisplay.length > characterLimit;
  const displayedContent =
    exceedsLimit && !isExpanded
      ? `${contentToDisplay.slice(0, characterLimit)}...`
      : contentToDisplay;

  const toggleExpanded = () => setIsExpanded((prev) => !prev);

  let content = null;
  if (loading) {
    return <Skeleton className="h-4 w-full" />;
  } else if (errorMessage) {
    content = (
      <Text
        className="text-feedback-danger whitespace-pre-wrap"
        kind="mono/md"
        {...attributes?.Text}
      >
        {displayedContent}
      </Text>
    );
  } else {
    content = (
      <Text className="whitespace-pre-wrap" kind="mono/md" {...attributes?.Text}>
        {displayedContent}
      </Text>
    );
  }
  return (
    <Stack gap="2">
      {content}
      {exceedsLimit && (
        <Anchor asChild {...attributes?.Anchor}>
          <button type="button" aria-expanded={isExpanded} onClick={toggleExpanded}>
            {isExpanded ? 'Show less' : 'Show more'}
          </button>
        </Anchor>
      )}
    </Stack>
  );
};
