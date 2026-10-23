// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Download } from '@nemo/icons/icons';
import { Button, ButtonProps } from '@nvidia/foundations-react-core';
import { FC } from 'react';
import { Link } from 'react-router';

type Props = {
  evaluatorUrl: string;
  evaluationJobId: string;
  compact?: boolean; // Removes Label
  size?: ButtonProps['size'];
  tone?: ButtonProps['kind'];
};
export const DownloadEvaluationResultsButton: FC<Props> = ({
  evaluatorUrl,
  evaluationJobId,
  compact = false,
  size = undefined,
  tone = 'tertiary',
}) => {
  const route = `${evaluatorUrl}/v1/evaluation/jobs/${evaluationJobId}/download-results`;

  return (
    <Link to={route} target="_blank" rel="noopener noreferrer">
      <Button kind={tone} size={size}>
        <Download width="16px" height="16px" />
        {compact ? null : 'Download Results'}
      </Button>
    </Link>
  );
};
