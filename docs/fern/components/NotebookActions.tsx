/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export interface NotebookActionsProps {
  colabUrl?: string;
}

export function NotebookActions({ colabUrl }: NotebookActionsProps) {
  if (!colabUrl) {
    return null;
  }

  return (
    <div className="notebook-actions">
      <a
        href={colabUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="notebook-actions__button notebook-actions__button--primary"
      >
        <span aria-hidden="true">&#9654;</span>
        <span>Run in Google Colab</span>
      </a>
    </div>
  );
}

export default NotebookActions;
