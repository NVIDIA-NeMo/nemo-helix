/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { MouseEvent } from "react";

export interface NotebookActionsProps {
  colabUrl?: string;
  downloadUrl?: string;
  filename?: string;
}

function filenameFromUrl(url: string): string {
  const pathname = new URL(url).pathname;
  return decodeURIComponent(pathname.split("/").pop() || "notebook.ipynb");
}

async function downloadNotebook(url: string, filename: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download notebook: ${response.status}`);
  }
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export function NotebookActions({ colabUrl, downloadUrl, filename }: NotebookActionsProps) {
  const resolvedFilename = filename ?? (downloadUrl ? filenameFromUrl(downloadUrl) : "notebook.ipynb");

  const handleDownload = (event: MouseEvent<HTMLAnchorElement>) => {
    if (!downloadUrl) {
      return;
    }
    event.preventDefault();
    downloadNotebook(downloadUrl, resolvedFilename).catch(() => {
      window.location.href = downloadUrl;
    });
  };

  if (!colabUrl && !downloadUrl) {
    return null;
  }

  return (
    <div className="notebook-actions">
      {colabUrl && (
        <a
          href={colabUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="notebook-actions__button notebook-actions__button--primary"
        >
          <span aria-hidden="true">&#9654;</span>
          <span>Run in Google Colab</span>
        </a>
      )}
      {downloadUrl && (
        <a
          href={downloadUrl}
          className="notebook-actions__button notebook-actions__button--secondary"
          download={resolvedFilename}
          onClick={handleDownload}
        >
          <span>Download notebook</span>
        </a>
      )}
    </div>
  );
}

export default NotebookActions;
