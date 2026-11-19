// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const mockFile = new File(['file-content'], 'test-file.txt', {
  type: 'text/plain',
});

export const mockFile2 = new File(['another-content'], 'test-file2.txt', {
  type: 'text/plain',
});

export const mockFile3 = new File(['image-content'], 'test-image.jpg', {
  type: 'image/jpeg',
});
