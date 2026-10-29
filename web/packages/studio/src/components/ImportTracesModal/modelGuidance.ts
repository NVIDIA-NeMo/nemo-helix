// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const isLightningModel = (model: string): boolean =>
  /(?:^|\/)(?:nvidia\/|(?:nvidia[-_]){1,2})nemotron[-_]3[._-]5[-_]lightning[-_]30b/i.test(model);
