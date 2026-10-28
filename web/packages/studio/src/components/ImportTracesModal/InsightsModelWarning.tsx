// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner } from '@nvidia/foundations-react-core';

export const InsightsModelWarning = () => (
  <Banner kind="inline" status="warning">
    Nemotron 3.5 Lightning 30B has known tool-calling failures in Compass. Try Nemotron 3.5 Super VL
    instead.
  </Banner>
);
