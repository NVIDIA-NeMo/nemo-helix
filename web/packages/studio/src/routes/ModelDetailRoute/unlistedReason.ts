// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';

const names = (refs: string[]) => refs.map((ref) => getPartsFromReference(ref).name).join(', ');

export const getUnlistedReason = ({
  missingProviders,
  unlistingProviders,
}: {
  missingProviders: string[];
  unlistingProviders: string[];
}): string => {
  const reasons = [
    unlistingProviders.length > 0 && `${names(unlistingProviders)} no longer lists this model`,
    missingProviders.length > 0 &&
      `${names(missingProviders)} no longer ${missingProviders.length > 1 ? 'exist' : 'exists'}`,
  ].filter(Boolean);
  return reasons.join('; ');
};
