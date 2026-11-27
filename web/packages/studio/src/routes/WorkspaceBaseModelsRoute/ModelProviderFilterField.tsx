// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_NAMESPACE } from '@nemo/common/src/constants';
import { useModelsListProviders } from '@nemo/sdk/generated/platform/model-providers';
import { Select } from '@nvidia/foundations-react-core';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { type FC, useMemo } from 'react';

export interface ModelProviderFilterFieldProps {
  workspace: string;
  value: string | undefined;
  onValueChange: (value: string | undefined) => void;
}

/** Select of `{workspace}/{provider}` refs from the current and default workspaces. */
export const ModelProviderFilterField: FC<ModelProviderFilterFieldProps> = ({
  workspace,
  value,
  onValueChange,
}) => {
  const params = { page_size: DEFAULT_LARGE_PAGE_SIZE };
  const { data: workspaceProviders } = useModelsListProviders(workspace, params);
  const { data: defaultProviders } = useModelsListProviders(DEFAULT_NAMESPACE, params, {
    query: { enabled: workspace !== DEFAULT_NAMESPACE },
  });

  const items = useMemo(() => {
    const refs = [...(workspaceProviders?.data ?? []), ...(defaultProviders?.data ?? [])].map(
      (p) => `${p.workspace}/${p.name}`
    );
    return [...new Set(refs)].sort();
  }, [workspaceProviders?.data, defaultProviders?.data]);

  return (
    <Select
      aria-label="Model provider"
      data-testid="model-catalog-model-providers-filter"
      dismissible
      items={items}
      value={value ?? ''}
      placeholder="Select a model provider"
      onValueChange={(v: string) => onValueChange(v || undefined)}
    />
  );
};
