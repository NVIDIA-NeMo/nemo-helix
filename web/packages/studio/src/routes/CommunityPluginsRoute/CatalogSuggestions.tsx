// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useCatalogApisPluginsCatalogGet,
  useInstalledApisPluginsInstalledGet,
} from '@nemo/sdk/generated/platform/external-plugins';
import { Button, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

/**
 * The operator's `catalog.json` suggestions that are not registered yet. A suggestion
 * only prefills the add dialog; it still goes through preview and consent.
 */
export const CatalogSuggestions: FC<{ onAdd: (url: string) => void }> = ({ onAdd }) => {
  const { data: catalog } = useCatalogApisPluginsCatalogGet();
  const { data: installed } = useInstalledApisPluginsInstalledGet();

  const registered = new Set((installed?.plugins ?? []).map((plugin) => plugin.name));
  const suggestions = (catalog?.entries ?? []).filter((entry) => !registered.has(entry.name));
  if (suggestions.length === 0) return null;

  return (
    <Panel elevation="high">
      <Stack gap="density-lg">
        <Text kind="body/bold/lg">Suggested plugins</Text>
        {suggestions.map((entry) => (
          <Flex key={entry.url} justify="between" align="center" gap="density-2xl">
            <Stack gap="density-xs">
              <Text kind="label/bold/md">{entry.title}</Text>
              <Text kind="body/regular/xs" className="text-subtle">
                {entry.url}
              </Text>
            </Stack>
            <Button color="neutral" kind="secondary" onClick={() => onAdd(entry.url)}>
              Review
            </Button>
          </Flex>
        ))}
      </Stack>
    </Panel>
  );
};
