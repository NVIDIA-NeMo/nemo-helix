// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { logger } from '@nemo/common/src/utils/logger';
import { NavigationDrawer } from '@studio/components/Layouts/NavigationDrawer';
import type {
  NavInputItem,
  NavItem as NavItemData,
} from '@studio/components/Layouts/NavigationDrawer/types';
import { isGroup } from '@studio/components/Layouts/NavigationDrawer/utils';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  usePluginInstalled,
  usePlugins,
  usePluginsError,
  usePluginsLoaded,
} from '@studio/plugins/PluginContext';
import { iconColorClass } from '@studio/routes/constants';
import {
  getAgentSideNavItems,
  getAnonymizerSideNavItems,
  getBaseModelsSideNavItems,
  getCustomizationSideNavItems,
  getDashboardSideNavItems,
  getDataDesignerSideNavItems,
  getDeploymentSideNavItems,
  getEvaluationSideNavItems,
  getExperimentSideNavItems,
  getFilesetSideNavItems,
  getGuardrailsSideNavItems,
  getInferenceProviderSideNavItems,
  getIntakeSideNavItems,
  getJobSideNavItems,
  getModelCompareSideNavItems,
  getOptimizerSideNavItems,
  getSafeSynthesizerSideNavItems,
  getSettingsSideNavItems,
  getTasksetSideNavItems,
  getVirtualModelsSideNavItems,
} from '@studio/routes/groups';
import { getAgentsListRoute, getWorkspaceDatasetsRoute } from '@studio/routes/utils';
import { lazy, Suspense, useMemo } from 'react';

const NoPluginNavIcon = () => null;

// Without the catch, a failed chunk (e.g. after a redeploy) reaches the root errorElement.
const PluginNavIcon = lazy(() =>
  import('@studio/plugins/PluginNavIcon')
    .then((m) => ({ default: m.PluginNavIcon }))
    .catch((err: unknown) => {
      logger.warn('[plugins] Failed to load plugin nav icons:', err);
      return { default: NoPluginNavIcon };
    })
);

// These rows stand for the entity their landing page lists, so they take
// that entity's canonical glyph.
const AgentsIcon = ENTITY_ICONS.agents;
const DatasetsIcon = ENTITY_ICONS.datasets;

export const WorkspaceSideNav = ({ collapsed }: { collapsed?: boolean }) => {
  const workspace = useWorkspaceFromPath();
  const plugins = usePlugins();
  const agentsInstalled = usePluginInstalled('agents');
  const pluginsLoaded = usePluginsLoaded();
  const pluginsError = usePluginsError();
  const manifestResolved = pluginsLoaded && !pluginsError;
  const showAgents = agentsInstalled || !manifestResolved;

  const items = useMemo<NavInputItem[]>(() => {
    const agentsRow = showAgents
      ? [
          {
            id: 'agents',
            slotIcon: <AgentsIcon className={iconColorClass} />,
            slotLabel: 'Agents',
            href: getAgentsListRoute(workspace),
          },
        ]
      : [];

    const sections: { group?: string; items: NavItemData[] }[] = [
      {
        items: [...getDashboardSideNavItems(workspace), ...getModelCompareSideNavItems(workspace)],
      },
      {
        group: 'Agent Development',
        items: [
          ...agentsRow,
          ...(showAgents ? getAgentSideNavItems(workspace) : []),
          ...getExperimentSideNavItems(workspace),
          ...getTasksetSideNavItems(workspace),
          ...getOptimizerSideNavItems(workspace),
          ...getIntakeSideNavItems(workspace),
        ],
      },
      {
        group: 'Models',
        items: [
          ...getBaseModelsSideNavItems(workspace),
          ...getEvaluationSideNavItems(workspace),
          ...getCustomizationSideNavItems(workspace),
          ...getDeploymentSideNavItems(workspace),
          ...getVirtualModelsSideNavItems(workspace),
        ],
      },
      {
        group: 'Resources',
        items: [
          {
            id: 'dataset-list',
            slotIcon: <DatasetsIcon className={iconColorClass} />,
            slotLabel: 'Datasets',
            href: getWorkspaceDatasetsRoute(workspace),
          },
        ],
      },
      { group: 'Safety', items: getGuardrailsSideNavItems(workspace) },
      {
        group: 'Data Preparation',
        items: [
          ...getDataDesignerSideNavItems(workspace),
          ...getSafeSynthesizerSideNavItems(workspace),
          ...getAnonymizerSideNavItems(workspace),
        ],
      },
      {
        group: 'Workspace',
        items: [
          ...getInferenceProviderSideNavItems(workspace),
          ...getFilesetSideNavItems(workspace),
          ...getJobSideNavItems(workspace),
          ...getSettingsSideNavItems(workspace),
        ],
      },
    ];

    // Empty sections are omitted; the heading-less first section is a bare list.
    return sections
      .filter((s) => s.items.length > 0)
      .map((s) =>
        s.group === undefined ? { items: s.items } : { group: s.group, items: s.items }
      );
  }, [workspace, showAgents]);

  const pluginNavGroups = useMemo(
    () =>
      plugins.flatMap((plugin) => {
        try {
          return plugin.navItems(workspace).map((group) => ({
            group: group.group,
            items: group.items.map((item) => ({
              // Namespaced: ids are React keys and accordion-state keys, and
              // merging puts plugin items in the same array as core ones.
              id: `${plugin.name}:${item.id}`,
              slotIcon: (
                <Suspense>
                  <PluginNavIcon iconName={item.iconName} className={iconColorClass} />
                </Suspense>
              ),
              slotLabel: item.label,
              href: item.href,
            })),
          }));
        } catch (err) {
          logger.warn(`[plugins] navItems() threw for plugin "${plugin.name}":`, err);
          return [];
        }
      }),
    [plugins, workspace]
  );

  // A plugin naming an existing group joins it rather than rendering a second
  // header with the same label. Unmatched groups append in plugin order.
  const allItems = useMemo<NavInputItem[]>(() => {
    const pending = new Map<string, NavItemData[]>();
    for (const { group, items: groupItems } of pluginNavGroups) {
      pending.set(group, [...(pending.get(group) ?? []), ...groupItems]);
    }

    const merged: NavInputItem[] = items.map((entry) => {
      if (!isGroup(entry) || entry.group === undefined) return entry;
      const extra = pending.get(entry.group);
      if (!extra) return entry;
      pending.delete(entry.group);
      return { ...entry, items: [...entry.items, ...extra] };
    });

    for (const [group, groupItems] of pending) {
      merged.push({ group, items: groupItems });
    }
    return merged;
  }, [items, pluginNavGroups]);

  return <NavigationDrawer items={allItems} collapsed={collapsed} />;
};
