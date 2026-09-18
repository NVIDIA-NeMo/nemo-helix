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
import { BASE_MODELS_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getPluginIcon } from '@studio/plugins/iconMap';
import {
  usePluginInstalled,
  usePlugins,
  usePluginsError,
  usePluginsLoaded,
} from '@studio/plugins/PluginContext';
import { iconColorClass } from '@studio/routes/constants';
import {
  getAnonymizerSideNavItems,
  getCustomizationSideNavItems,
  getDashboardSideNavItems,
  getDataDesignerSideNavItems,
  getDeploymentSideNavItems,
  getEvaluationSideNavItems,
  getExperimentSideNavItems,
  getFilesetSideNavItems,
  getGuardrailsSideNavItems,
  getIntakeSideNavItems,
  getJobSideNavItems,
  getModelCompareSideNavItems,
  getOptimizerSideNavItems,
  getSafeSynthesizerSideNavItems,
  getSettingsSideNavItems,
  getVirtualModelsSideNavItems,
} from '@studio/routes/groups';
import { getAgentsListRoute, getWorkspaceBaseModelsRoute } from '@studio/routes/utils';
import { useMemo } from 'react';
import { useLocation } from 'react-router';

// The parent rows stand for the entity their landing page lists, so they take
// that entity's canonical glyph. `Datasets` only groups the data jobs today,
// but it is becoming an entity in its own right, so it is treated as one here.
const AgentsIcon = ENTITY_ICONS.agents;
const ModelsIcon = ENTITY_ICONS.baseModels;
const DatasetsIcon = ENTITY_ICONS.datasets;

/** Whether the path is on, or nested under, one of these items. */
const isUnder = (pathname: string, items: { href?: string }[]) =>
  items.some(
    (item) =>
      item.href !== undefined && (pathname === item.href || pathname.startsWith(`${item.href}/`))
  );

const eachItem = (entries: NavInputItem[]): NavItemData[] =>
  entries.flatMap((entry) => (isGroup(entry) ? entry.items : [entry]));

/** A parent starts expanded when the current page is its own landing page or one of its children. */
const ownsCurrentPage = (item: NavItemData, pathname: string) =>
  isUnder(pathname, [{ href: item.href }, ...(item.subItems ?? [])]);

/**
 * The ids of every expanded parent, joined into one string. Navigating deeper inside a section
 * leaves this untouched, which is what lets `withDefaultOpen` hand back the identical tree.
 */
const openParentKey = (entries: NavInputItem[], pathname: string): string =>
  eachItem(entries)
    .filter((item) => item.subItems !== undefined && ownsCurrentPage(item, pathname))
    .map((item) => item.id)
    .join('|');

/**
 * Stamp `defaultOpen` onto the parents named by `openIds`. Kept apart from building the tree so a
 * navigation rewrites a few parent objects instead of rebuilding every leaf and icon.
 */
const withDefaultOpen = (entries: NavInputItem[], openIds: string): NavInputItem[] => {
  const open = new Set(openIds.split('|'));
  const stamp = (item: NavItemData): NavItemData =>
    item.subItems === undefined ? item : { ...item, defaultOpen: open.has(item.id) };
  return entries.map((entry) =>
    isGroup(entry) ? { ...entry, items: entry.items.map(stamp) } : stamp(entry)
  );
};

export const WorkspaceSideNav = ({ collapsed }: { collapsed?: boolean }) => {
  const workspace = useWorkspaceFromPath();
  const { pathname } = useLocation();
  const plugins = usePlugins();
  const agentsInstalled = usePluginInstalled('agents');
  const pluginsLoaded = usePluginsLoaded();
  const pluginsError = usePluginsError();
  const manifestResolved = pluginsLoaded && !pluginsError;
  const showAgents = agentsInstalled || !manifestResolved;

  const baseItems = useMemo<NavInputItem[]>(() => {
    const dashboardNav = getDashboardSideNavItems(workspace);
    const customizerNav = getCustomizationSideNavItems(workspace);
    const evalNav = getEvaluationSideNavItems(workspace);
    const tracesNav = getIntakeSideNavItems(workspace);
    const experimentNav = getExperimentSideNavItems(workspace);
    const safeSynthesizerNav = getSafeSynthesizerSideNavItems(workspace);
    const dataDesignerNav = getDataDesignerSideNavItems(workspace);
    const anonymizerNav = getAnonymizerSideNavItems(workspace);
    const optimizerNav = getOptimizerSideNavItems(workspace);
    const virtualModelsNav = getVirtualModelsSideNavItems(workspace);
    const modelCompareNav = getModelCompareSideNavItems(workspace);
    const deploymentsNav = getDeploymentSideNavItems(workspace);

    const observabilityItems = [...optimizerNav, ...tracesNav];

    const modelSubItems = [
      ...deploymentsNav,
      ...customizerNav,
      ...evalNav,
      ...modelCompareNav,
      ...virtualModelsNav,
    ];
    const datasetSubItems = [...anonymizerNav, ...dataDesignerNav, ...safeSynthesizerNav];

    // Agents and Models link to their own entity list page; Models' chevron expands its sub-items (Agents has none).
    const agentsHref = showAgents ? getAgentsListRoute(workspace) : undefined;
    const modelsHref = BASE_MODELS_ENABLED ? getWorkspaceBaseModelsRoute(workspace) : undefined;

    const componentItems = [
      ...(agentsHref !== undefined
        ? [
            {
              id: 'agents-group',
              slotIcon: <AgentsIcon className={iconColorClass} />,
              slotLabel: 'Agents',
              href: agentsHref,
            },
          ]
        : []),
      ...(modelsHref !== undefined || modelSubItems.length > 0
        ? [
            {
              id: 'models-group',
              slotIcon: <ModelsIcon className={iconColorClass} />,
              slotLabel: 'Models',
              href: modelsHref,
              subItems: modelSubItems,
            },
          ]
        : []),
    ];

    const dataItems =
      datasetSubItems.length > 0
        ? [
            {
              id: 'datasets-group',
              slotIcon: <DatasetsIcon className={iconColorClass} />,
              slotLabel: 'Datasets',
              subItems: datasetSubItems,
            },
          ]
        : [];

    const governanceItems = getGuardrailsSideNavItems(workspace);

    return [
      ...dashboardNav,
      ...(observabilityItems.length > 0
        ? [{ group: 'Observability', items: observabilityItems }]
        : []),
      ...(componentItems.length > 0 ? [{ group: 'Components', items: componentItems }] : []),
      ...(experimentNav.length > 0 ? [{ group: 'Evaluations', items: experimentNav }] : []),
      ...(dataItems.length > 0 ? [{ group: 'Data', items: dataItems }] : []),
      ...(governanceItems.length > 0 ? [{ group: 'Governance', items: governanceItems }] : []),
    ];
  }, [workspace, showAgents]);

  const openIds = useMemo(() => openParentKey(baseItems, pathname), [baseItems, pathname]);
  const items = useMemo(() => withDefaultOpen(baseItems, openIds), [baseItems, openIds]);

  const systemNavGroup = useMemo(() => {
    const systemItems = [
      ...getFilesetSideNavItems(workspace),
      ...getJobSideNavItems(workspace),
      ...getSettingsSideNavItems(workspace),
    ];
    return systemItems.length > 0 ? [{ group: 'System', items: systemItems }] : [];
  }, [workspace]);

  const pluginNavGroups = useMemo(
    () =>
      plugins.flatMap((plugin) => {
        try {
          return plugin.navItems(workspace).map((group) => ({
            group: group.group,
            items: group.items.map((item) => {
              const Icon = getPluginIcon(item.iconName);
              return {
                // Namespaced: ids are React keys and accordion-state keys, and
                // merging puts plugin items in the same array as core ones.
                id: `${plugin.name}:${item.id}`,
                slotIcon: Icon ? <Icon className={iconColorClass} /> : undefined,
                slotLabel: item.label,
                href: item.href,
              };
            }),
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
    return [...merged, ...systemNavGroup];
  }, [items, pluginNavGroups, systemNavGroup]);

  return <NavigationDrawer items={allItems} collapsed={collapsed} />;
};
