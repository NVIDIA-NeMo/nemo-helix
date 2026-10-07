// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { flagDefinitions } from '@studio/constants/featureFlags/featureFlags';
import { ROUTE_PARAMS, ROUTES } from '@studio/constants/routes';
import type { RouteObject } from 'react-router';

/**
 * Placeholder workspace segment matching React Router param patterns in ROUTES.
 *
 * Route helpers build paths with `generatePath`, which percent-encodes param values
 * (react-router v7), so feeding it this placeholder yields `%3Aworkspace`. Call sites
 * decode the result to recover the raw `:workspace` pattern for comparison.
 */
const WORKSPACE_ROUTE_PLACEHOLDER = `:${ROUTE_PARAMS.workspace}`;

const isWorkspaceScopedPath = (path: string) =>
  path === ROUTES.workspace.index || path.startsWith(`${ROUTES.workspace.index}/`);

/** React Router child segments are relative unless they start with `/`. */
const joinRoutePath = (parentPath: string | undefined, segment: string): string => {
  if (segment.startsWith('/')) {
    return segment;
  }
  const base = parentPath?.replace(/\/$/, '') ?? '';
  return base ? `${base}/${segment}` : `/${segment}`;
};

const collectAllPaths = (routes: RouteObject[], parentPath?: string): string[] => {
  return routes.flatMap((route) => {
    const resolvedPath =
      route.path !== undefined ? joinRoutePath(parentPath, route.path) : parentPath;

    const paths = resolvedPath !== undefined && route.path !== undefined ? [resolvedPath] : [];

    return [...paths, ...(route.children ? collectAllPaths(route.children, resolvedPath) : [])];
  });
};

const findIfRouteExists = (routes: RouteObject[], path: string, parentPath?: string): boolean => {
  return routes.some((route) => {
    const resolvedPath =
      route.path !== undefined ? joinRoutePath(parentPath, route.path) : parentPath;

    if (resolvedPath === path) {
      return true;
    }
    if (route.children) {
      return findIfRouteExists(route.children, path, resolvedPath);
    }
    return false;
  });
};

/** Stubs every route-gating feature flag off (see flagDefinitions). */
const stubAllFeatureFlagsOff = () => {
  for (const { envVar } of Object.values(flagDefinitions)) {
    vi.stubEnv(envVar, 'false');
  }
};

const customizationRoutes = [
  ROUTES.workspace.customizationJobList,
  ROUTES.workspace.customizationJobDetails,
];

const evalRoutes = [
  ROUTES.workspace.evaluation,
  ROUTES.workspace.evaluationMetrics,
  ROUTES.workspace.evaluationMetricNew,
  ROUTES.workspace.evaluationMetricDetails,
  ROUTES.workspace.evaluationMetricRun,
  ROUTES.workspace.evaluationBenchmarks,
  ROUTES.workspace.evaluationBenchmarkDetails,
  ROUTES.workspace.evaluationResults,
  ROUTES.workspace.evaluationResultDetails,
];

const intakeRoutes = [
  ROUTES.workspace.intake,
  ROUTES.workspace.intakeTraces,
  ROUTES.workspace.intakeSpans,
  ROUTES.workspace.intakeSession,
];

const optimizerRoutes = [ROUTES.workspace.optimizer, ROUTES.workspace.optimizerInsight];

describe('Routes', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  // This test is the exhaustive safety net: with every feature flag disabled,
  // only the workspace default route should remain. If a new route is added to
  // index.tsx without a gate function, this test will fail because that path
  // will appear in remainingWorkspacePaths.
  //
  // New flags in flagDefinitions are stubbed automatically below.
  describe('with all feature flags disabled', () => {
    let routes: RouteObject[];
    let getWorkspaceDetailsDefaultRoute: (workspace: string) => string;

    beforeAll(async () => {
      vi.resetModules();
      stubAllFeatureFlagsOff();
      [{ routes }, { getWorkspaceDetailsDefaultRoute }] = await Promise.all([
        import('./index'),
        import('./utils'),
      ]);
    }, 30_000);

    it('every workspace route except the default route is behind a feature flag', () => {
      const defaultRoute = decodeURIComponent(
        getWorkspaceDetailsDefaultRoute(WORKSPACE_ROUTE_PLACEHOLDER)
      );
      // Intentionally always-on (ungated) workspace routes: the workspace index, plus
      // the features that have graduated out of flag gating. A route only belongs here
      // once its feature ships unconditionally — a new feature should arrive flag-gated.
      const alwaysOnWorkspaceRoutes = new Set<string>([
        ROUTES.workspace.virtualModels,
        ROUTES.workspace.index,
        ROUTES.workspace.agentDetail,
        ROUTES.workspace.agentEvaluationDetail,
        ROUTES.workspace.baseModels,
        ROUTES.workspace.baseModelsModel,
        ROUTES.workspace.dataDesignerJobList,
        ROUTES.workspace.dataDesignerJobDetails,
        ROUTES.workspace.dataDesignerJobNew,
        ROUTES.workspace.dataDesignerJobBuild,
        ROUTES.workspace.dataDesignerJobNewLegacy,
        ROUTES.workspace.deployments,
        ROUTES.workspace.deploymentsNew,
        ROUTES.workspace.deploymentsDeployment,
        ROUTES.workspace.filesets,
        ROUTES.workspace.filesetNew,
        ROUTES.workspace.filesetDetails,
        ROUTES.workspace.filesetFile,
        ROUTES.workspace.jobs,
        ROUTES.workspace.jobDetail,
        `${ROUTES.workspace.plugin}/*`,
        ROUTES.workspace.safeSynthesizer,
        ROUTES.workspace.safeSynthesizerNew,
        ROUTES.workspace.safeSynthesizerJob,
        ROUTES.workspace.safeSynthesizerJobReport,
        ROUTES.workspace.secrets,
        ROUTES.workspace.settings,
      ]);
      const remainingWorkspacePaths = [
        ...new Set(
          collectAllPaths(routes)
            .filter(isWorkspaceScopedPath)
            .filter((path) => !alwaysOnWorkspaceRoutes.has(path))
        ),
      ].sort();

      expect(remainingWorkspacePaths).toEqual([defaultRoute]);
    });
  });

  describe('individual feature flag gating', () => {
    beforeEach(() => {
      vi.resetModules();
    });

    it('should exclude customization routes if customizer is disabled', async () => {
      vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'false');
      const { routes } = await import('./index');
      const { getWorkspaceDetailsDefaultRoute } = await import('./utils');
      customizationRoutes.forEach((route) => {
        expect(findIfRouteExists(routes, route)).toBe(false);
      });
      expect(
        findIfRouteExists(
          routes,
          decodeURIComponent(getWorkspaceDetailsDefaultRoute(WORKSPACE_ROUTE_PLACEHOLDER))
        )
      ).toBe(true);
    });

    it('should exclude evaluation routes if evaluator is disabled', async () => {
      vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'true');
      vi.stubEnv('VITE_FF_EVALUATOR_ENABLED', 'false');
      vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'true');
      const { routes } = await import('./index');
      const { getWorkspaceDetailsDefaultRoute } = await import('./utils');
      evalRoutes.forEach((route) => {
        expect(findIfRouteExists(routes, route)).toBe(false);
      });
      expect(
        findIfRouteExists(
          routes,
          decodeURIComponent(getWorkspaceDetailsDefaultRoute(WORKSPACE_ROUTE_PLACEHOLDER))
        )
      ).toBe(true);
    });

    it('should exclude intake routes if intake is disabled', async () => {
      vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'true');
      vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'false');
      const { routes } = await import('./index');
      const { getWorkspaceDetailsDefaultRoute } = await import('./utils');
      intakeRoutes.forEach((route) => {
        expect(findIfRouteExists(routes, route)).toBe(false);
      });
      expect(
        findIfRouteExists(
          routes,
          decodeURIComponent(getWorkspaceDetailsDefaultRoute(WORKSPACE_ROUTE_PLACEHOLDER))
        )
      ).toBe(true);
    });

    it('should include the dashboard route if assistant studio is enabled', async () => {
      vi.stubEnv('VITE_FF_ASSISTANT_STUDIO_ENABLED', 'true');
      vi.stubEnv('VITE_FF_DASHBOARD_SANDBOX_ENABLED', 'false');
      const { routes } = await import('./index');
      expect(findIfRouteExists(routes, ROUTES.workspace.dashboard)).toBe(true);
    });

    it('should exclude the dashboard route if neither dashboard variant is enabled', async () => {
      vi.stubEnv('VITE_FF_ASSISTANT_STUDIO_ENABLED', 'false');
      vi.stubEnv('VITE_FF_DASHBOARD_SANDBOX_ENABLED', 'false');
      const { routes } = await import('./index');
      expect(findIfRouteExists(routes, ROUTES.workspace.dashboard)).toBe(false);
    });

    it('includes Optimizer routes when Optimizer is enabled for preview', async () => {
      vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'preview');
      const { routes } = await import('./index');

      const missingRoutes = optimizerRoutes.filter((route) => !findIfRouteExists(routes, route));

      expect(missingRoutes).toHaveLength(0);
    });
  });
});
