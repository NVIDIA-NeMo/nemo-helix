// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as customQueries from '@nemo/common/src/tests/customQueries';
import { Loading } from '@studio/components/Layouts/Loading';
import { routes as appRoutes } from '@studio/routes';
import { TestProviders, TestProvidersOptions } from '@studio/tests/util/TestProviders';
import { queries, render, within } from '@testing-library/react';
import { Suspense } from 'react';
import { createMemoryRouter, MemoryRouter, type RouteObject } from 'react-router';
// Only the `createMemoryRouter` branch below honors `flushSync`; the declarative
// `MemoryRouter` fallback ignores it, so tests that depend on it need `routes`.
import { RouterProvider } from 'react-router/dom';

/**
 * Options for the renderRoute function
 */
export interface RenderRouteOptions {
  /**
   * Mock browsing history - initializes the router at these URL(s).
   * If array, last item will be used as the current path.
   */
  history?: string | string[];

  /**
   * Custom routes to use instead of rendering a single element.
   * Use this when you need to test navigation between routes.
   */
  routes?: RouteObject[];

  /**
   * Options to pass to TestProviders (e.g., queryClientConfig)
   */
  testProviderOptions?: TestProvidersOptions;
}

/**
 * @deprecated Use renderRoute instead
 * Options for passing to TestProviders component
 */
interface MemoryRouterOptions {
  // mock browsing history
  // if array, last item will be used as the current path
  history: string | string[];

  // optionally override the element at this path in the app router tree
  overrideRoutes?: OverrideRoute[];

  // additional routes to be added to the app router
  otherRoutes?: RouteObject[];
}

interface OverrideRoute {
  path: string;
  element: React.ReactElement;
}

const allQueries = {
  ...queries,
  ...customQueries,
};

export const customRender = (
  element: React.ReactElement,
  testProviderOptions?: TestProvidersOptions
) => {
  return render(
    <TestProviders options={testProviderOptions}>
      <Suspense fallback={<Loading />}>
        <MemoryRouter>{element}</MemoryRouter>
      </Suspense>
    </TestProviders>,
    { queries: allQueries, ...testProviderOptions }
  );
};

/**
 * Overwrites the element of a specific route in the provided routes array
 * Uses recursive search through children array
 * @param routes
 * @param path
 * @param newElement
 */
function overwriteRouteElement(
  routes: RouteObject[],
  path: string,
  newElement: React.ReactElement
): RouteObject[] {
  return routes.map((route) => {
    if (route.path === path) {
      const newRoute = { ...route, element: newElement };
      return newRoute;
    }
    if (route.children) {
      return { ...route, children: overwriteRouteElement(route.children, path, newElement) };
    }
    return route;
  });
}

/**
 * Renders a component or set of routes wrapped in TestProviders and MemoryRouter.
 * Preferred method for testing route components.
 *
 * @example
 * // Simple render without routing
 * renderRoute(<MyComponent />);
 *
 * @example
 * // Render with initial URL (e.g., to test URL params)
 * renderRoute(<MyComponent />, { history: '/projects/default/my-project/page' });
 *
 * @example
 * // Render with custom routes for navigation testing
 * renderRoute(undefined, {
 *   history: '/projects/default/my-project',
 *   routes: [
 *     { path: '/projects/:namespace/:name', element: <ProjectPage /> },
 *     { path: '/projects/:namespace/:name/details', element: <DetailsPage /> },
 *   ],
 * });
 */
export const renderRoute = (element?: React.ReactElement, options?: RenderRouteOptions) => {
  const { history, routes, testProviderOptions } = options ?? {};

  // Convert history to array for initialEntries
  const initialEntries = history ? (typeof history === 'string' ? [history] : history) : undefined;

  // If custom routes provided, use createMemoryRouter
  if (routes) {
    const router = createMemoryRouter(routes, { initialEntries });
    return render(
      <TestProviders options={testProviderOptions}>
        <Suspense fallback={<Loading />}>
          <RouterProvider router={router} />
        </Suspense>
      </TestProviders>,
      { queries: allQueries }
    );
  }

  // Otherwise, use simple MemoryRouter with the element
  return render(
    <TestProviders options={testProviderOptions}>
      <Suspense fallback={<Loading />}>
        <MemoryRouter initialEntries={initialEntries}>{element}</MemoryRouter>
      </Suspense>
    </TestProviders>,
    { queries: allQueries }
  );
};

/**
 * Returns given `element` wrapped in global context providers and MemoryRouter compnent to enable unit testing
 * components that use these dependencies.
 * @deprecated Use createMemoryRouter and RouterProvider instead. This function can lead to flaky tests.
 * @param routerOptions optional options to pass to MemoryRouter component
 * @param testProviderOptions optional options to pass to global context providers
 * @returns RenderResult for the given `element`
 */
export const renderWithRouter = (
  routerOptions: MemoryRouterOptions,
  testProviderOptions?: TestProvidersOptions
) => {
  const { history, otherRoutes, overrideRoutes } = routerOptions;

  // handle initialEntries to be passed to memoryRouter
  // convert initialEntries to an array if it's a single string
  const entries: string[] = typeof history === 'string' ? [history] : history;

  // by default, routes is the same as the app's router tree
  let routes: RouteObject[] = [...appRoutes];

  if (otherRoutes) {
    // if there are extra routes to add, prepend to appRoutes
    routes = [...otherRoutes, ...routes];
  }

  if (overrideRoutes) {
    for (const { path, element } of overrideRoutes) {
      // Recursively update the route tree to replace the element at the given path
      routes = overwriteRouteElement(routes, path, element);
    }
  }

  const router = createMemoryRouter(routes, {
    initialEntries: entries,
  });

  return render(
    <TestProviders options={testProviderOptions}>
      <Suspense fallback={<Loading />}>
        <RouterProvider router={router} />
      </Suspense>
    </TestProviders>,
    { queries: allQueries, ...testProviderOptions }
  );
};

const customScreen = within(document.body, allQueries);
const customWithin = (element: HTMLElement) => within(element, allQueries);

// Re-export everything to enable overriding @testing-library's `render` method
// eslint-disable-next-line react-refresh/only-export-components
export * from '@testing-library/react';
export { customRender as render, customScreen as screen, customWithin as within };
