// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Shared vitest setup for React/DOM test-bearing packages.
 * Include this in your vitest setupFiles: ['@nemo/testing/react/setup']
 *
 * https://vitest.dev/config/#setupfiles
 */
import * as matchers from '@testing-library/jest-dom/matchers';
import { cleanup } from '@testing-library/react';

expect.extend(matchers);

/*
 * Browser API mocks for KUI and other component libraries.
 * Vitest 4: constructor mocks must use function/class, not arrow functions.
 */

/**
 * This addresses a bug in testing components (ex. KUI or MUI icons) that use libraries that rely on
 * IntersectionObserver (ex. `external-svg-loader`). Without this, we'll see an error like "TypeError:
 * Cannot read properties of undefined (reading 'observe')".
 */
const IntersectionObserverMock = vi.fn(function IntersectionObserverMock() {
  return {
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  };
});
vi.stubGlobal('IntersectionObserver', IntersectionObserverMock);

/**
 * This is a bug in testing components that use KUI. Without this, we'll see
 * an error like "window.ResizeObserver is not a function".
 */
const ResizeObserverMock = vi.fn(function ResizeObserverMock() {
  return {
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  };
});
vi.stubGlobal('ResizeObserver', ResizeObserverMock);

/**
 * Mock MutationObserver for Ariakit components.
 */
const MutationObserverMock = vi.fn(function MutationObserverMock() {
  return {
    observe: vi.fn(),
    disconnect: vi.fn(),
    takeRecords: vi.fn(() => []),
  };
});
vi.stubGlobal('MutationObserver', MutationObserverMock);

/**
 * Mock additional observer APIs that Ariakit might use.
 */
const PerformanceObserverMock = vi.fn(function PerformanceObserverMock() {
  return {
    observe: vi.fn(),
    disconnect: vi.fn(),
    takeRecords: vi.fn(),
  };
});
vi.stubGlobal('PerformanceObserver', PerformanceObserverMock);

/**
 * Mock localStorage to ensure consistent behavior across all tests.
 * jsdom provides localStorage but some Node.js configurations can interfere with it.
 */
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => Object.keys(store)[index] ?? null,
  };
})();
vi.stubGlobal('localStorage', localStorageMock);

/**
 * user-event's `upload` redefines `input.files` as getter-only, and KUI 1.19's Upload writes
 * `input.files` back during the change event, which throws. Give that getter a setter.
 */
const writableUploadFiles = Symbol.for('nemo.testing.writableUploadFiles');
if (!(writableUploadFiles in Object.defineProperties)) {
  const defineProperties = Object.defineProperties;
  const patched = <T>(target: T, properties: PropertyDescriptorMap & ThisType<unknown>): T => {
    const files = properties.files;
    if (target instanceof HTMLInputElement && files?.get && !files.set) {
      let current: unknown = files.get();
      properties = {
        ...properties,
        files: {
          configurable: true,
          get: () => current,
          set: (next: unknown) => {
            current = next;
          },
        },
      };
    }
    return defineProperties(target, properties);
  };
  Object.defineProperties = Object.assign(patched, { [writableUploadFiles]: true });
}

beforeEach(() => {
  // Unmount any React trees from a previous test. Vitest 4 does not fire
  // `afterEach` hooks registered in setup files reliably (observed: setupFiles'
  // `beforeEach` fires per test, `afterEach` never does), so run cleanup at the
  // start of each test instead.
  cleanup();
  /**
   * This is a bug in testing components that use KUI. Without this, we'll see
   * an error like "window.matchMedia is not a function".
   */
  global.matchMedia = vi.fn().mockImplementation((query) => {
    return {
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    };
  });

  /**
   * Mock HTMLCanvasElement.prototype.getContext to avoid jsdom error:
   * "Not implemented: HTMLCanvasElement.prototype.getContext (without installing the canvas npm package)"
   */
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
    fillRect: vi.fn(),
    clearRect: vi.fn(),
    getImageData: vi.fn(),
    putImageData: vi.fn(),
    createImageData: vi.fn(),
    setTransform: vi.fn(),
    drawImage: vi.fn(),
    save: vi.fn(),
    fillText: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    measureText: vi.fn(() => ({ width: 0 })),
    transform: vi.fn(),
    rect: vi.fn(),
    clip: vi.fn(),
  });

  // Set up pointer capture mocks
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
  window.HTMLElement.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
  window.HTMLElement.prototype.releasePointerCapture = vi.fn();
  window.HTMLElement.prototype.setPointerCapture = vi.fn();

  /**
   * Popover API polyfill — happy-dom does not implement showPopover/hidePopover/togglePopover.
   * KUI v1.0 uses these to drive Select/Tooltip/Popover open state.
   * We toggle a data attribute so testing-library can see the popover as visible.
   */
  function syncPopoverState(this: HTMLElement, open: boolean) {
    const oldState = this.hasAttribute('popover-open') ? 'open' : 'closed';
    const newState = open ? 'open' : 'closed';
    if (oldState === newState) {
      return;
    }
    if (open) {
      this.setAttribute('popover-open', '');
      this.removeAttribute('hidden');
    } else {
      this.removeAttribute('popover-open');
      // Hide so testing-library's accessible queries treat it as absent.
      this.setAttribute('hidden', 'until-found');
    }
    const toggleEvent = new Event('toggle', { bubbles: false }) as Event & {
      oldState?: string;
      newState?: string;
    };
    toggleEvent.oldState = oldState;
    toggleEvent.newState = newState;
    this.dispatchEvent(toggleEvent);
  }
  window.HTMLElement.prototype.showPopover = function () {
    syncPopoverState.call(this, true);
  };
  window.HTMLElement.prototype.hidePopover = function () {
    syncPopoverState.call(this, false);
  };
  window.HTMLElement.prototype.togglePopover = function (force?: boolean | { force?: boolean }) {
    const desired =
      typeof force === 'boolean'
        ? force
        : typeof force === 'object' && force !== null && typeof force.force === 'boolean'
          ? force.force
          : this.dataset.popoverOpen !== 'true';
    syncPopoverState.call(this, desired);
    return desired;
  };
});

/**
 * Browsers auto-toggle the popover targeted by a button's `popovertarget`
 * attribute on click. happy-dom does not. Add the missing behavior here so
 * KUI Select/Tooltip/Popover tests can open/close via user interaction.
 *
 * Registered once per test file via `beforeAll` — registering in `beforeEach`
 * stacks duplicate handlers across tests, causing N toggles per click and
 * order-dependent flakes.
 */
beforeAll(() => {
  document.addEventListener(
    'click',
    (event) => {
      const target = event.target;
      if (!(target instanceof window.HTMLElement)) {
        return;
      }
      const button = target.closest('[popovertarget]');
      if (button instanceof window.HTMLElement) {
        const popoverId = button.getAttribute('popovertarget');
        if (popoverId) {
          const popover = document.getElementById(popoverId);
          popover?.togglePopover();
          return;
        }
      }
      // Light-dismiss: clicking outside an open popover="auto" closes it.
      const openPopovers = document.querySelectorAll<HTMLElement>('[popover="auto"][popover-open]');
      openPopovers.forEach((popover) => {
        if (!popover.contains(target)) {
          popover.hidePopover();
        }
      });
    },
    true
  );
});

afterEach(() => {
  // Reset localStorage mock to prevent cross-test state pollution.
  localStorageMock.clear();
  // Clear all timers to prevent timeouts from firing after test completion.
  // This prevents "window is not defined" errors when React tries to update state
  // after the test environment has been torn down.
  vi.clearAllTimers();
  // Restore original implementations for spied functions
  vi.restoreAllMocks();
});
