// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { ModelDropdown } from '@nemo/common/src/components/ModelSelectV2/ModelDropdown';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

const makeModel = (name: string, overrides: Partial<ModelEntity> = {}): ModelEntity =>
  ({ id: name, name, workspace: 'nvidia', ...overrides }) as unknown as ModelEntity;

const groups: ModelWorkspaceGroup[] = [
  { workspace: 'nvidia', models: [makeModel('nemotron-8b'), makeModel('llama-3.1-8b')] },
];

const withAdapters: ModelWorkspaceGroup[] = [
  {
    workspace: 'nvidia',
    models: [
      makeModel('nemotron-8b', {
        adapters: [
          {
            name: 'support-v1',
            created_at: '2026-01-10T00:00:00Z',
            workspace: 'nvidia',
            fileset: 'nvidia/support-v1',
            finetuning_type: 'lora',
          },
        ],
      }),
    ],
  },
];

const MANY_MODELS = 200;

const manyGroups: ModelWorkspaceGroup[] = [
  {
    workspace: 'nvidia',
    models: Array.from({ length: MANY_MODELS }, (_, i) => makeModel(`model-${i}`)),
  },
];

/**
 * jsdom has no layout, so the viewport is stubbed to make the virtualizer engage. Returns a
 * handle for driving the resize callback the list subscribes to.
 */
const stubViewport = () => {
  type ResizeCallback = (entries: ResizeObserverEntry[]) => void;
  const callbacks: ResizeCallback[] = [];
  let height = 0;

  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(() => height);
  // Rows measure themselves; jsdom would report every one as zero-height, collapsing the window.
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    width: 360,
    height: 36,
  } as DOMRect);
  vi.stubGlobal(
    'ResizeObserver',
    class {
      constructor(callback: ResizeCallback) {
        callbacks.push(callback);
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  );

  return {
    resizeTo: (next: number) => {
      height = next;
      // The virtualizer reads `borderBoxSize` off the entry; the list ignores the argument.
      const entries = [
        { borderBoxSize: [{ inlineSize: 360, blockSize: next }] },
      ] as unknown as ResizeObserverEntry[];
      callbacks.forEach((callback) => callback(entries));
    },
  };
};

type Props = React.ComponentProps<typeof ModelDropdown>;

const renderOpen = (props: Partial<Props> = {}) =>
  render(
    <ModelDropdown
      open
      onOpenChange={vi.fn()}
      value={null}
      onValueChange={vi.fn()}
      groups={groups}
      {...props}
    />
  );

const typeFilter = (text: string) =>
  fireEvent.change(screen.getByTestId('model-select-v2-filter'), { target: { value: text } });

/** Names of the rows in the list, ignoring the details panel each row also renders. */
const listedModels = () =>
  screen.queryAllByTestId('model-dropdown-item').map((item) => item.textContent);

describe('ModelDropdown', () => {
  describe('search', () => {
    it('filters the given groups itself when no onSearchChange is provided', async () => {
      renderOpen();
      await waitFor(() => expect(listedModels()).toHaveLength(2));

      typeFilter('llama');

      await waitFor(() => expect(listedModels()).toEqual(['llama-3.1-8b']));
    });

    it('reports the debounced term and leaves the groups alone when onSearchChange is provided', async () => {
      const onSearchChange = vi.fn();
      renderOpen({ onSearchChange, searchDebounceMs: 0 });

      typeFilter('llama');

      await waitFor(() => expect(onSearchChange).toHaveBeenCalledWith('llama'));
      // The caller owns the query, so both models stay listed until new groups arrive.
      expect(listedModels()).toEqual(['nemotron-8b', 'llama-3.1-8b']);
    });
  });

  describe('paging', () => {
    it('asks for the next page while more remain', async () => {
      const onLoadMore = vi.fn();
      renderOpen({ onLoadMore, hasMore: true });

      await waitFor(() => expect(onLoadMore).toHaveBeenCalled());
    });

    it('does not ask for more once the last page has loaded', async () => {
      const onLoadMore = vi.fn();
      renderOpen({ onLoadMore, hasMore: false });

      await waitFor(() => expect(listedModels()).toHaveLength(2));
      expect(onLoadMore).not.toHaveBeenCalled();
    });

    it('shows the loading message instead of the empty state while paging past filtered-out pages', () => {
      renderOpen({
        groups: [],
        hasMore: true,
        isLoadingMore: true,
        emptyMessage: 'No models found',
      });

      expect(screen.getByText('Loading models...')).toBeInTheDocument();
      expect(screen.queryByText('No models found')).not.toBeInTheDocument();
    });

    it('shows the done message only after the last page', async () => {
      const { rerender } = renderOpen({
        onLoadMore: vi.fn(),
        hasMore: true,
        doneLoadingMessage: 'No more models',
      });
      await waitFor(() => expect(listedModels()).toHaveLength(2));
      expect(screen.queryByText('No more models')).not.toBeInTheDocument();

      rerender(
        <ModelDropdown
          open
          onOpenChange={vi.fn()}
          value={null}
          onValueChange={vi.fn()}
          groups={groups}
          onLoadMore={vi.fn()}
          hasMore={false}
          doneLoadingMessage="No more models"
        />
      );

      expect(await screen.findByText('No more models')).toBeInTheDocument();
    });
  });

  describe('virtualization', () => {
    afterEach(() => {
      vi.restoreAllMocks();
      vi.unstubAllGlobals();
    });

    it('renders nothing at all while the menu is closed', () => {
      renderOpen({ open: false, groups: manyGroups });

      // DropdownContent is a native popover and stays mounted, so a closed menu would otherwise
      // keep a row (and its submenu popover) in the DOM for every model in the catalogue.
      expect(listedModels()).toHaveLength(0);
      expect(screen.queryByTestId('model-select-v2-filter')).not.toBeInTheDocument();
    });

    // Regression: the viewport used to be measured once on mount. The menu is a popover, so that
    // measurement read zero height and disabled windowing for good — every model in the workspace
    // rendered at once, which is what ground the page to a halt on large catalogues.
    it('windows the list once the popover gains height after mount', async () => {
      const viewport = stubViewport();
      renderOpen({ groups: manyGroups });

      await act(async () => viewport.resizeTo(300));

      // A 300px viewport over 36px rows is ~9 visible, plus overscan either side — nowhere near
      // the 200 in the catalogue.
      const rendered = listedModels().length;
      expect(rendered).toBeGreaterThan(0);
      expect(rendered).toBeLessThan(40);
    });

    it('falls back to rendering every row when there is no viewport to window into', async () => {
      renderOpen({ groups: manyGroups });

      await waitFor(() => expect(listedModels()).toHaveLength(MANY_MODELS));
    });
  });

  describe('row details', () => {
    // KUI's DropdownSubContent is always in the DOM, so rendering its body eagerly would mount a
    // details panel (and every adapter's) for each row in view — the cost that made typing lag.
    it('does not mount a row’s details until its submenu is opened', async () => {
      renderOpen({ groups: withAdapters });
      await waitFor(() =>
        expect(screen.getAllByTestId('model-dropdown-item-with-adapters')).toHaveLength(1)
      );

      expect(screen.queryByTestId('model-dropdown-adapter-option')).not.toBeInTheDocument();
      expect(screen.queryByText('Fine-tuning Type')).not.toBeInTheDocument();
    });

    it('mounts the details on hover and keeps them mounted afterwards', async () => {
      renderOpen({ groups: withAdapters });
      const sub = (await screen.findAllByTestId('nv-dropdown-sub'))[0];

      fireEvent.pointerEnter(sub);

      expect(await screen.findByTestId('model-dropdown-adapter-option')).toBeInTheDocument();

      fireEvent.pointerLeave(sub, { relatedTarget: document.body });

      // Latched: leaving does not throw the panel away, so a second hover is instant.
      expect(screen.getByTestId('model-dropdown-adapter-option')).toBeInTheDocument();
    });

    it('checks the selected adapter and not the base model it belongs to', async () => {
      renderOpen({
        groups: withAdapters,
        value: { model: 'nvidia/nemotron-8b', adapter: 'support-v1' },
      });
      const sub = (await screen.findAllByTestId('nv-dropdown-sub'))[0];

      fireEvent.pointerEnter(sub);

      const adapterRow = await screen.findByTestId('model-dropdown-adapter-option');
      expect(within(adapterRow).getByTestId('model-dropdown-selected-check')).toBeInTheDocument();
      expect(
        within(screen.getByTestId('model-dropdown-base-option')).queryByTestId(
          'model-dropdown-selected-check'
        )
      ).not.toBeInTheDocument();
    });
  });

  describe('trigger', () => {
    it('falls back to the name in the URN when the selection is not in the loaded pages', () => {
      renderOpen({ open: false, value: { model: 'nvidia/not-yet-loaded' }, groups: [] });

      expect(screen.getByTestId('model-select-v2-trigger')).toHaveTextContent('not-yet-loaded');
    });

    // Data Designer templates seed a bare model name, which only becomes a URN once auto-fill
    // resolves it. Showing the placeholder over it reads as "nothing selected".
    it('shows a reference that has no workspace prefix rather than the placeholder', () => {
      renderOpen({
        open: false,
        value: { model: 'nvidia-llama-3-3-nemotron-super-49b-v1' },
        groups: [],
        placeholder: 'Select a model',
      });

      expect(screen.getByTestId('model-select-v2-trigger')).toHaveTextContent(
        'nvidia-llama-3-3-nemotron-super-49b-v1'
      );
    });

    it('prefers the entity the selection carries', () => {
      renderOpen({
        open: false,
        value: { model: 'nvidia/nemotron-8b', entity: makeModel('nemotron-8b') },
        groups: [],
      });

      expect(screen.getByTestId('model-select-v2-trigger')).toHaveTextContent('nemotron-8b');
    });

    it('qualifies the label with the adapter when one is selected', () => {
      renderOpen({
        open: false,
        value: { model: 'nvidia/nemotron-8b', adapter: 'support-v1' },
        groups: withAdapters,
      });

      expect(screen.getByTestId('model-select-v2-trigger')).toHaveTextContent(
        'nemotron-8b / support-v1'
      );
    });
  });
});
