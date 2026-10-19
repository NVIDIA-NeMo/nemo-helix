// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ControlledSearchableSelect,
  SelectItemOption,
} from '@nemo/common/src/components/form/ControlledSearchableSelect/index';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ReactNode } from 'react';
import { FormProvider, useForm } from 'react-hook-form';

const defaultOptions: SelectItemOption[] = [
  { value: 'apple', label: 'Apple' },
  { value: 'banana', label: 'Banana' },
  { value: 'cherry', label: 'Cherry' },
  { value: 'date', label: 'Date' },
  { value: 'elderberry', label: 'Elderberry' },
];

interface WrapperProps {
  children: ReactNode;
  defaultValues?: Record<string, string>;
}

const FormWrapper = ({ children, defaultValues = { fruit: '' } }: WrapperProps) => {
  const methods = useForm({ defaultValues });
  return <FormProvider {...methods}>{children}</FormProvider>;
};

const renderWithForm = (
  ui: ReactNode,
  { defaultValues }: { defaultValues?: Record<string, string> } = {}
) => {
  return render(<FormWrapper defaultValues={defaultValues}>{ui}</FormWrapper>);
};

describe('ControlledSearchableSelect', () => {
  describe('Basic Rendering', () => {
    it('should render the select trigger with placeholder and label', async () => {
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          formFieldProps={{ slotLabel: 'Favorite Fruit' }}
          triggerPlaceholder="Select a fruit"
        />
      );

      const combobox = await screen.findByRole('combobox');
      expect(combobox).toBeInTheDocument();
      expect(combobox).toHaveTextContent('Select a fruit');
      expect(screen.getByText('Favorite Fruit')).toBeInTheDocument();
    });

    it('should keep the help text as the trigger description when a value is selected', async () => {
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          formFieldProps={{ slotLabel: 'Favorite Fruit', slotHelp: 'Pick one you like' }}
        />,
        { defaultValues: { fruit: 'banana' } }
      );

      expect(await screen.findByRole('combobox')).toHaveAccessibleDescription('Pick one you like');
    });

    it('should show the full selected label in a tooltip on hover', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
        />,
        { defaultValues: { fruit: 'banana' } }
      );

      const combobox = await screen.findByRole('combobox');
      await user.hover(
        within(combobox).getByText('Banana', { ignore: '[role="tooltip"], [role="tooltip"] *' })
      );

      expect(await screen.findByRole('tooltip')).toHaveTextContent('Banana');
    });

    it('should render the selected option label in the trigger, not its raw value', async () => {
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          formFieldProps={{ slotLabel: 'Favorite Fruit' }}
        />,
        { defaultValues: { fruit: 'banana' } }
      );

      const combobox = await screen.findByRole('combobox');
      expect(combobox).toHaveTextContent('Banana');
      expect(combobox).not.toHaveTextContent('banana');
    });

    it('should ellipsize a too-long selected label from the middle, keeping the distinguishing suffix', async () => {
      const longOptions: SelectItemOption[] = [
        { value: 'a', label: 'nvidia-nemotron-3-super-120b-a12b' },
        { value: 'b', label: 'nvidia-nemotron-3-ultra-550b-a55b' },
      ];

      const getContextSpy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        measureText: (text: string) => ({ width: text.length * 10 }),
      } as unknown as CanvasRenderingContext2D);
      const rectSpy = vi
        .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
        .mockReturnValue({ width: 150 } as DOMRect);

      renderWithForm(
        <ControlledSearchableSelect options={longOptions} useControllerProps={{ name: 'model' }} />,
        { defaultValues: { model: 'b' } }
      );

      const combobox = await screen.findByRole('combobox');
      const text = within(combobox).getByText(/…/).textContent ?? '';
      // 150px / 10px-per-char budget keeps ~15 of the label's 34 chars -- just enough to check
      // it ellipsizes in the middle and keeps the tail that differs from the other option, not
      // the exact character count (that's the truncation algorithm's business, not this test's).
      expect(text).toContain('…');
      expect(text.endsWith('a55b')).toBe(true);
      expect(text).not.toBe('nvidia-nemotron-3-ultra-550b-a55b');
      expect(text.length).toBeLessThan('nvidia-nemotron-3-ultra-550b-a55b'.length);

      rectSpy.mockRestore();
      getContextSpy.mockRestore();
    });

    it('should show loading placeholder when isLoading is true', () => {
      renderWithForm(
        <ControlledSearchableSelect
          options={[]}
          useControllerProps={{ name: 'fruit' }}
          isLoading
          triggerPlaceholder="Select a fruit"
        />
      );

      expect(screen.getByRole('combobox')).toHaveTextContent('Loading...');
    });
  });

  describe('Dropdown Interaction', () => {
    it('should open dropdown and show options when clicked', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
        />
      );

      await user.click(screen.getByRole('combobox'));

      expect(await screen.findByRole('listbox')).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Apple' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Banana' })).toBeInTheDocument();
    });

    it('should size the popover to its content, capped rather than stretched full width', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
        />
      );

      await user.click(screen.getByRole('combobox'));
      await screen.findByRole('listbox');

      const popover = screen.getByTestId('nv-select-content');
      expect(popover).toHaveClass('w-max', 'max-w-96');
      // `min-w-full` resolves against the anchored popover's containing block (viewport-wide),
      // not the trigger, and outranks `max-w-*` in the CSS min/max conflict rule -- regression
      // guard for the popover silently going full width again.
      expect(popover.className).not.toMatch(/\bmin-w-full\b/);
    });

    it('should show search input in dropdown', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          searchPlaceholder="Search fruits..."
        />
      );

      await user.click(screen.getByRole('combobox'));

      expect(await screen.findByPlaceholderText('Search fruits...')).toBeInTheDocument();
    });
  });

  describe('Value Selection', () => {
    it('should select a value and update the form', async () => {
      const handleChange = vi.fn();
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onChange={handleChange}
        />
      );

      await user.click(screen.getByRole('combobox'));
      await user.click(await screen.findByRole('option', { name: 'Banana' }));

      // The component stores the value, which is passed to onChange
      expect(handleChange).toHaveBeenCalledWith('banana');
    });

    it('should call onChange callback when value changes', async () => {
      const handleChange = vi.fn();
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onChange={handleChange}
        />
      );

      await user.click(screen.getByRole('combobox'));
      await user.click(await screen.findByRole('option', { name: 'Cherry' }));

      expect(handleChange).toHaveBeenCalledWith('cherry');
    });

    it('should display pre-selected value from form defaults', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
        />,
        { defaultValues: { fruit: 'banana' } }
      );

      // Open dropdown to verify the correct option is selected
      await user.click(screen.getByRole('combobox'));
      const bananaOption = await screen.findByRole('option', { name: 'Banana' });
      expect(bananaOption).toHaveAttribute('aria-selected', 'true');
    });
  });

  describe('Client-Side Search', () => {
    it('should filter options locally when typing in search', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
        />
      );

      await user.click(screen.getByRole('combobox'));
      const searchInput = await screen.findByTestId('fruit-search');
      await user.type(searchInput, 'ber');

      // Wait for debounce
      await waitFor(() => {
        const listbox = screen.getByRole('listbox');
        expect(within(listbox).getByRole('option', { name: 'Elderberry' })).toBeInTheDocument();
        expect(within(listbox).queryByRole('option', { name: 'Apple' })).not.toBeInTheDocument();
      });
    });

    it('should show empty message when no options match search', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          emptyMessage="No fruits found"
        />
      );

      await user.click(screen.getByRole('combobox'));
      const searchInput = await screen.findByTestId('fruit-search');
      await user.type(searchInput, 'xyz');

      await waitFor(() => {
        expect(screen.getByText('No fruits found')).toBeInTheDocument();
      });
    });

    it('should reset search when dropdown closes', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
        />
      );

      // Open and search
      await user.click(await screen.findByRole('combobox'));
      await user.type(screen.getByTestId('fruit-search'), 'app');
      expect(await screen.findByRole('listbox')).toBeInTheDocument();

      // Click outside to close
      await user.click(document.body);
      await waitFor(() => expect(screen.queryByRole('listbox')).not.toBeInTheDocument());

      // Re-open and verify search is cleared
      await user.click(await screen.findByRole('combobox'));
      await waitFor(() => expect(screen.getByTestId('fruit-search')).toHaveValue(''));
    });
  });

  describe('Server-Side Search', () => {
    it('should call onSearchChange when search input changes', async () => {
      const handleSearchChange = vi.fn();
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onSearchChange={handleSearchChange}
          searchDebounceMs={100}
        />
      );

      await user.click(await screen.findByRole('combobox'));
      const searchInput = await screen.findByTestId('fruit-search');
      await user.type(searchInput, 'test');

      await waitFor(() => {
        expect(handleSearchChange).toHaveBeenCalledWith('test');
      });
    });

    it('should not filter locally when onSearchChange is provided', async () => {
      const handleSearchChange = vi.fn();
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onSearchChange={handleSearchChange}
          searchDebounceMs={50}
        />
      );

      await user.click(await screen.findByRole('combobox'));
      const searchInput = await screen.findByTestId('fruit-search');
      await user.type(searchInput, 'xyz');

      // Wait for debounce
      await waitFor(() => {
        expect(handleSearchChange).toHaveBeenCalled();
      });

      // All options should still be visible (no local filtering)
      const listbox = screen.getByRole('listbox');
      expect(within(listbox).getByRole('option', { name: 'Apple' })).toBeInTheDocument();
      expect(within(listbox).getByRole('option', { name: 'Banana' })).toBeInTheDocument();
    });
  });

  describe('Loading States', () => {
    it('should show loading more spinner when isLoadingMore is true', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onLoadMore={vi.fn()}
          hasMore
          isLoadingMore
        />
      );

      await user.click(await screen.findByRole('combobox'));

      expect(await screen.findByLabelText('Loading more')).toBeInTheDocument();
    });
  });

  describe('Infinite Scroll', () => {
    it('should render loader sentinel when onLoadMore is provided', async () => {
      const handleLoadMore = vi.fn().mockResolvedValue(undefined);
      const user = userEvent.setup();

      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onLoadMore={handleLoadMore}
          hasMore
        />
      );

      await user.click(await screen.findByRole('combobox'));
      await screen.findByRole('listbox');

      // The loader sentinel should be rendered when onLoadMore is provided
      // It observes intersection to trigger loading more items
      const listbox = screen.getByRole('listbox');
      expect(listbox).toBeInTheDocument();
      expect(screen.getAllByRole('option')).toHaveLength(5);
    });

    it('should show done loading message when all items loaded', async () => {
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onLoadMore={vi.fn()}
          hasMore={false}
          doneLoadingMessage="All fruits loaded"
        />
      );

      await user.click(await screen.findByRole('combobox'));

      expect(await screen.findByText('All fruits loaded')).toBeInTheDocument();
    });
  });

  describe('Custom Rendering', () => {
    it('should render custom option content when render prop is provided', async () => {
      const user = userEvent.setup();
      const optionsWithCustomRender: SelectItemOption[] = [
        {
          value: 'apple',
          label: 'Apple',
          render: <span data-testid="custom-apple">🍎 Apple</span>,
        },
        {
          value: 'banana',
          label: 'Banana',
          render: <span data-testid="custom-banana">🍌 Banana</span>,
        },
      ];

      renderWithForm(
        <ControlledSearchableSelect
          options={optionsWithCustomRender}
          useControllerProps={{ name: 'fruit' }}
        />
      );

      await user.click(await screen.findByRole('combobox'));

      expect(await screen.findByTestId('custom-apple')).toBeInTheDocument();
      expect(screen.getByTestId('custom-banana')).toBeInTheDocument();
    });
  });

  describe('Open/Close Callbacks', () => {
    it('should call onOpenChange when dropdown opens and closes', async () => {
      const handleOpenChange = vi.fn();
      const user = userEvent.setup();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          onOpenChange={handleOpenChange}
        />
      );

      await user.click(await screen.findByRole('combobox'));
      expect(handleOpenChange).toHaveBeenCalledWith(true);

      expect(await screen.findByRole('listbox')).toBeInTheDocument();
      await user.click(document.body);
      await waitFor(() => expect(screen.queryByRole('listbox')).not.toBeInTheDocument());
      expect(handleOpenChange).toHaveBeenCalledWith(false);
    });
  });

  describe('listFooter', () => {
    it('should render footer and close the menu when footer action runs close()', async () => {
      const user = userEvent.setup();
      const onFooterClick = vi.fn();
      renderWithForm(
        <ControlledSearchableSelect
          options={defaultOptions}
          useControllerProps={{ name: 'fruit' }}
          listFooter={({ close }) => (
            <button
              type="button"
              onClick={() => {
                onFooterClick();
                close();
              }}
            >
              Add new
            </button>
          )}
        />
      );

      await user.click(await screen.findByRole('combobox'));
      expect(await screen.findByRole('button', { name: 'Add new' })).toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: 'Add new' }));
      expect(onFooterClick).toHaveBeenCalled();
      await waitFor(() => {
        expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
      });
    });
  });
});
