// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsListModels as useListModels } from '@nemo/sdk/generated/platform/models';
import { suppressConsoleError } from '@nemo/testing/utils/suppress-console';
import { SearchBaseModels } from '@studio/components/FilterFields/SearchBaseModels';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useDebounce } from 'use-debounce';

// Mock the SDK hook
vi.mock('@nemo/sdk/generated/platform/models', () => ({
  useModelsListModels: vi.fn(),
}));

// Mock useDebounce to avoid timing issues in tests
vi.mock('use-debounce', () => ({
  useDebounce: vi.fn(),
}));

const mockUseListModels = vi.mocked(useListModels);
const mockUseDebounce = vi.mocked(useDebounce);

const mockModelsData = {
  data: [
    { base_model: 'gpt-3.5-turbo', id: '1' },
    { base_model: 'gpt-4', id: '2' },
    { base_model: 'claude-3-opus', id: '3' },
    { base_model: 'llama-2-7b', id: '4' },
  ],
};

// Create a complete mock function that avoids type casting issues
const createMockQueryResult = (overrides: Record<string, unknown> = {}) => ({
  data: mockModelsData,
  isError: false,
  error: null,
  isPending: false,
  isLoading: false,
  isFetching: false,
  isSuccess: true,
  queryKey: ['listModels'],
  ...overrides,
});

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('SearchBaseModels', () => {
  const defaultProps = {
    workspace: 'test-workspace',
    selectedModels: [],
    setSelectedModels: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseListModels.mockReturnValue(createMockQueryResult() as never);
    // Set up default debounce mock behavior
    mockUseDebounce.mockReturnValue(['', false, vi.fn()] as never);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('updates search filter when typing in input2', async () => {
    const user = userEvent.setup();
    render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

    const searchInput = screen.getByPlaceholderText('Search base models...');
    await user.type(searchInput, 'gpt');
    expect(searchInput).toHaveValue('gpt');
  });

  describe('Component Rendering', () => {
    it('renders search input with placeholder text', async () => {
      render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

      expect(await screen.findByPlaceholderText('Search base models...')).toBeInTheDocument();
    });

    it('renders selected models list when models are selected', async () => {
      const propsWithSelected = {
        ...defaultProps,
        selectedModels: ['gpt-3.5-turbo', 'gpt-4'],
      };

      render(<SearchBaseModels {...propsWithSelected} />, { wrapper: createWrapper() });

      expect(await screen.findByText('gpt-3.5-turbo')).toBeInTheDocument();
      expect(await screen.findByText('gpt-4')).toBeInTheDocument();
    });

    it('shows spinner when fetching data', async () => {
      mockUseListModels.mockReturnValue(
        createMockQueryResult({
          isFetching: true,
          isLoading: true,
        }) as never
      );

      render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

      expect(await screen.findByTestId('nv-spinner-spinner')).toBeInTheDocument();
    });
  });

  describe('Search Functionality', () => {
    it('calls useListModels with correct parameters', () => {
      render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

      expect(mockUseListModels).toHaveBeenCalledWith(
        'test-workspace',
        {
          page: 1,
          page_size: 20,
          filter: {
            workspace: 'test-workspace',
          },
        },
        {
          query: {
            enabled: false,
            placeholderData: expect.any(Function),
          },
        }
      );
    });

    it('updates search filter when typing in input', async () => {
      const user = userEvent.setup();
      render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

      const searchInput = screen.getByPlaceholderText('Search base models...');
      await user.type(searchInput, 'gpt');

      await waitFor(() => {
        expect(searchInput).toHaveValue('gpt');
      });
    });

    it('uses debounced search value', () => {
      // Mock debounce to return a search value so the query is enabled
      mockUseDebounce.mockReturnValue(['gpt', false, vi.fn()] as never);

      render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

      expect(mockUseListModels).toHaveBeenCalledWith(
        'test-workspace',
        {
          page: 1,
          page_size: 20,
          filter: {
            workspace: 'test-workspace',
          },
        },
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: true,
            placeholderData: expect.any(Function),
          }),
        })
      );
    });

    it('shows "Start typing to search..." message when no search term', async () => {
      mockUseListModels.mockReturnValue(
        createMockQueryResult({
          data: { data: [] },
        }) as never
      );

      render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByPlaceholderText('Search base models...'));

      await waitFor(() =>
        expect(screen.getByRole('listbox')).toHaveAttribute(
          'data-empty-message',
          'Start typing to search...'
        )
      );
    });

    it('shows "No models found" message when search returns no results', async () => {
      // Mock debounce to return a search term so the query is enabled
      mockUseDebounce.mockReturnValue(['search-term', false, vi.fn()] as never);
      mockUseListModels.mockReturnValue(
        createMockQueryResult({
          data: { data: [] },
        }) as never
      );

      render(<SearchBaseModels {...defaultProps} />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByPlaceholderText('Search base models...'));

      await waitFor(() =>
        expect(screen.getByRole('listbox')).toHaveAttribute(
          'data-empty-message',
          'No models found matching your search'
        )
      );
    });

    it('shows "All available models are already selected" when all models are selected', async () => {
      // Mock debounce to return a search term so the query is enabled
      mockUseDebounce.mockReturnValue(['search-term', false, vi.fn()] as never);
      const propsWithAllSelected = {
        ...defaultProps,
        selectedModels: ['gpt-3.5-turbo', 'gpt-4', 'claude-3-opus', 'llama-2-7b'],
      };

      render(<SearchBaseModels {...propsWithAllSelected} />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByPlaceholderText('Search base models...'));

      await waitFor(() =>
        expect(screen.getByRole('listbox')).toHaveAttribute(
          'data-empty-message',
          'All available models are already selected'
        )
      );
    });
  });

  describe('Model Selection', () => {
    it('filters out already selected models from dropdown', async () => {
      // Mock debounce to return a search term so models are shown
      mockUseDebounce.mockReturnValue(['gpt', false, vi.fn()] as never);
      const propsWithSelected = {
        ...defaultProps,
        selectedModels: ['gpt-3.5-turbo'],
      };

      render(<SearchBaseModels {...propsWithSelected} />, { wrapper: createWrapper() });

      // Click to open the dropdown
      await userEvent.click(screen.getByPlaceholderText('Search base models...'));

      // Should show other models in the dropdown
      expect(screen.getByText('gpt-4')).toBeInTheDocument();
      expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    });

    it('calls setSelectedModels when a model is selected', async () => {
      // Mock debounce to return a search term so models are shown
      mockUseDebounce.mockReturnValue(['gpt', false, vi.fn()] as never);
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };
      const user = userEvent.setup();

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      await user.click(screen.getByPlaceholderText('Search base models...'));

      await user.click(await screen.findByText('gpt-3.5-turbo'));

      expect(mockSetSelectedModels).toHaveBeenCalledWith(['gpt-3.5-turbo']);
    });

    it('clears search input when a model is selected', async () => {
      suppressConsoleError('component suspended inside an `act` scope');
      // Mock debounce to return a search term so models are shown
      mockUseDebounce.mockReturnValue(['gpt', false, vi.fn()] as never);
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };
      const user = userEvent.setup();

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      const searchInput = screen.getByPlaceholderText('Search base models...');
      await user.type(searchInput, 'gpt');

      await user.click(searchInput);
      await user.click(await screen.findByText('gpt-3.5-turbo'));

      await waitFor(() => expect(searchInput).toHaveValue(''));
    });
  });

  describe('Model Removal', () => {
    it('shows remove buttons for selected models', async () => {
      const mockSetSelectedModels = vi.fn();
      const propsWithSelected = {
        ...defaultProps,
        selectedModels: ['gpt-3.5-turbo', 'gpt-4'],
        setSelectedModels: mockSetSelectedModels,
      };

      render(<SearchBaseModels {...propsWithSelected} />, { wrapper: createWrapper() });

      await waitFor(() =>
        expect(screen.getAllByLabelText('Remove base model from filter')).toHaveLength(2)
      );
    });

    it('calls setSelectedModels to remove a model when remove button is clicked', async () => {
      const user = userEvent.setup();
      const mockSetSelectedModels = vi.fn();
      const propsWithSelected = {
        ...defaultProps,
        selectedModels: ['gpt-3.5-turbo', 'gpt-4'],
        setSelectedModels: mockSetSelectedModels,
      };

      render(<SearchBaseModels {...propsWithSelected} />, { wrapper: createWrapper() });

      const removeButtons = screen.getAllByLabelText('Remove base model from filter');
      await user.click(removeButtons[0]);

      expect(mockSetSelectedModels).toHaveBeenCalledWith(['gpt-4']);
    });
  });

  describe('Data Handling', () => {
    it('handles empty models data gracefully', async () => {
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };

      mockUseListModels.mockReturnValue(
        createMockQueryResult({
          data: null,
        }) as never
      );

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      expect(await screen.findByPlaceholderText('Search base models...')).toBeInTheDocument();
    });

    it('handles models with invalid base_model values', async () => {
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };

      const invalidModelsData = {
        data: [
          { base_model: null, id: '1' },
          { base_model: undefined, id: '2' },
          { base_model: 123, id: '3' }, // non-string value
          { base_model: 'valid-model', id: '4' },
        ],
      };

      mockUseDebounce.mockReturnValue(['search', false, vi.fn()] as never);
      mockUseListModels.mockReturnValue(
        createMockQueryResult({
          data: invalidModelsData,
        }) as never
      );

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByPlaceholderText('Search base models...'));

      // Should only show the valid model
      await screen.findByText('valid-model');
      await waitFor(() => expect(screen.queryByText('null')).not.toBeInTheDocument());
      await waitFor(() => expect(screen.queryByText('undefined')).not.toBeInTheDocument());
      await waitFor(() => expect(screen.queryByText('123')).not.toBeInTheDocument());
    });

    it('removes duplicate base models from the dropdown', async () => {
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };

      const duplicateModelsData = {
        data: [
          { base_model: 'gpt-3.5-turbo', id: '1' },
          { base_model: 'gpt-3.5-turbo', id: '2' },
          { base_model: 'gpt-4', id: '3' },
          { base_model: 'gpt-4', id: '4' },
        ],
      };

      mockUseDebounce.mockReturnValue(['gpt', false, vi.fn()] as never);
      mockUseListModels.mockReturnValue(
        createMockQueryResult({
          data: duplicateModelsData,
        }) as never
      );

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByPlaceholderText('Search base models...'));

      // Should only show each model once
      const gpt35Options = screen.getAllByText('gpt-3.5-turbo');
      const gpt4Options = screen.getAllByText('gpt-4');

      await waitFor(() => expect(gpt35Options).toHaveLength(1));
      await waitFor(() => expect(gpt4Options).toHaveLength(1));
    });
  });

  describe('Debounce Integration', () => {
    it('uses debounced value for API calls', () => {
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };

      mockUseDebounce.mockReturnValue(['debounced-search', false, vi.fn()] as never);

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      expect(mockUseListModels).toHaveBeenCalledWith(
        'test-workspace',
        {
          page: 1,
          page_size: 20,
          filter: {
            workspace: 'test-workspace',
          },
        },
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: true,
            placeholderData: expect.any(Function),
          }),
        })
      );
    });

    it('enables query when debounced search has value', () => {
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };

      mockUseDebounce.mockReturnValue(['search-term', false, vi.fn()] as never);

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      expect(mockUseListModels).toHaveBeenCalledWith(
        'test-workspace',
        expect.any(Object),
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: true,
          }),
        })
      );
    });

    it('disables query when debounced search is empty', () => {
      const mockSetSelectedModels = vi.fn();
      const testProps = {
        ...defaultProps,
        setSelectedModels: mockSetSelectedModels,
      };

      mockUseDebounce.mockReturnValue(['', false, vi.fn()] as never);

      render(<SearchBaseModels {...testProps} />, { wrapper: createWrapper() });

      expect(mockUseListModels).toHaveBeenCalledWith(
        'test-workspace',
        expect.any(Object),
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: false,
          }),
        })
      );
    });
  });
});
