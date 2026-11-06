// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Plugin API: the surface Studio serves to plugin bundles as `@nemo/common`.
// Removals are breaking. Explicit exports, not `export *`.

export { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
export { AssistantChat } from '@nemo/common/src/components/AssistantChat';
export type {
  AssistantChatMessageContentProps,
  AssistantChatProps,
} from '@nemo/common/src/components/AssistantChat/types';
export { AccordionSection } from '@nemo/common/src/components/AccordionSection';
// Presentational only: pass `agents` (and `onSearchChange`/`onLoadMore` to page) from `host.sdk`.
// WorkspaceAgentSelect is deliberately NOT exported — it fetches.
export { AgentSelect } from '@nemo/common/src/components/AgentSelect/AgentSelect';
export type {
  AgentSelectProps,
  MultiAgentSelectProps,
  SingleAgentSelectProps,
} from '@nemo/common/src/components/AgentSelect/types';
export {
  groupAgentsByFormat,
  matchesAgentSearch,
} from '@nemo/common/src/components/AgentSelect/utils';
// CancelJobButton is deliberately NOT exported
export { ConfirmationModal } from '@nemo/common/src/components/ConfirmationModal';
export { CreateSecretModal } from '@nemo/common/src/components/CreateSecretModal';
export type {
  CreateSecretFormData,
  CreateSecretModalProps,
} from '@nemo/common/src/components/CreateSecretModal';
export { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
export type { AccordionSectionProps } from '@nemo/common/src/components/AccordionSection';
// ErrorPanel is deliberately NOT exported
export { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
export { FileTag } from '@nemo/common/src/components/FileTag';
export type { FileTagProps, FileTagStatus } from '@nemo/common/src/components/FileTag';
export { FileUpload } from '@nemo/common/src/components/FileUpload';
export type { FileUploadProps, RenderFileTagFn } from '@nemo/common/src/components/FileUpload';
export { InputErrorText } from '@nemo/common/src/components/InputErrorText';
export { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
export type { QuickActionItem } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
export type { DeleteAction, DeleteActionLabel } from '@nemo/common/src/components/ActionsMenu';

export {
  StudioDataView,
  StudioDataViewToolbar,
} from '@nemo/common/src/components/DataView/StudioDataView';
export type { StudioDataViewToolbarProps } from '@nemo/common/src/components/DataView/StudioDataView';
export {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
} from '@nemo/common/src/components/DataView/StudioDataView';
export * as DataView from '@nemo/common/src/components/DataView/internal';

export { FormModal } from '@nemo/common/src/components/FormModal';
export type { FormModalProps } from '@nemo/common/src/components/FormModal';
export { LoadingButton } from '@nemo/common/src/components/LoadingButton';
export { LogViewer } from '@nemo/common/src/components/LogViewer';
export { ModelSelectV2 } from '@nemo/common/src/components/ModelSelectV2/ModelSelectV2';
export {
  WorkspaceModelSelectCore as WorkspaceModelSelect,
  type WorkspaceModelSelectCoreProps as WorkspaceModelSelectProps,
} from '@nemo/common/src/components/ModelSelectV2/WorkspaceModelSelectCore';
export type {
  ModelSelection,
  ModelSelectV2Props,
  ModelType,
} from '@nemo/common/src/components/ModelSelectV2/types';
export type { LogStatus } from '@nemo/common/src/utils/logs';
export { RadioCard } from '@nemo/common/src/components/RadioCard';
export type { RadioCardProps } from '@nemo/common/src/components/RadioCard';
export { RelativeTime, useRelativeTimeSince } from '@nemo/common/src/components/RelativeTime';
export { StatusBadge } from '@nemo/common/src/components/StatusBadge';
export type { BadgeStatus, StatusConfigEntry } from '@nemo/common/src/components/StatusBadge';
export { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';

export { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
export { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
export { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';

export { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
export type {
  ApiFilter,
  StudioDataViewState,
  UseStudioDataViewStateOptions,
} from '@nemo/common/src/hooks/useStudioDataViewState';

export { withOperators } from '@nemo/common/src/api/filterOperators';
export type { FilterOperators, WithFilterOperators } from '@nemo/common/src/api/filterOperators';

export {
  MODEL_SEARCH_PAGE_SIZE,
  useModelSearchCore as useModelSearch,
  type ModelSearchFilter,
  type ModelSearchProps,
  type UseModelSearchCoreOptions as UseModelSearchOptions,
  type UseModelSearchResult,
} from '@nemo/common/src/api/models/useModelSearchCore';
export type { ListModels, ModelWorkspaceGroup } from '@nemo/common/src/api/models/modelPages';

export { fetchAllPages } from '@nemo/common/src/api/fetchAllPages';
export type { FetchAllPagesOptions, PaginatedResponse } from '@nemo/common/src/api/fetchAllPages';

export {
  getJobRefetchInterval,
  getSortParam,
  getSortParamWithWhitelist,
} from '@nemo/common/src/utils/query';
export { triggerDownload } from '@nemo/common/src/utils/file';
export { MIME_TYPES, toCsv, toCsvCell, toJsonl } from '@nemo/common/src/utils/download';
export type { CsvOptions, CsvRow, DownloadFormat } from '@nemo/common/src/utils/download';

export { ENTITY_NAME_HELP, entityNameSchema } from '@nemo/common/src/utils/entityName';

export { getErrorMessage } from '@nemo/common/src/utils/error';
export { handleFormErrorsGeneric } from '@nemo/common/src/utils/forms/error';
export { logger, toError } from '@nemo/common/src/utils/logger';

export type { NotifyFn, NotifyType } from '@nemo/common/src/providers/toast/types';

export { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
export {
  DEFAULT_PAGE,
  DEFAULT_PAGE_SIZE,
  DEFAULT_PAGE_SIZE_OPTIONS,
} from '@nemo/common/src/constants/pagination';
export {
  CJobCancellableStatuses,
  CJobLaunchableStatuses,
  CJobTerminalStatuses,
  HelixJobTerminalStatuses,
} from '@nemo/common/src/constants/query';
