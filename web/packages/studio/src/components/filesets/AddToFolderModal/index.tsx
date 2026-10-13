// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import {
  Button,
  Flex,
  FormField,
  Modal,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Spinner,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { useDatasetFilesMove } from '@studio/api/datasets/useDatasetFilesMove';
import {
  getParentFolder,
  getSiblingFolders,
} from '@studio/components/filesets/AddToFolderModal/utils';
import { FileSystemNode } from '@studio/components/FilesTable/utils';
import { useDatasetNavigator } from '@studio/hooks/useDatasetNavigator';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getFilesetDetailsRoute } from '@studio/routes/utils';
import { FolderOpen, Info } from 'lucide-react';
import { type FC, useMemo } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const PARENT_FOLDER_VALUE = '..';
const PARENT_FOLDER_DISPLAY = '.. (parent folder)';
const NEW_FOLDER_VALUE = 'New Folder';
export const FOLDER_DELETION_WARNING =
  'This is the last file in the folder, moving it will also delete the folder.';

const formSchema = z
  .object({
    selectedFolder: z.string().min(1, 'Please select a destination'),
    newFolderName: z.string().optional(),
  })
  .refine(
    (data) => {
      if (data.selectedFolder === NEW_FOLDER_VALUE) {
        return data.newFolderName && data.newFolderName.trim().length > 0;
      }
      return true;
    },
    {
      message: 'Folder name is required',
      path: ['newFolderName'],
    }
  );

type FormValues = z.infer<typeof formSchema>;

interface AddToFolderModalProps {
  /** Whether the modal is open */
  open: boolean;
  /** Callback when modal should close */
  onClose: () => void;
  /** Items to move (files only) */
  selectedItems: FileSystemNode[];
  /** Dataset workspace */
  workspace: string;
  /** Dataset name */
  datasetName: string;
  /** Current folder path the user is viewing */
  currentFolder?: string;
  /** Contents of the current folder (for sibling folder options) */
  folderContents?: FileSystemNode[];
  /** Callback when move operation completes (success or error) */
  onComplete?: () => void;
}

export const AddToFolderModal: FC<AddToFolderModalProps> = ({
  open,
  onClose,
  selectedItems,
  workspace,
  datasetName,
  currentFolder,
  folderContents: folderContentsProp,
  onComplete,
}) => {
  const toast = useToast();
  const navigate = useNavigate();
  const routeWorkspace = useWorkspaceFromPath();

  // Fetch files list if folder contents not provided (e.g., from quick actions)
  const { data: fetchedFilesResponse, isLoading: isLoadingFiles } = useFilesListFilesetFiles(
    workspace,
    datasetName,
    undefined,
    { query: { enabled: open && !folderContentsProp } }
  );
  const fetchedFilesList = fetchedFilesResponse?.data;

  // Navigate to current folder to get folder contents
  const navigatedContents = useDatasetNavigator(fetchedFilesList, currentFolder ?? '');

  // Use provided folder contents or navigated contents from fetched files
  const folderContents = folderContentsProp ?? navigatedContents;
  const isLoading = !folderContentsProp && isLoadingFiles;

  // Get file paths to move (only files, no directories)
  const allFilePaths = useMemo(() => {
    return selectedItems.filter((item) => item.type === 'file').map((file) => file.path);
  }, [selectedItems]);

  // Count files in current folder (to detect if moving last file)
  const filesInCurrentFolder = useMemo(() => {
    return folderContents?.filter((item) => item.type === 'file').length ?? 0;
  }, [folderContents]);

  // Check if moving all files from the folder
  const isMovingAllFiles = allFilePaths.length >= filesInCurrentFolder && filesInCurrentFolder > 0;

  // Check if we can go up a level (not at root)
  const canGoUp = Boolean(currentFolder);

  // Get sibling folder options (folders at the same level)
  const siblingFolders = useMemo(() => {
    return getSiblingFolders(folderContents);
  }, [folderContents]);

  // Check if there are any existing folder options (parent or siblings)
  const hasExistingFolders = canGoUp || siblingFolders.length > 0;

  const {
    control,
    handleSubmit,
    reset,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      selectedFolder: '',
      newFolderName: '',
    },
  });

  const selectedFolder = useWatch({ control, name: 'selectedFolder' });

  const isNewFolder = selectedFolder === NEW_FOLDER_VALUE;
  const isMovingToParent = selectedFolder === PARENT_FOLDER_VALUE;

  // Show warning when moving last file to parent (folder will be deleted)
  const showFolderDeletionWarning = isMovingToParent && isMovingAllFiles && canGoUp;

  const { mutateAsync: moveFiles, isPending } = useDatasetFilesMove({
    onError: (err) => {
      toast.error(`Failed to move files: ${err.message}`);
      handleClose();
      onComplete?.();
    },
  });

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleSelectChange = (value: string) => {
    setValue('selectedFolder', value);
    // Clear new folder name when switching away from "New folder"
    if (value !== NEW_FOLDER_VALUE) {
      setValue('newFolderName', '');
    }
  };

  const navigateToParentFolder = () => {
    if (!routeWorkspace) {
      logger.warn('AddToFolderModal: Cannot navigate - workspace context not available');
      return;
    }

    const parentFolder = getParentFolder(currentFolder);
    const datasetFullName = `${workspace}/${datasetName}`;
    navigate(
      // Pass raw values: generatePath encodes path params and URLSearchParams
      // encodes the folder query param.
      getFilesetDetailsRoute(routeWorkspace, datasetFullName, parentFolder ?? undefined)
    );
  };

  const onSubmit = async (values: FormValues) => {
    let targetFolder: string;

    // Determine if we should navigate after move (calculated before move to avoid stale closure)
    const shouldNavigateToParent =
      values.selectedFolder === PARENT_FOLDER_VALUE && isMovingAllFiles && canGoUp;

    if (values.selectedFolder === PARENT_FOLDER_VALUE) {
      // Move up one level
      targetFolder = getParentFolder(currentFolder);
    } else if (values.selectedFolder === NEW_FOLDER_VALUE) {
      // Create new folder path
      const newFolderName = values.newFolderName?.trim() ?? '';
      targetFolder = currentFolder ? `${currentFolder}/${newFolderName}` : newFolderName;
    } else {
      // Move to sibling folder
      targetFolder = currentFolder
        ? `${currentFolder}/${values.selectedFolder}`
        : values.selectedFolder;
    }

    if (allFilePaths.length === 0) {
      toast.error('No files to move');
      return;
    }

    await moveFiles({
      workspace,
      name: datasetName,
      filePaths: allFilePaths,
      targetFolder,
    });

    // Show success message
    toast.success(
      `Successfully moved ${allFilePaths.length} file${allFilePaths.length > 1 ? 's' : ''}`
    );

    // Navigate to parent folder if we moved all files out (folder will be deleted)
    if (shouldNavigateToParent) {
      navigateToParentFolder();
    }

    handleClose();
    onComplete?.();
  };

  return (
    <Modal
      open={open}
      onOpenChange={(isOpen) => !isOpen && !isPending && handleClose()}
      slotHeading={
        <>
          <FolderOpen />
          Move
        </>
      }
      slotFooter={
        <Flex justify="end" gap="density-xs" align="center" className="w-full">
          <Button onClick={handleClose} kind="tertiary" color="neutral" disabled={isPending}>
            Cancel
          </Button>
          <LoadingButton
            onClick={handleSubmit(onSubmit)}
            disabled={!selectedFolder || isLoading}
            loading={isPending}
          >
            Move
          </LoadingButton>
        </Flex>
      }
    >
      <Stack className="overflow-y-hidden" gap="density-md">
        {isLoading ? (
          <Flex justify="center" align="center" className="py-4">
            <Spinner description="Loading folders..." />
          </Flex>
        ) : (
          <>
            <FormField
              name="selectedFolder"
              slotLabel="Folder"
              slotError={errors.selectedFolder?.message}
              status={errors.selectedFolder && 'error'}
            >
              {({ status }) => (
                <SelectRoot
                  value={selectedFolder}
                  onValueChange={handleSelectChange}
                  disabled={isPending}
                >
                  <SelectTrigger
                    placeholder="Select a folder"
                    status={status}
                    aria-label="destination-folder-select"
                  />
                  <SelectContent>
                    <SelectListbox>
                      {/* Parent folder option (if not at root) */}
                      {canGoUp && (
                        <SelectItem value={PARENT_FOLDER_VALUE}>{PARENT_FOLDER_DISPLAY}</SelectItem>
                      )}

                      {/* Sibling folders */}
                      {siblingFolders.map((folder) => (
                        <SelectItem key={folder} value={folder}>
                          {folder}
                        </SelectItem>
                      ))}

                      {/* Empty state message when no existing folders */}
                      {!hasExistingFolders && (
                        <div className="px-3 py-2">
                          <Text className="text-secondary" fontSize="12">
                            No existing folders
                          </Text>
                        </div>
                      )}

                      {/* New folder option with separator */}
                      <SelectItem className="border-t border-base" value={NEW_FOLDER_VALUE}>
                        <Flex gap="density-md" align="center">
                          New Folder
                        </Flex>
                      </SelectItem>
                    </SelectListbox>
                  </SelectContent>
                </SelectRoot>
              )}
            </FormField>

            {/* New folder name input (shown when "New folder" is selected) */}
            {isNewFolder && (
              <ControlledTextInput
                useControllerProps={{ control, name: 'newFolderName' }}
                label="Folder Name"
                placeholder="Enter Folder Name"
                disabled={isPending}
                autoFocus
              />
            )}

            {/* Warning when moving last file to parent */}
            {showFolderDeletionWarning && (
              <Text fontSize="12" role="alert">
                <Flex gap="density-xs" align="center">
                  <Info />
                  {FOLDER_DELETION_WARNING}
                </Flex>
              </Text>
            )}
          </>
        )}
      </Stack>
    </Modal>
  );
};
