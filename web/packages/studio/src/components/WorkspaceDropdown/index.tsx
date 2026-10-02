// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { useEntitiesListWorkspaces } from '@nemo/sdk/generated/platform/entity-store';
import { Workspace } from '@nemo/sdk/generated/platform/schema';
import {
  Block,
  DropdownContent,
  DropdownHeading,
  DropdownItem,
  DropdownRoot,
  DropdownSection,
  DropdownTrigger,
  Flex,
  Skeleton,
  Spinner,
  Stack,
  Text,
  TextInput,
} from '@nvidia/foundations-react-core';
import { dataTestIds } from '@studio/components/WorkspaceDropdown/constants';
import { useRecentWorkspaces } from '@studio/components/WorkspaceDropdown/useRecentWorkspaces';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { useBoolean } from '@studio/util/hooks/useBoolean';
import cn from 'classnames';
import { Plus, Filter } from 'lucide-react';
import { ChangeEvent, FC, lazy, startTransition, Suspense, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router';

const loadWorkspaceCreateModal = () => import('@studio/components/WorkspaceCreateModal');

const WorkspaceCreateModal = lazy(() =>
  loadWorkspaceCreateModal().then((module) => ({
    default: module.WorkspaceCreateModal,
  }))
);

interface Props {
  onValueChange?: (value: string) => void;
}

const URL_SEGMENT_ALLOWLIST = new Set(['new', 'launch', 'focused-view']);

// TODO: Pagination
export const WorkspaceDropdown: FC<Props> = ({ onValueChange }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [filter, setFilter] = useState('');
  const [open, setOpen] = useState(false);
  const { workspace: activeWorkspaceName } = useParams();
  const { recentWorkspaces, addRecentWorkspace } = useRecentWorkspaces();

  const mostRecentWorkspace = recentWorkspaces?.[0];

  const {
    data: workspacesResponse,
    isPending,
    isError,
    refetch,
  } = useEntitiesListWorkspaces(
    {
      page: 1,
      page_size: DEFAULT_LARGE_PAGE_SIZE,
    },
    {
      query: {
        // Always enable when there's an active workspace route or dropdown is open.
        // The query stays mounted, so the list is refetched whenever the dropdown opens.
        enabled: open || !mostRecentWorkspace || !!activeWorkspaceName,
        staleTime: 5_000,
      },
    }
  );

  const [isModalOpen, openModal, closeModal] = useBoolean(false);

  const workspaces = useMemo(() => {
    return workspacesResponse?.data?.sort((workspaceA, workspaceB) => {
      const a = workspaceA.updated_at;
      const b = workspaceB.updated_at;
      if (a && b) {
        return new Date(b).getTime() - new Date(a).getTime();
      }
      return 0;
    });
  }, [workspacesResponse]);

  const groups = useMemo(() => {
    const filterLower = filter.toLowerCase();
    if (!workspaces) {
      return [];
    }
    const baseGroups: { header: string; workspaces: Workspace[] }[] = [];
    if (filter) {
      baseGroups.push({ header: 'Search results', workspaces: [] });
    } else {
      baseGroups.push({ header: 'Recent Workspaces', workspaces: [] });
      baseGroups.push({ header: 'Workspaces', workspaces: [] });
    }
    return workspaces.reduce((acc, workspace) => {
      const name = workspace.name?.toLowerCase() || '';
      const isRecent = recentWorkspaces.some((recentName) => recentName === workspace.name);
      const filterMatch = name.includes(filterLower);
      if ((filter && filterMatch) || (!filter && isRecent)) {
        acc[0].workspaces.push(workspace);
      }
      if (filterMatch && !filter) {
        acc[1].workspaces.push(workspace);
      }

      return acc;
    }, baseGroups);
  }, [workspaces, filter, recentWorkspaces]);

  const onSelectChange = (newWorkspace: Workspace) => {
    // Only the "Create Workspace" button should reach this case
    if (!newWorkspace || !newWorkspace.name) {
      if (!newWorkspace) {
        // Open modal for new workspace
        openModal();
      } else {
        logger.error(`Workspace selected without name: ${JSON.stringify(newWorkspace)}`);
      }
      return;
    }

    const newworkspace = newWorkspace.name;
    const segments = location.pathname.split('/');
    // Segments should be: empty string, 'workspaces', workspace name, subpaths.
    // In most cases we only want the first 3, because after the 3rd segment
    // it tends to be resources that are specific to that workspace, but there
    // are a small handful of exceptions where we want the 4th segment, for
    // detail views or creating a new instance of something.
    if (segments[1] === 'workspaces' && segments[2] === activeWorkspaceName) {
      segments[2] = newworkspace;

      let segmentLength = 4;
      if (URL_SEGMENT_ALLOWLIST.has(segments[4])) {
        segmentLength = 5;
      }
      const newPath = segments.slice(0, segmentLength).join('/');
      navigate(newPath);
    } else {
      navigate(getWorkspaceDetailsDefaultRoute(newworkspace));
    }

    addRecentWorkspace(newworkspace);
    onValueChange?.(newworkspace);
  };

  if (!activeWorkspaceName) {
    return null;
  }

  if (!mostRecentWorkspace && !workspaces?.length) {
    return <Skeleton className="w-3xs" data-testid={dataTestIds.skeleton} />;
  }

  const activeWorkspace = workspaces
    ? workspaces.find((workspace) => {
        return workspace.name === activeWorkspaceName;
      })
    : undefined;

  const workspaceDisplayName = activeWorkspace?.name || mostRecentWorkspace;

  if (activeWorkspace && !mostRecentWorkspace) {
    const workspace = activeWorkspace.name;
    addRecentWorkspace(workspace);
  }

  const emptyFilterResults =
    groups.find((group) => group.header === 'Search results')?.workspaces.length === 0;

  return (
    <>
      <DropdownRoot
        onOpenChange={(open) => {
          setOpen(open);
          if (open) {
            // Workspaces can be created outside this dropdown (CLI, other tabs, the assistant)
            void refetch();
            // Preload so "New Workspace" opens without waiting on the chunk
            loadWorkspaceCreateModal().catch(() => {
              // The lazy component re-imports on render and surfaces any error there
            });
          } else {
            setFilter('');
          }
        }}
        size="small"
      >
        <DropdownTrigger
          className="max-h-8 max-w-3xs pointer-events-auto -mx-1" // Pointer events are otherwise disabled when the last item in breadcrumbs
          disabled={isError}
          aria-label="Select workspace"
        >
          {/* leading-[normal] fixes the line-height cutting off text in KUI */}
          <Text className="text-ellipsis overflow-hidden leading-[normal]">
            {workspaceDisplayName}
          </Text>
        </DropdownTrigger>
        <DropdownContent className="overflow-x-hidden w-3xs">
          <>
            <Block className="p-2 w-full">
              <TextInput
                disabled={isPending || isError}
                name="workspace-filter"
                tabIndex={0}
                className="overflow-hidden"
                slotStart={<Filter />}
                placeholder="Search workspaces"
                onChange={(e: ChangeEvent<HTMLInputElement>) => {
                  setFilter(e.target.value);
                }}
                // Prevents focusing dropdown items while typing
                onKeyDownCapture={(e) => {
                  e.stopPropagation();
                }}
                onKeyUpCapture={(e) => {
                  e.stopPropagation();
                }}
              />
            </Block>
            <div className="max-h-[45vh] overflow-y-auto w-3xs">
              {isPending && (
                <DropdownItem disabled>
                  <Spinner aria-label="Loading workspaces" size="small" />
                </DropdownItem>
              )}
              {groups.map((group, i) => {
                return (
                  <Stack key={group.header}>
                    <DropdownHeading className={cn(i !== 0 && 'border-base border-t')}>
                      {group.header}
                    </DropdownHeading>
                    {emptyFilterResults ? (
                      <Flex className="p-3" gap="density-md">
                        <Text>No workspaces found for "{filter}"</Text>
                      </Flex>
                    ) : (
                      <DropdownSection>
                        {group.workspaces.map((workspace) => {
                          return (
                            <DropdownItem
                              key={workspace.name}
                              onSelect={() => onSelectChange(workspace)}
                            >
                              {workspace.name}
                            </DropdownItem>
                          );
                        })}
                      </DropdownSection>
                    )}
                  </Stack>
                );
              })}
            </div>
            <div className="border-base border-t flex items-center w-full">
              <DropdownItem onSelect={() => startTransition(openModal)}>
                <Flex gap="density-md">
                  <Plus /> New Workspace
                </Flex>
              </DropdownItem>
            </div>
          </>
        </DropdownContent>
      </DropdownRoot>

      {/* Scoped boundary so loading the modal doesn't fall back to RootLayout's full-page loader.
          Kept mounted so the transition in onSelect waits for the module instead of committing
          the fallback, which React throttles before revealing the modal. */}
      <Suspense fallback={null}>
        {isModalOpen && (
          <WorkspaceCreateModal
            open={isModalOpen}
            onClose={closeModal}
            onCreate={(workspace) => addRecentWorkspace(workspace.name)}
          />
        )}
      </Suspense>
    </>
  );
};
