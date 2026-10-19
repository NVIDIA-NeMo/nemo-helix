// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledCheckbox } from '@nemo/common/src/components/form/ControlledCheckbox';
import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { Banner, Button, Stack } from '@nvidia/foundations-react-core';
import type { DeployOnCreate } from '@studio/api/agents/useDeployOnCreate';
import { DeploymentModeSelect } from '@studio/components/DeploymentModeSelect';
import { ImageBuildFirstNotice } from '@studio/components/ImageBuildFirstNotice';
import { type FC, useEffect } from 'react';
import { useController } from 'react-hook-form';

type ControllerProps = UseControllerComponentProps['useControllerProps'];

interface DeployOnCreateFieldsProps {
  deployOnCreate: DeployOnCreate;
  /** The form's `deploy` field. */
  deployControllerProps: ControllerProps;
  /** The form's `deploymentMode` field. */
  modeControllerProps: ControllerProps;
}

/** The "Deploy after creating" checkbox and runtime picker for a flow that creates an agent. */
export const DeployOnCreateFields: FC<DeployOnCreateFieldsProps> = ({
  deployOnCreate: {
    modes,
    isLoading,
    canDeploy,
    needsImageBuild,
    discoveryError,
    isRetrying,
    retry,
  },
  deployControllerProps,
  modeControllerProps,
}) => {
  const {
    field: { value: deploy },
  } = useController(deployControllerProps);
  const {
    field: { value: mode, onChange: setMode },
  } = useController(modeControllerProps);

  // The platform can turn the default runtime off, so the first one it offers stands in.
  useEffect(() => {
    const firstMode = modes.at(0);
    if (firstMode && !modes.includes(mode)) setMode(firstMode);
  }, [modes, mode, setMode]);

  if (!canDeploy && !isLoading && !discoveryError) return null;

  return (
    <Stack gap="density-md">
      <ControlledCheckbox
        useControllerProps={deployControllerProps}
        slotLabel="Deploy after creating"
      />
      {deploy && modes.length > 1 ? (
        <DeploymentModeSelect
          useControllerProps={modeControllerProps}
          modes={modes}
          loading={isLoading}
        />
      ) : null}
      {deploy && needsImageBuild(mode) ? <ImageBuildFirstNotice mode={mode} /> : null}
      {deploy && discoveryError ? (
        <Banner
          kind="inline"
          status="error"
          slotActions={
            <Button kind="secondary" size="small" disabled={isRetrying} onClick={retry}>
              {isRetrying ? 'Retrying…' : 'Retry'}
            </Button>
          }
        >
          Could not load the available runtimes: {discoveryError}. Retry, or uncheck &quot;Deploy
          after creating&quot; to create the agent without deploying it.
        </Banner>
      ) : null}
    </Stack>
  );
};
