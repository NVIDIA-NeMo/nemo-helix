// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { getDeployAction } from '@studio/components/dataViews/CustomModelsDataView/deployAction';

const model = { id: 'm', name: 'my-model', workspace: 'default' } as ModelEntity;

describe('getDeployAction', () => {
  it('offers Deploy when nothing serves the model', () => {
    const action = getDeployAction({ kind: 'not-deployed' }, model);
    expect(action?.label).toBe('Deploy');
    expect(action?.href).toBe('/workspaces/default/deployments/~new?model=default%2Fmy-model');
  });

  it('labels an adapter row "Deploy" too, pointed at the base model that gets deployed', () => {
    const action = getDeployAction({ kind: 'adapter-not-loaded' }, model);
    expect(action?.label).toBe('Deploy');
    // Targets the Create Deployment wizard route, not the deployments list.
    expect(action?.href).toBe('/workspaces/default/deployments/~new?model=default%2Fmy-model');
  });

  it.each(['ERROR', 'DELETED', 'LOST'] as const)(
    'offers Deploy when the deployment is %s, since nothing is serving it',
    (status) => {
      const action = getDeployAction({ kind: 'served', hasDeployment: true, status }, model);
      expect(action?.label).toBe('Deploy');
    }
  );

  it.each(['READY', 'PENDING', 'CREATED', 'DELETING'] as const)(
    'offers nothing when the deployment is %s',
    (status) => {
      expect(getDeployAction({ kind: 'served', hasDeployment: true, status }, model)).toBeNull();
    }
  );

  it('offers nothing for a model reachable through an external provider', () => {
    expect(
      getDeployAction({ kind: 'served', hasDeployment: false, providerRef: 'default/build' }, model)
    ).toBeNull();
  });

  it('offers nothing when status is unknown, to avoid duplicating a live deployment', () => {
    expect(getDeployAction({ kind: 'unknown' }, model)).toBeNull();
  });

  it('offers nothing while the row is still resolving', () => {
    expect(getDeployAction({ kind: 'loading' }, model)).toBeNull();
    expect(getDeployAction(undefined, model)).toBeNull();
  });
});
