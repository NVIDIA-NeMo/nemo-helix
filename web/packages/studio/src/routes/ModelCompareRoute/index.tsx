// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelEntity } from '@nemo/common/src/api/models/useModelEntity';
import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { hasModelProvider } from '@nemo/common/src/utils/models';
import { PageHeader, Tabs } from '@nvidia/foundations-react-core';
import { ChatEmptyState } from '@studio/components/chat/ChatEmptyState';
import { ComparePlayground } from '@studio/components/ModelCompareChat/ComparePlayground';
import { ModelComparePrompts } from '@studio/components/ModelComparePrompts';
import { useCompareModels } from '@studio/hooks/useCompareModels';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { type FC, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router';

type CompareView = 'compare' | 'prompts';

export const ModelCompareRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const availableModels = useModelSearch({ workspace, include: hasModelProvider });
  const hasNoModels =
    !availableModels.loading &&
    !availableModels.error &&
    !availableModels.hasMore &&
    availableModels.models.length === 0;

  const [searchParams] = useSearchParams();
  const [activeView, setActiveView] = useState<CompareView>('compare');
  const { models, setModels, addModel, removeModel, setModel } = useCompareModels([
    { modelURN: null },
    { modelURN: null },
  ]);
  const preselectedUrnRef = useRef<string | null>(null);

  // Preselect panel 0 from ?model=, which carries either a full URN or a bare name in this
  // workspace. Resolved with a single lookup rather than by scanning the whole catalogue.
  const modelParam = searchParams.get('model');
  const preselectUrn =
    modelParam && workspace
      ? modelParam.includes('/')
        ? modelParam
        : `${workspace}/${modelParam}`
      : null;
  const preselectModel = useModelEntity(preselectUrn, {
    enabled: !!preselectUrn && preselectedUrnRef.current !== preselectUrn,
  });

  useEffect(() => {
    if (!preselectUrn || preselectedUrnRef.current === preselectUrn) return;
    if (!preselectModel) return;
    if (preselectModel.workspace === workspace && hasModelProvider(preselectModel)) {
      setModels((prev) =>
        prev.map((m, i) =>
          i === 0 ? { ...m, modelURN: `${preselectModel.workspace}/${preselectModel.name}` } : m
        )
      );
    }
    preselectedUrnRef.current = preselectUrn;
  }, [preselectModel, preselectUrn, workspace, setModels]);

  // TODO: re-add Run Evaluation button once the RunEvaluationModal flow is ready

  if (availableModels.error) {
    return <ErrorMessage header="Couldn't load models" message={availableModels.error.message} />;
  }

  // Empty state when the workspace has no servable models and we're not still loading.
  if (hasNoModels) {
    return <ChatEmptyState hasModels={false} />;
  }

  const showChatPanels = activeView !== 'prompts';

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 px-6 pt-6 pb-4">
        <PageHeader className="p-0 pb-4" slotHeading="Playground" />
        <Tabs
          className="min-w-0 overflow-visible"
          value={activeView}
          onValueChange={(value) => setActiveView(value as CompareView)}
          items={[
            { value: 'compare', children: 'Compare' },
            { value: 'prompts', children: 'Run Prompts' },
          ]}
        />
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">
        <div className={`h-full ${showChatPanels ? '' : 'hidden'}`}>
          <ComparePlayground
            workspace={workspace}
            models={models}
            onRemoveModel={removeModel}
            onSetModel={setModel}
            onAddModel={addModel}
          />
        </div>
        <div className={`h-full overflow-hidden ${activeView !== 'prompts' ? 'hidden' : ''}`}>
          <ModelComparePrompts
            workspace={workspace}
            models={models}
            onRemoveModel={removeModel}
            onSetModel={setModel}
            onAddModel={addModel}
          />
        </div>
      </div>
    </div>
  );
};
