// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { UploadModal } from '@nemo/common/src/components/UploadModal';
import { Button, Flex, Modal, Text, Tooltip } from '@nvidia/foundations-react-core';
import { Add } from '@nvidia/react-gui-icons/icons';
import { StatsBadge } from '@studio/components/chat/StatsBadge';
import { ModelCompareTable } from '@studio/components/ModelComparePrompts/ModelCompareTable';
import type { ModelComparePromptsProps } from '@studio/components/ModelComparePrompts/types';
import { useModelComparePrompts } from '@studio/components/ModelComparePrompts/useModelComparePrompts';
import { type FC } from 'react';

export const ModelComparePrompts: FC<ModelComparePromptsProps> = ({
  workspace,
  models,
  onRemoveModel,
  onSetModel,
  onReadyChange,
  agentName,
  onAddModel,
}) => {
  const {
    fileResult,
    promptRows,
    isRunning,
    sampleSize,
    setSampleSize,
    sampleMethod,
    setSampleMethod,
    expandedCell,
    setExpandedCell,
    pickerValue,
    pickerSelectKey,
    parseError,
    isFilesetPickerOpen,
    setIsFilesetPickerOpen,
    promptKeyAutoDetected,
    rowCount,
    handlePromptKeyChange,
    clearResponses,
    runInference,
    cancelRun,
    hasAssignedModel,
    hasPrompts,
    averagesByModelId,
    anyAverages,
    handleDatasetSelect,
    handleFilesetPickerSubmit,
    datasetItems,
  } = useModelComparePrompts({ workspace, models, onReadyChange, agentName });

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden px-6 py-2">
      {/* Results table fills remaining height; this is the main vertical scroll region. */}
      <div className="flex min-h-0 flex-1">
        <div className="max-h-full flex-1 self-start overflow-auto rounded-lg border border-base bg-surface-raised">
          <ModelCompareTable
            models={models}
            workspace={workspace}
            promptRows={promptRows}
            fileResult={fileResult}
            sampleMethod={sampleMethod}
            setSampleMethod={setSampleMethod}
            sampleSize={sampleSize}
            setSampleSize={setSampleSize}
            rowCount={rowCount}
            isRunning={isRunning}
            hasPrompts={hasPrompts}
            hasAssignedModel={hasAssignedModel}
            promptKeyAutoDetected={promptKeyAutoDetected}
            pickerSelectKey={pickerSelectKey}
            pickerValue={pickerValue}
            datasetItems={datasetItems}
            parseError={parseError}
            averagesByModelId={averagesByModelId}
            anyAverages={anyAverages}
            onRemoveModel={onRemoveModel}
            onSetModel={onSetModel}
            clearResponses={clearResponses}
            runInference={runInference}
            cancelRun={cancelRun}
            handleDatasetSelect={handleDatasetSelect}
            handlePromptKeyChange={handlePromptKeyChange}
            setExpandedCell={setExpandedCell}
          />
        </div>
        {onAddModel && (
          <div className="flex shrink-0 self-start pl-1">
            <Tooltip slotContent="Add model">
              <button
                onClick={onAddModel}
                className="flex cursor-pointer items-center justify-center rounded border border-base bg-surface-raised p-1.5 text-fg-subdued transition-colors hover:bg-surface-sunken hover:text-fg-base"
                aria-label="Add model"
              >
                <Add size="16" />
              </button>
            </Tooltip>
          </div>
        )}
      </div>

      <UploadModal
        workspace={workspace}
        open={isFilesetPickerOpen}
        includeDataset
        allowNewDataset
        title="Select File"
        submitButtonText="Select File"
        onClose={() => setIsFilesetPickerOpen(false)}
        onSubmit={handleFilesetPickerSubmit}
      />

      <Modal
        open={expandedCell !== null}
        onOpenChange={(open) => {
          if (!open) setExpandedCell(null);
        }}
        slotHeading={expandedCell?.title ?? 'Cell Content'}
        className="w-[90vw] max-w-[1000px]"
        slotFooter={
          <Flex justify="between" align="center" className="w-full">
            {expandedCell?.stats ? <StatsBadge metrics={expandedCell.stats} emphasis /> : <span />}
            <Button kind="tertiary" onClick={() => setExpandedCell(null)}>
              Close
            </Button>
          </Flex>
        }
      >
        <div className="max-h-[70vh] overflow-auto">
          <Text kind="body/regular/md" className="whitespace-pre-wrap">
            {expandedCell?.content}
          </Text>
        </div>
      </Modal>
    </div>
  );
};
