// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableExpandableCell } from '@nemo/common/src/components/DataView/TableExpandableCell';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import type { FileSampleMethod } from '@nemo/common/src/utils/sampleTextLines';
import { Button, Flex, Select, Text } from '@nvidia/foundations-react-core';
import { Trash } from '@nvidia/react-gui-icons/icons';
import { StatsBadge } from '@studio/components/chat/StatsBadge';
import { FileSamplingMethodSelect } from '@studio/components/FileSamplingSnippet/FileSamplingMethodSelect';
import { ModelColumnSelect } from '@studio/components/ModelComparePrompts/ModelColumnSelect';
import type {
  ExpandedCellState,
  PromptRow,
  ResponseStats,
} from '@studio/components/ModelComparePrompts/types';
import {
  PANEL_ROLE_COLORS,
  PANEL_ROLE_DOT_CLASS,
  PANEL_ROLE_LABELS,
  type SharedModelEntry,
} from '@studio/routes/ModelCompareRoute/types';
import type { DatasetInputFileResult } from '@studio/types/dataset';
import type { Dispatch, FC, SetStateAction } from 'react';

interface ModelCompareTableProps {
  models: SharedModelEntry[];
  workspace: string;
  promptRows: PromptRow[];
  fileResult: DatasetInputFileResult | null;
  sampleMethod: FileSampleMethod;
  setSampleMethod: Dispatch<SetStateAction<FileSampleMethod>>;
  sampleSize: number;
  setSampleSize: Dispatch<SetStateAction<number>>;
  rowCount: number;
  isRunning: boolean;
  hasPrompts: boolean;
  hasAssignedModel: boolean;
  promptKeyAutoDetected: boolean;
  pickerSelectKey: number;
  pickerValue: string | undefined;
  datasetItems: { value: string; children: string }[];
  parseError: string | null;
  averagesByModelId: Record<number, (ResponseStats & { count: number }) | null>;
  anyAverages: boolean;
  onRemoveModel: (id: number) => void;
  onSetModel: (id: number, modelURN: string | null) => void;
  clearResponses: (columnId?: number) => void;
  runInference: () => void;
  cancelRun: () => void;
  handleDatasetSelect: (value: string) => void;
  handlePromptKeyChange: (key: string) => void;
  setExpandedCell: Dispatch<SetStateAction<ExpandedCellState | null>>;
}

export const ModelCompareTable: FC<ModelCompareTableProps> = ({
  models,
  workspace,
  promptRows,
  fileResult,
  sampleMethod,
  setSampleMethod,
  sampleSize,
  setSampleSize,
  rowCount,
  isRunning,
  hasPrompts,
  hasAssignedModel,
  promptKeyAutoDetected,
  pickerSelectKey,
  pickerValue,
  datasetItems,
  parseError,
  averagesByModelId,
  anyAverages,
  onRemoveModel,
  onSetModel,
  clearResponses,
  runInference,
  cancelRun,
  handleDatasetSelect,
  handlePromptKeyChange,
  setExpandedCell,
}) => {
  return (
    <table className="min-w-full table-fixed border-separate border-spacing-0">
      <colgroup>
        <col className="w-[320px] min-w-[280px]" />
        {models.map((m) => (
          <col key={m.id} className="w-[320px] min-w-[280px]" />
        ))}
      </colgroup>
      <thead className="sticky top-0 z-10 bg-surface-raised">
        {/* Row 1: sampling controls + role labels */}
        <tr>
          <th className="border-b border-r border-base px-3 py-2 text-left align-middle">
            <Flex align="center" justify="between" gap="density-sm">
              <Text kind="label/bold/md" className="shrink-0">
                Prompts
              </Text>
              <FileSamplingMethodSelect
                value={sampleMethod}
                onValueChange={setSampleMethod}
                size="medium"
                rowCountGroup={{
                  value: sampleSize,
                  onValueChange: setSampleSize,
                  maxRows: Math.max(1, rowCount),
                  disabled: isRunning || rowCount === 0,
                }}
                attributes={{ select: { disabled: isRunning || rowCount === 0 } }}
              />
              {isRunning ? (
                <Button kind="primary" color="danger" onClick={cancelRun}>
                  Stop
                </Button>
              ) : (
                <Button
                  kind="primary"
                  color="brand"
                  onClick={runInference}
                  disabled={!hasPrompts || !hasAssignedModel}
                >
                  Run
                </Button>
              )}
            </Flex>
          </th>
          {models.map((m, idx) => {
            const roleColor = PANEL_ROLE_COLORS[Math.min(idx, PANEL_ROLE_COLORS.length - 1)];
            const colBorder = idx < models.length - 1 ? 'border-r ' : '';
            return (
              <th key={m.id} className={`border-b ${colBorder}border-base px-3 py-2 align-middle`}>
                <Flex align="center" justify="between">
                  <Flex align="center" gap="density-xs">
                    <span
                      className={`h-2 w-2 shrink-0 rounded-full ${PANEL_ROLE_DOT_CLASS[roleColor]}`}
                    />
                    <Text kind="label/bold/md">{PANEL_ROLE_LABELS[roleColor]}</Text>
                  </Flex>
                  <button
                    onClick={() => onRemoveModel(m.id)}
                    disabled={isRunning}
                    className="cursor-pointer rounded p-1 text-fg-subdued hover:bg-surface-sunken hover:text-fg-base"
                    aria-label="Remove model column"
                  >
                    <Trash size="14" />
                  </button>
                </Flex>
              </th>
            );
          })}
        </tr>
        {/* Row 2: dataset picker + model selects */}
        <tr>
          <th
            className={`${hasPrompts ? 'border-b ' : ''}border-r border-base px-3 py-2 align-top`}
          >
            <Select
              // Remount after the picker sentinel is chosen so its internal
              // selection resets to the real value — otherwise re-clicking
              // "Select from dataset file..." is a no-op (already selected).
              key={pickerSelectKey}
              items={datasetItems}
              value={pickerValue}
              onValueChange={handleDatasetSelect}
              placeholder="Select prompts"
              disabled={isRunning}
              className="w-full"
            />
            {parseError && (
              <Text kind="label/regular/sm" className="mt-1 text-fg-error">
                {parseError}
              </Text>
            )}
            {fileResult && !promptKeyAutoDetected && fileResult.availableKeys.length > 0 && (
              <Flex align="center" gap="density-sm" className="mt-2">
                <Text kind="label/regular/sm" className="shrink-0 text-fg-subdued">
                  Prompt Field
                </Text>
                <Select
                  items={fileResult.availableKeys.map((k) => ({
                    value: k.value,
                    children: k.label,
                  }))}
                  value={fileResult.keyMapping.promptKey ?? undefined}
                  onValueChange={handlePromptKeyChange}
                  placeholder="Select a field"
                  disabled={isRunning}
                  size="small"
                  className="w-full"
                />
              </Flex>
            )}
          </th>
          {models.map((m, idx) => (
            <th
              key={m.id}
              className={`${hasPrompts ? 'border-b ' : ''}${idx < models.length - 1 ? 'border-r ' : ''}border-base px-2 py-2 align-top`}
            >
              <ModelColumnSelect
                workspace={workspace}
                value={m.modelURN}
                disabled={isRunning}
                onChange={(ref) => {
                  onSetModel(m.id, ref || null);
                  clearResponses(m.id);
                }}
              />
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {promptRows.map((row, rowIdx) => {
          const rowBottom = rowIdx < promptRows.length - 1 || anyAverages ? 'border-b ' : '';
          return (
            <tr key={row.sourceIndex} className="bg-surface-raised">
              <td className={`${rowBottom}border-r border-base p-0 align-top`}>
                <TableExpandableCell
                  content={row.prompt}
                  title={`Prompt (dataset row ${row.sourceIndex})`}
                  onExpand={setExpandedCell}
                  boldContent
                />
              </td>
              {models.map((m, idx) => {
                const response = row.responses[m.id];
                const modelName = m.modelURN ? getPartsFromReference(m.modelURN).name : 'Model';
                const colBorder = idx < models.length - 1 ? 'border-r ' : '';
                if (response === undefined) {
                  return (
                    <td
                      key={m.id}
                      className={`${rowBottom}${colBorder}border-base px-3 py-2 align-top`}
                    >
                      <Text kind="body/regular/md" className="text-fg-subdued">
                        -
                      </Text>
                    </td>
                  );
                }
                if (response === null) {
                  return (
                    <td
                      key={m.id}
                      className={`${rowBottom}${colBorder}border-base px-3 py-2 align-top`}
                    >
                      <Text kind="body/regular/md" className="text-fg-error">
                        Error
                      </Text>
                    </td>
                  );
                }
                return (
                  <td key={m.id} className={`${rowBottom}${colBorder}border-base p-0 align-top`}>
                    <TableExpandableCell
                      content={response.text}
                      title={`${modelName} response (dataset row ${row.sourceIndex})`}
                      onExpand={(state) => setExpandedCell({ ...state, stats: response.stats })}
                      footer={<StatsBadge metrics={response.stats} className="px-3 pb-2" />}
                    />
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
      {hasPrompts && anyAverages && (
        <tfoot className="sticky bottom-0 z-10 bg-surface-raised">
          <tr>
            <td className="border-t-2 border-r border-base px-3 py-2 align-middle">
              <Text kind="label/bold/md">Average</Text>
            </td>
            {models.map((m, idx) => {
              const avg = averagesByModelId[m.id];
              return (
                <td
                  key={m.id}
                  className={`border-t-2 ${idx < models.length - 1 ? 'border-r ' : ''}border-base px-3 py-2 align-middle`}
                >
                  {avg ? (
                    <StatsBadge metrics={avg} emphasis tone="brand" />
                  ) : (
                    <Text kind="body/regular/md" className="text-fg-subdued">
                      —
                    </Text>
                  )}
                </td>
              );
            })}
          </tr>
        </tfoot>
      )}
    </table>
  );
};
