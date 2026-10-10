// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { WorkspaceModelSelect } from '@nemo/common/src/components/ModelSelectV2';
import {
  Button,
  Card,
  Checkbox,
  Divider,
  Flex,
  FormField,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  ROUTING_STRATEGIES,
  type RoutingStrategyId,
  type RoutingStrategyOption,
  type SwitchyardFormValues,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/formValues';
import cn from 'classnames';
import { type FC } from 'react';
import { useController, useFormContext } from 'react-hook-form';

/** The plugin's own defaults, so leaving a knob alone submits what the CLI would. */
export const DEFAULT_THRESHOLD = 0.5;

interface SectionProps {
  workspace: string;
  disabled?: boolean;
}

/** Empty submits no `judge_model`, so the plugin falls back to each pair's capable model. */
const JudgeModelField: FC<SectionProps> = ({ workspace, disabled }) => {
  const {
    field: { value, onChange },
  } = useController<SwitchyardFormValues, 'judgeModel'>({ name: 'judgeModel' });

  return (
    <FormField
      slotLabel="Judge model (optional)"
      slotHelp="Scores each request. Leave unset to use each pair's capable model."
    >
      <Flex gap="density-sm" align="center" className="w-full">
        <WorkspaceModelSelect
          workspace={workspace}
          value={value ? { model: value } : null}
          onValueChange={({ model }) => onChange(model)}
          placeholder="Each pair's capable model"
          disabled={disabled}
          hideAdapters
          fullWidth
          aria-label="Judge model"
        />
        {value && (
          <Button kind="tertiary" disabled={disabled} onClick={() => onChange('')}>
            Clear
          </Button>
        )}
      </Flex>
    </FormField>
  );
};

const StrategySettings: FC<SectionProps & { option: RoutingStrategyOption }> = ({
  option,
  workspace,
  disabled,
}) => {
  const {
    control,
    formState: { errors },
  } = useFormContext<SwitchyardFormValues>();
  const { field } = option.threshold;

  return (
    <Stack gap="density-md" className="w-full">
      <ControlledSliderWithTextInput
        useControllerProps={{ control, name: field }}
        id={`switchyard-${field}`}
        defaultValue={DEFAULT_THRESHOLD}
        min={0}
        max={1}
        step={0.05}
        disabled={disabled}
        displayName={option.threshold.label}
        formFieldProps={{
          slotLabel: option.threshold.label,
          slotHelp: option.threshold.help,
          slotError: errors[field]?.message,
        }}
      />
      {option.id === 'llm_classifier' && (
        <JudgeModelField workspace={workspace} disabled={disabled} />
      )}
    </Stack>
  );
};

/**
 * The routing strategies to build, one card each. A strategy's settings open inside its card once
 * it is picked, so every knob sits beside the choice it tunes.
 */
export const RoutingStrategiesSection: FC<SectionProps> = ({ workspace, disabled }) => {
  const {
    field: { value, onChange },
  } = useController<SwitchyardFormValues, 'routingStrategies'>({ name: 'routingStrategies' });

  // Kept in catalogue order so the submitted list does not depend on click order.
  const toggle = (id: RoutingStrategyId, checked: boolean) =>
    onChange(
      ROUTING_STRATEGIES.map((option) => option.id).filter((other) =>
        other === id ? checked : value.includes(other)
      )
    );

  return (
    <Stack gap="density-md" className="w-full">
      {ROUTING_STRATEGIES.map((option) => {
        const checked = value.includes(option.id);
        return (
          <Card key={option.id} className={cn('w-full', checked && 'border-interaction-selected')}>
            <Stack gap="density-md" className="w-full">
              <Checkbox
                disabled={disabled}
                checked={checked}
                onCheckedChange={(next) => toggle(option.id, next === true)}
                slotLabel={
                  <Stack gap="density-xxs">
                    <Text kind="body/bold/md">{option.title}</Text>
                    <Text kind="body/regular/sm" color="secondary">
                      {option.description}
                    </Text>
                  </Stack>
                }
              />
              {checked && (
                <>
                  <Divider />
                  <StrategySettings option={option} workspace={workspace} disabled={disabled} />
                </>
              )}
            </Stack>
          </Card>
        );
      })}
    </Stack>
  );
};
