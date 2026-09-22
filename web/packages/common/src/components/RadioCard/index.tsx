/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import { isDefined } from '@nemo/common/src/utils/isDefined';
import { Card, Flex, RadioGroupInput, RadioGroupItem, Text } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { ComponentProps, FC, ReactNode } from 'react';

export interface RadioCardProps extends Omit<ComponentProps<typeof RadioGroupItem>, 'children'> {
  /** The radio value (forwarded to RadioGroupInput) */
  value: string;
  /** Primary label (e.g. "Radio Label") */
  label: ReactNode;
  /** Optional secondary description text */
  description?: ReactNode;
  /** Rendered at the end of the label row, pushed right — a status or metadata Badge. */
  slotEnd?: ReactNode;
  /** Optional icon or element shown between the radio indicator and the label */
  icon?: ReactNode;
  /** Id for the label element (used for aria-labelledby). Defaults to `${value}-label` */
  labelId?: string;
  /** Which side of the card the radio input renders on. @default "right" */
  labelSide?: 'left' | 'right';
  /** Shows the radio indicator dot; when false the input is only visually hidden and the selected border conveys state. @default true */
  showIndicator?: boolean;
  /** When true, shows the card as selected. When used inside RadioGroupRoot, the group's value controls this; pass checked so the card reflects the active state (e.g. checked={value === 'this-option'}). */
  checked?: boolean;
  /** Whether the underlying radio input is disabled */
  disabled?: boolean;
  /** Additional attributes to pass to the Panel or RadioGroupItem components */
  attributes?: {
    Card?: Partial<ComponentProps<typeof Card>>;
    RadioGroupItem?: Partial<ComponentProps<typeof RadioGroupItem>>;
    RadioGroupInput?: Partial<ComponentProps<typeof RadioGroupInput>>;
  };
}

/**
 * A card-style radio option with optional icon, primary label, and description.
 * Use inside RadioGroupRoot for single-selection from a list of options.
 *
 * @example
 * <RadioGroupRoot value={value} onValueChange={setValue}>
 *   <RadioCard value="a" label="Option A" description="First option" icon={<Cube />} />
 *   <RadioCard value="b" label="Option B" description="Second option" />
 * </RadioGroupRoot>
 */
export const RadioCard: FC<RadioCardProps> = ({
  label,
  description,
  icon,
  slotEnd,
  value,
  labelId,
  labelSide = 'right',
  showIndicator = true,
  checked,
  disabled,
  className,
  attributes = {},
}) => {
  const id = labelId ?? `${String(value).replace(/\s+/g, '-')}-label`;
  const hasDescription = isDefined(description);

  const textStartClass =
    'text-left ' + (showIndicator && labelSide === 'right' ? 'col-start-2' : 'col-start-1');
  const labelClass = `${textStartClass} row-start-1`;
  const descriptionClass = `${textStartClass} row-start-2`;

  // The hidden input is absolutely positioned, so it leaves grid flow entirely.
  const colClass = !showIndicator
    ? '[&_.nv-card-content]:grid-cols-1'
    : labelSide === 'right'
      ? '[&_.nv-card-content]:grid-cols-[auto_1fr]'
      : '[&_.nv-card-content]:grid-cols-[1fr_auto]';
  const inputClass = !showIndicator
    ? ''
    : labelSide === 'right'
      ? '[&_.nv-radio-group-input]:col-start-1'
      : '[&_.nv-radio-group-input]:col-start-2';
  const gapClass = hasDescription
    ? '[&_.nv-card-content]:gap-2!'
    : '[&_.nv-card-content]:gap-0! [&_.nv-card-content]:gap-x-2!';
  const nvPanelContentClass = `[&_.nv-card-content]:grid ${colClass} [&_.nv-card-content]:grid-rows-[auto_auto] [&_.nv-card-content]:items-center! [&_.nv-card-content]:w-full ${inputClass} ${gapClass} ${hasDescription ? '[&_.nv-card-content]:row-gap-2' : ''}`;

  return (
    <RadioGroupItem
      aria-labelledby={id}
      {...attributes?.RadioGroupItem}
      className={cn('group block w-full items-start', attributes?.RadioGroupItem?.className)}
    >
      <Card
        className={cn(
          'cursor-pointer [&_*]:cursor-pointer',
          'hover:bg-interaction-hover',
          // KUI's RadioGroupItem does not emit data-state/data-disabled, so key
          // these off the real input via :has().
          'has-[.nv-radio-group-input:checked]:border-interaction-selected',
          checked === true && 'border-interaction-selected',
          'has-[.nv-radio-group-input:disabled]:pointer-events-none has-[.nv-radio-group-input:disabled]:opacity-50',
          nvPanelContentClass,
          className
        )}
        {...attributes?.Card}
      >
        <RadioGroupInput
          value={value}
          disabled={disabled}
          showIndicator={showIndicator}
          aria-labelledby={id}
          {...attributes?.RadioGroupInput}
        />
        {/* Single column for label + description (separate from the radio indicator column) */}
        <Flex direction="col" gap="density-sm" className={labelClass}>
          <Flex
            gap="density-md"
            align="center"
            className={cn('min-h-0', slotEnd != null && 'w-full')}
          >
            {icon != null && (
              <Flex align="center" className="shrink-0 text-base-foreground" aria-hidden>
                {icon}
              </Flex>
            )}
            <Text kind="body/bold/lg" id={id}>
              {label}
            </Text>
            {slotEnd != null && <div className="ml-auto shrink-0">{slotEnd}</div>}
          </Flex>
        </Flex>
        {hasDescription && (
          <Text kind="body/regular/md" color="secondary" className={descriptionClass}>
            {description}
          </Text>
        )}
      </Card>
    </RadioGroupItem>
  );
};
