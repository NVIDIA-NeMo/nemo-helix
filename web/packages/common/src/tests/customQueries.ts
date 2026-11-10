// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  queryHelpers,
  buildQueries,
  type Matcher,
  type MatcherOptions,
} from '@testing-library/react';

// The queryAllByAttribute is a shortcut for attribute-based matchers
// You can also use document.querySelector or a combination of existing
// testing library utilities to find matching nodes for your query
const queryAllByIconName = (
  container: HTMLElement,
  id: Matcher,
  options?: MatcherOptions | undefined
) => queryHelpers.queryAllByAttribute('data-icon-name', container, id, options);

const getMultipleError = (_c: Element | null, value: string) =>
  `Found multiple elements with the data-icon-name attribute of: ${value}`;
const getMissingError = (_c: Element | null, value: string) =>
  `Unable to find an element with the data-icon-name attribute of: ${value}`;

const [queryByIconName, getAllByIconName, getByIconName, findAllByIconName, findByIconName] =
  buildQueries(queryAllByIconName, getMultipleError, getMissingError);

export {
  queryByIconName,
  queryAllByIconName,
  getByIconName,
  getAllByIconName,
  findAllByIconName,
  findByIconName,
};
