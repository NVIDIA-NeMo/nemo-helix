// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  notifEventToInboxItem,
  parseNotifEvent,
} from '@studio/prototypes/notifications/notif/utils';

const frame = (overrides: Record<string, unknown> = {}): string =>
  JSON.stringify({
    id: 'e1',
    type: 'jobs.job.failed',
    workspace: 'default',
    title: 'Fine-tune failed',
    message: 'CUDA out of memory',
    severity: 'error',
    link: '/workspaces/default/jobs/abc',
    data: {},
    created_at: '2026-11-18T20:33:23.123456Z',
    created_by: 'alice',
    ...overrides,
  });

describe('parseNotifEvent', () => {
  it('parses a valid frame', () => {
    expect(parseNotifEvent(frame())?.id).toBe('e1');
  });

  it('rejects malformed JSON, missing fields and unknown severities', () => {
    expect(parseNotifEvent('not json')).toBeUndefined();
    expect(parseNotifEvent(frame({ title: undefined }))).toBeUndefined();
    expect(parseNotifEvent(frame({ severity: 'critical' }))).toBeUndefined();
  });

  it('defaults a missing data payload to an empty object', () => {
    expect(parseNotifEvent(frame({ data: null }))?.data).toEqual({});
  });

  it('accepts a missing created_at and defaults read to false', () => {
    const event = parseNotifEvent(frame({ created_at: null, read: undefined }));
    expect(event?.created_at).toBeNull();
    expect(event?.read).toBe(false);
  });
});

describe('notifEventToInboxItem', () => {
  const parse = (overrides: Record<string, unknown> = {}) => {
    const event = parseNotifEvent(frame(overrides));
    if (!event) {
      throw new Error('fixture did not parse');
    }
    return notifEventToInboxItem(event);
  };

  it('maps the event onto an inbox item', () => {
    expect(parse()).toEqual({
      id: 'e1',
      severity: 'error',
      title: 'Fine-tune failed',
      description: 'CUDA out of memory',
      source: 'Jobs',
      createdAt: '2026-11-18T20:33:23.123456Z',
      link: { label: 'View details', href: '/workspaces/default/jobs/abc' },
      read: false,
    });
  });

  it('carries the read flag from the server', () => {
    expect(parse({ read: true }).read).toBe(true);
  });

  it('uses data.link_label when present and drops an absent link', () => {
    expect(parse({ data: { link_label: 'View logs' } }).link?.label).toBe('View logs');
    expect(parse({ link: null }).link).toBeUndefined();
  });

  it('derives a readable source from an unknown type prefix', () => {
    expect(parse({ type: 'safe_synth.run.done' }).source).toBe('Safe synth');
  });
});
