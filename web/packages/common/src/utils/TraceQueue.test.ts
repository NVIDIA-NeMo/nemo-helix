// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TraceQueue, type TraceData } from './TraceQueue';

describe('TraceQueue', () => {
  let traceQueue: TraceQueue;
  let mockTrace: TraceData;
  let mockTrace2: TraceData;

  beforeEach(() => {
    // Reset singleton instance for each test
    TraceQueue.resetInstance();
    traceQueue = TraceQueue.getInstance();

    mockTrace = {
      id: 'trace-1',
      traceId: 'trace-123',
      timestamp: Date.now(),
      severity: 'error',
      message: 'Test error message',
      context: {
        url: '/api/test',
        method: 'POST',
        statusCode: 500,
        service: 'test-service',
      },
      spans: [
        {
          traceId: 'trace-123',
          spanId: 'span-1',
          operationName: 'test-operation',
          startTime: Date.now(),
          endTime: Date.now() + 100,
          duration: 100,
          tags: { component: 'test' },
          status: { code: 1, message: 'OK' },
        },
      ],
      error: {
        name: 'TestError',
        message: 'Test error occurred',
        stack: 'Error stack trace',
      },
      metadata: { test: 'value' },
    };

    mockTrace2 = {
      ...mockTrace,
      id: 'trace-2',
      traceId: 'trace-456',
      timestamp: Date.now() + 1000,
      severity: 'info',
      message: 'Test info message',
    };

    vi.useFakeTimers();
  });

  afterEach(() => {
    traceQueue.destroy();
    TraceQueue.resetInstance();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  describe('Singleton Pattern', () => {
    it('should return the same instance and maintain state', () => {
      const instance1 = TraceQueue.getInstance();
      instance1.addTrace(mockTrace);

      const instance2 = TraceQueue.getInstance();

      expect(instance1).toBe(instance2);
      expect(instance2.getSize()).toBe(1);
    });

    it('should reset instance properly', () => {
      const instance1 = TraceQueue.getInstance();
      instance1.addTrace(mockTrace);

      TraceQueue.resetInstance();
      const instance2 = TraceQueue.getInstance();

      expect(instance2).not.toBe(instance1);
      expect(instance2.getSize()).toBe(0);
    });
  });

  describe('Configuration & Limits', () => {
    it('should enforce max size limit (default 100)', () => {
      TraceQueue.resetInstance();
      const instance = TraceQueue.getInstance();

      // Add traces beyond max size
      for (let i = 0; i < 102; i++) {
        instance.addTrace({
          ...mockTrace,
          id: `trace-${i}`,
          timestamp: Date.now() + i,
        });
      }

      expect(instance.getSize()).toBe(100);
    });
  });

  describe('Adding Traces', () => {
    it('should add a trace to the queue', () => {
      traceQueue.addTrace(mockTrace);

      expect(traceQueue.getSize()).toBe(1);
      expect(traceQueue.getTrace(mockTrace.id)).toEqual(mockTrace);
    });

    it('should sort traces by timestamp with most recent first', () => {
      const olderTrace = { ...mockTrace, timestamp: Date.now() - 1000 };
      const newerTrace = { ...mockTrace2, timestamp: Date.now() + 1000 };

      traceQueue.addTrace(olderTrace);
      traceQueue.addTrace(newerTrace);

      const traces = traceQueue.getTraces();
      expect(traces[0]).toEqual(newerTrace);
      expect(traces[1]).toEqual(olderTrace);
    });

    it('should handle duplicate traces by updating existing ones', () => {
      traceQueue.addTrace(mockTrace);

      const updatedTrace = { ...mockTrace, message: 'Updated message' };
      traceQueue.addTrace(updatedTrace);

      expect(traceQueue.getSize()).toBe(1);
      expect(traceQueue.getTrace(mockTrace.id)?.message).toBe('Updated message');
    });
  });

  describe('Removing Traces', () => {
    it('should remove a trace from the queue', () => {
      traceQueue.addTrace(mockTrace);
      const removed = traceQueue.removeTrace(mockTrace.id);

      expect(removed).toBe(true);
      expect(traceQueue.getSize()).toBe(0);
      expect(traceQueue.getTrace(mockTrace.id)).toBeUndefined();
    });

    it('should return false when removing non-existent trace', () => {
      const removed = traceQueue.removeTrace('non-existent');
      expect(removed).toBe(false);
    });

    it('should clear all traces', () => {
      traceQueue.addTrace(mockTrace);
      traceQueue.addTrace(mockTrace2);

      traceQueue.clearTraces();

      expect(traceQueue.getSize()).toBe(0);
      expect(traceQueue.getTraces()).toHaveLength(0);
    });
  });

  describe('Cleanup System', () => {
    it('should not clean up traces when there are active subscribers', () => {
      TraceQueue.resetInstance();
      const instance = TraceQueue.getInstance();

      const expiredTrace = { ...mockTrace, timestamp: Date.now() - 400000 };
      instance.addTrace(expiredTrace);

      const listener = vi.fn();
      instance.subscribe(listener);

      instance.cleanup();
      expect(instance.getSize()).toBe(1);
    });

    it('should clean up expired traces when no subscribers', () => {
      TraceQueue.resetInstance();
      const instance = TraceQueue.getInstance();

      const expiredTrace = { ...mockTrace, timestamp: Date.now() - 400000 };
      const validTrace = { ...mockTrace2, timestamp: Date.now() };

      instance.addTrace(expiredTrace);
      vi.advanceTimersByTime(400000);
      instance.addTrace(validTrace);

      instance.cleanup();

      expect(instance.getSize()).toBe(1);
      expect(instance.getTrace(validTrace.id)).toBeDefined();
      expect(instance.getTrace(expiredTrace.id)).toBeUndefined();
    });

    it('should automatically purge expired traces periodically', () => {
      TraceQueue.resetInstance();
      const instance = TraceQueue.getInstance();

      const expiredTrace = { ...mockTrace, timestamp: Date.now() - 400000 };
      instance.addTrace(expiredTrace);

      vi.advanceTimersByTime(400000);
      expect(instance.getSize()).toBe(0);
    });
  });

  describe('Events & Subscriptions', () => {
    it('should notify listeners of trace lifecycle events', () => {
      const listener = vi.fn();
      traceQueue.subscribe(listener);

      // Test add events
      traceQueue.addTrace(mockTrace);
      expect(listener).toHaveBeenCalledWith({ type: 'added', trace: mockTrace });
      expect(listener).toHaveBeenCalledWith({ type: 'changed', traces: [mockTrace] });

      listener.mockClear();

      // Test remove events
      traceQueue.removeTrace(mockTrace.id);
      expect(listener).toHaveBeenCalledWith({ type: 'removed', trace: mockTrace });
      expect(listener).toHaveBeenCalledWith({ type: 'changed', traces: [] });
    });

    it('should notify listeners when traces are cleared', () => {
      const listener = vi.fn();
      traceQueue.addTrace(mockTrace);
      traceQueue.addTrace(mockTrace2);
      traceQueue.subscribe(listener);

      traceQueue.clearTraces();

      expect(listener).toHaveBeenCalledWith({
        type: 'cleared',
        traces: [mockTrace2, mockTrace],
      });
      expect(listener).toHaveBeenCalledWith({ type: 'changed', traces: [] });
    });

    it('should return unsubscribe function and track subscriber count', () => {
      expect(traceQueue.getSubscriberCount()).toBe(0);

      const listener = vi.fn();
      const unsubscribe = traceQueue.subscribe(listener);
      expect(traceQueue.getSubscriberCount()).toBe(1);

      traceQueue.addTrace(mockTrace);
      expect(listener).toHaveBeenCalledTimes(2); // 'added' and 'changed' events

      unsubscribe();
      expect(traceQueue.getSubscriberCount()).toBe(0);

      traceQueue.addTrace(mockTrace2);
      expect(listener).toHaveBeenCalledTimes(2); // Should not be called again
    });

    it('should handle listener errors gracefully', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const faultyListener = vi.fn(() => {
        throw new Error('Listener error');
      });
      const goodListener = vi.fn();

      traceQueue.subscribe(faultyListener);
      traceQueue.subscribe(goodListener);

      traceQueue.addTrace(mockTrace);

      expect(consoleSpy).toHaveBeenCalledWith('Error in TraceQueue listener:', expect.any(Error));
      expect(goodListener).toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });

  describe('Filtering', () => {
    beforeEach(() => {
      const errorTrace = {
        ...mockTrace,
        severity: 'error' as const,
        message: 'Test error message',
      };
      const infoTrace = {
        ...mockTrace2,
        severity: 'info' as const,
        message: 'Test info message',
        error: undefined,
      };
      const debugTrace = {
        ...mockTrace,
        id: 'trace-3',
        severity: 'debug' as const,
        message: 'Test debug message',
        error: undefined,
      };

      traceQueue.addTrace(errorTrace);
      traceQueue.addTrace(infoTrace);
      traceQueue.addTrace(debugTrace);
    });

    it('should filter by severity', () => {
      const errorTraces = traceQueue.filterTraces({ severity: 'error' });
      expect(errorTraces).toHaveLength(1);
      expect(errorTraces[0].severity).toBe('error');
    });

    it('should filter by time range', () => {
      const now = Date.now();
      const timeRange = { start: now - 500, end: now + 500 };

      const filtered = traceQueue.filterTraces({ timeRange });
      expect(filtered.length).toBeGreaterThan(0);
    });

    it('should filter by search text across message, error, and spans', () => {
      const errorFiltered = traceQueue.filterTraces({ searchText: 'error' });
      expect(errorFiltered).toHaveLength(1);
      expect(errorFiltered[0].message).toContain('error');

      const spanFiltered = traceQueue.filterTraces({ searchText: 'test-operation' });
      expect(spanFiltered.length).toBeGreaterThan(0);
    });

    it('should filter by service', () => {
      const filtered = traceQueue.filterTraces({ service: 'test-service' });
      expect(filtered.length).toBeGreaterThan(0);
    });

    it('should filter out expired traces when excludeExpired is true', () => {
      TraceQueue.resetInstance();
      const instance = TraceQueue.getInstance();

      const expiredTrace = { ...mockTrace, timestamp: Date.now() - 400000 };
      const validTrace = { ...mockTrace2, timestamp: Date.now() };

      instance.addTrace(expiredTrace);
      vi.advanceTimersByTime(400000);
      instance.addTrace(validTrace);

      const filtered = instance.filterTraces({ excludeExpired: true });
      expect(filtered).toHaveLength(1);
      expect(filtered[0].id).toBe(validTrace.id);
    });

    it('should combine multiple filters', () => {
      const filtered = traceQueue.filterTraces({
        severity: 'error',
        searchText: 'Test',
        service: 'test-service',
      });
      expect(filtered).toHaveLength(1);
    });
  });

  describe('Getters', () => {
    it('should get all traces as readonly array', () => {
      traceQueue.addTrace(mockTrace);
      traceQueue.addTrace(mockTrace2);

      const traces = traceQueue.getTraces();
      expect(traces).toHaveLength(2);

      // Should be readonly - mutations should not affect original
      expect(() => {
        // @ts-expect-error - we want to test that the array is readonly
        traces.push(mockTrace);
      }).not.toThrow();

      expect(traceQueue.getSize()).toBe(2);
    });

    it('should get specific trace by ID or return undefined', () => {
      traceQueue.addTrace(mockTrace);

      const retrieved = traceQueue.getTrace(mockTrace.id);
      expect(retrieved).toEqual(mockTrace);

      const nonExistent = traceQueue.getTrace('non-existent');
      expect(nonExistent).toBeUndefined();
    });

    it('should track size correctly', () => {
      expect(traceQueue.getSize()).toBe(0);

      traceQueue.addTrace(mockTrace);
      expect(traceQueue.getSize()).toBe(1);

      traceQueue.addTrace(mockTrace2);
      expect(traceQueue.getSize()).toBe(2);
    });
  });

  describe('Destroy', () => {
    it('should clean up resources when destroyed', () => {
      traceQueue.addTrace(mockTrace);
      const listener = vi.fn();
      traceQueue.subscribe(listener);

      expect(traceQueue.getSize()).toBe(1);
      expect(traceQueue.getSubscriberCount()).toBe(1);

      traceQueue.destroy();

      expect(traceQueue.getSize()).toBe(0);
      expect(traceQueue.getSubscriberCount()).toBe(0);
    });
  });
});
