// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  triggerDownload,
  findMessagesArray,
  getFirstRow,
  extractUserFriendlyKeysFromRow,
  resolveKeyPath,
} from './file';

describe('triggerDownload', () => {
  // Mock DOM methods and properties
  let mockCreateObjectURL: ReturnType<typeof vi.fn>;
  let mockRevokeObjectURL: ReturnType<typeof vi.fn>;
  let mockAppendChild: ReturnType<typeof vi.fn>;
  let mockRemoveChild: ReturnType<typeof vi.fn>;
  let mockClick: ReturnType<typeof vi.fn>;
  let mockAnchorElement: HTMLAnchorElement;

  beforeEach(() => {
    // Mock URL methods
    mockCreateObjectURL = vi.fn().mockReturnValue('blob:mock-url');
    mockRevokeObjectURL = vi.fn();
    Object.assign(global.URL, {
      createObjectURL: mockCreateObjectURL,
      revokeObjectURL: mockRevokeObjectURL,
    });

    // Mock DOM methods
    mockAppendChild = vi.fn();
    mockRemoveChild = vi.fn();
    Object.assign(document.body, {
      appendChild: mockAppendChild,
      removeChild: mockRemoveChild,
    });

    // Mock anchor element
    mockClick = vi.fn();
    mockAnchorElement = {
      href: '',
      download: '',
      style: { display: '' },
      click: mockClick,
    } as unknown as HTMLAnchorElement;

    // Mock createElement to return our mock anchor
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName === 'a') {
        return mockAnchorElement;
      }
      return document.createElement(tagName);
    });

    // Mock Blob constructor (Vitest 4: must use function/class, not arrow)
    vi.stubGlobal(
      'Blob',
      vi.fn().mockImplementation(function BlobMock(this: unknown, data: BlobPart[]) {
        return {
          data,
          type: 'application/octet-stream',
        };
      })
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  describe('Basic functionality', () => {
    it('should create a Blob from the provided data', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(Blob).toHaveBeenCalledWith([testData]);
    });

    it('should pass a Blob through untouched so its type survives', () => {
      vi.unstubAllGlobals();
      const blob = new Blob(['tar bytes'], { type: 'application/x-tar' });

      triggerDownload(blob, 'artifacts');

      expect(mockCreateObjectURL).toHaveBeenCalledWith(blob);
      expect((mockCreateObjectURL.mock.calls[0][0] as Blob).type).toBe('application/x-tar');
    });

    it('should create an object URL from the Blob', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(mockCreateObjectURL).toHaveBeenCalledWith(
        expect.objectContaining({ data: [testData] })
      );
    });

    it('should create an anchor element', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(document.createElement).toHaveBeenCalledWith('a');
    });

    it('should set the anchor element properties correctly', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(mockAnchorElement.href).toBe('blob:mock-url');
      expect(mockAnchorElement.download).toBe(filename);
      expect(mockAnchorElement.style.display).toBe('none');
    });

    it('should append the anchor to document body', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(mockAppendChild).toHaveBeenCalledWith(mockAnchorElement);
    });

    it('should trigger a click on the anchor element', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(mockClick).toHaveBeenCalled();
    });

    it('should clean up by removing the anchor element', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(mockRemoveChild).toHaveBeenCalledWith(mockAnchorElement);
    });

    it('should clean up by revoking the object URL', () => {
      const testData = 'test file content';
      const filename = 'test.txt';

      triggerDownload(testData, filename);

      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });
  });

  describe('Data types', () => {
    it.each([
      ['string data', 'Hello, world!', 'greeting.txt'],
      ['ArrayBuffer data', new ArrayBuffer(16), 'data.bin'],
      ['Uint8Array data', new Uint8Array([1, 2, 3, 4]), 'bytes.bin'],
      ['empty string data', '', 'empty.txt'],
    ])('should handle %s', (_description, testData, filename) => {
      triggerDownload(testData, filename);

      expect(Blob).toHaveBeenCalledWith([testData]);
      expect(mockClick).toHaveBeenCalled();
    });
  });

  describe('Filename handling', () => {
    it.each([
      ['simple filenames', 'simple.txt'],
      ['filenames with spaces', 'file with spaces.txt'],
      ['filenames with special characters', 'file-name_v1.2.3.txt'],
      ['filenames with unicode characters', 'файл.txt'],
      ['very long filenames', 'a'.repeat(255) + '.txt'],
    ])('should handle %s', (_description, filename) => {
      const testData = 'test';

      triggerDownload(testData, filename);

      expect(mockAnchorElement.download).toBe(filename);
    });
  });

  describe('Error handling', () => {
    it('should still throw error if createObjectURL fails', () => {
      mockCreateObjectURL.mockImplementation(() => {
        throw new Error('Failed to create object URL');
      });

      const testData = 'test';
      const filename = 'test.txt';

      expect(() => triggerDownload(testData, filename)).toThrow('Failed to create object URL');

      // Blob should still be created
      expect(Blob).toHaveBeenCalledWith([testData]);
    });

    it('should cleanup properly even if click fails', () => {
      mockClick.mockImplementation(() => {
        throw new Error('Click failed');
      });

      const testData = 'test';
      const filename = 'test.txt';

      expect(() => triggerDownload(testData, filename)).toThrow('Click failed');

      // Cleanup should still happen in the finally block
      expect(mockRemoveChild).toHaveBeenCalledWith(mockAnchorElement);
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });

    it('should cleanup properly even if appendChild fails', () => {
      mockAppendChild.mockImplementation(() => {
        throw new Error('appendChild failed');
      });

      const testData = 'test';
      const filename = 'test.txt';

      expect(() => triggerDownload(testData, filename)).toThrow('appendChild failed');

      // Cleanup should still happen in the finally block
      expect(mockRemoveChild).toHaveBeenCalledWith(mockAnchorElement);
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });

    it('should cleanup properly even if setting properties fails', () => {
      // Make setting href throw an error
      Object.defineProperty(mockAnchorElement, 'href', {
        set: () => {
          throw new Error('Setting href failed');
        },
      });

      const testData = 'test';
      const filename = 'test.txt';

      expect(() => triggerDownload(testData, filename)).toThrow('Setting href failed');

      // Cleanup should still happen in the finally block
      expect(mockRemoveChild).toHaveBeenCalledWith(mockAnchorElement);
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });
  });

  describe('Integration scenarios', () => {
    it('should handle the complete download flow', () => {
      const testData = JSON.stringify({ message: 'Hello, world!' });
      const filename = 'data.json';

      triggerDownload(testData, filename);

      // Verify the complete flow
      expect(Blob).toHaveBeenCalledWith([testData]);
      expect(mockCreateObjectURL).toHaveBeenCalled();
      expect(document.createElement).toHaveBeenCalledWith('a');
      expect(mockAnchorElement.href).toBe('blob:mock-url');
      expect(mockAnchorElement.download).toBe(filename);
      expect(mockAnchorElement.style.display).toBe('none');
      expect(mockAppendChild).toHaveBeenCalledWith(mockAnchorElement);
      expect(mockClick).toHaveBeenCalled();
      expect(mockRemoveChild).toHaveBeenCalledWith(mockAnchorElement);
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });

    it('should handle multiple downloads sequentially', () => {
      const testData1 = 'first file';
      const filename1 = 'first.txt';
      const testData2 = 'second file';
      const filename2 = 'second.txt';

      triggerDownload(testData1, filename1);
      triggerDownload(testData2, filename2);

      // Both downloads should have been processed
      expect(Blob).toHaveBeenCalledTimes(2);
      expect(mockCreateObjectURL).toHaveBeenCalledTimes(2);
      expect(mockClick).toHaveBeenCalledTimes(2);
      expect(mockRemoveChild).toHaveBeenCalledTimes(2);
      expect(mockRevokeObjectURL).toHaveBeenCalledTimes(2);
    });
  });
});

describe('findMessagesArray', () => {
  describe('Finding messages array with different key names', () => {
    it('should find messages array under "messages" key', () => {
      const item = {
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should find messages array under "conversation" key', () => {
      const item = {
        conversation: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('conversation');
      expect(result?.value).toEqual(item.conversation);
    });

    it('should find messages array under "chat" key', () => {
      const item = {
        chat: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('chat');
      expect(result?.value).toEqual(item.chat);
    });

    it('should find messages array under "history" key', () => {
      const item = {
        history: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('history');
      expect(result?.value).toEqual(item.history);
    });

    it('should find messages array when it is not the first key', () => {
      const item = {
        id: '123',
        metadata: { version: '1.0' },
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
        otherField: 'value',
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should find the first valid messages array when multiple arrays exist', () => {
      const item = {
        otherArray: [1, 2, 3],
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
        anotherArray: ['a', 'b', 'c'],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });
  });

  describe('Messages array with system role', () => {
    it('should find messages array that includes system role', () => {
      const item = {
        messages: [
          { role: 'system', content: 'You are helpful' },
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });
  });

  describe('Edge cases', () => {
    it('should return result when array has only user message', () => {
      const item = {
        messages: [{ role: 'user', content: 'Hello' }],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should return result when array has user but no assistant', () => {
      const item = {
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'system', content: 'You are helpful' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should return result when array has assistant but no user', () => {
      const item = {
        messages: [
          { role: 'assistant', content: 'Hi there!' },
          { role: 'system', content: 'You are helpful' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should return result when array has only system message', () => {
      const item = {
        messages: [{ role: 'system', content: 'You are helpful' }],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should return result when user message has empty content but assistant has content', () => {
      const item = {
        messages: [
          { role: 'user', content: '' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      // Should return result because assistant message has non-empty content
      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should return result when assistant message has empty content but user has content', () => {
      const item = {
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: '' },
        ],
      };

      // Should return result because user message has non-empty content
      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should return null when both user and assistant have empty content', () => {
      const item = {
        messages: [
          { role: 'user', content: '' },
          { role: 'assistant', content: '' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).toBeNull();
    });

    it('should return null when array elements are not objects', () => {
      const item = {
        messages: ['string1', 'string2'],
      };

      const result = findMessagesArray(item);

      expect(result).toBeNull();
    });

    it('should find valid messages even when array contains null elements', () => {
      const item = {
        messages: [null, { role: 'user', content: 'Hello' }],
      };

      const result = findMessagesArray(item);

      // Should return result because user message is present
      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should find valid messages when array contains null elements and multiple roles', () => {
      const item = {
        messages: [
          null,
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      // Should return result because user message is present
      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should return null when object has no arrays', () => {
      const item = {
        id: '123',
        name: 'test',
        count: 42,
      };

      const result = findMessagesArray(item);

      expect(result).toBeNull();
    });

    it('should return null when object has arrays but none contain valid messages', () => {
      const item = {
        numbers: [1, 2, 3],
        strings: ['a', 'b', 'c'],
        objects: [{ id: 1 }, { id: 2 }],
      };

      const result = findMessagesArray(item);

      expect(result).toBeNull();
    });

    it('should return null for empty object', () => {
      const item = {};

      const result = findMessagesArray(item);

      expect(result).toBeNull();
    });

    it('should find valid messages even when some have missing role property', () => {
      const item = {
        messages: [{ content: 'Hello' }, { role: 'assistant', content: 'Hi there!' }],
      };

      const result = findMessagesArray(item);

      // Should return result because assistant message is present
      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should find valid messages when some have missing content property', () => {
      const item = {
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      // Should return result because user message has content
      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });

    it('should find messages array even when other invalid arrays exist', () => {
      const item = {
        invalidArray1: [1, 2, 3],
        invalidArray2: [{ noRole: 'test' }],
        validMessages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there!' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('validMessages');
      expect(result?.value).toEqual(item.validMessages);
    });

    it('should find messages array when user and assistant are not adjacent', () => {
      const item = {
        messages: [
          { role: 'system', content: 'Context 1' },
          { role: 'user', content: 'Hello' },
          { role: 'system', content: 'Context 2' },
          { role: 'assistant', content: 'Hi there!' },
          { role: 'system', content: 'Context 3' },
        ],
      };

      const result = findMessagesArray(item);

      expect(result).not.toBeNull();
      expect(result?.key).toBe('messages');
      expect(result?.value).toEqual(item.messages);
    });
  });
});

describe('getFirstRow', () => {
  const createMockFile = (content: string, filename = 'test.json'): File => {
    const blob = new Blob([content], { type: 'application/json' });
    const file = new File([blob], filename, { type: 'application/json' });
    // Mock the text() method to return the content
    file.text = vi.fn().mockResolvedValue(content);
    return file;
  };

  describe('JSON format', () => {
    it('should extract first item from JSON array', async () => {
      const data = [
        { id: 1, name: 'First' },
        { id: 2, name: 'Second' },
      ];
      const file = createMockFile(JSON.stringify(data));

      const result = await getFirstRow(file, 'json');

      expect(result).toEqual({ id: 1, name: 'First' });
    });

    it('should handle single JSON object (non-array)', async () => {
      const data = { id: 1, name: 'Single' };
      const file = createMockFile(JSON.stringify(data));

      const result = await getFirstRow(file, 'json');

      expect(result).toEqual(data);
    });

    it('should handle complex nested objects', async () => {
      const data = {
        user: { name: 'John', age: 30 },
        messages: [{ role: 'user', content: 'Hello' }],
      };
      const file = createMockFile(JSON.stringify(data));

      const result = await getFirstRow(file, 'json');

      expect(result).toEqual(data);
    });
  });

  describe('JSONL format', () => {
    it('should extract first line from JSONL', async () => {
      const content = '{"id":1,"name":"First"}\n{"id":2,"name":"Second"}';
      const file = createMockFile(content);

      const result = await getFirstRow(file, 'jsonl');

      expect(result).toEqual({ id: 1, name: 'First' });
    });

    it('should handle JSONL with whitespace (trim)', async () => {
      const content = '  \n{"id":1,"name":"First"}\n{"id":2,"name":"Second"}\n  ';
      const file = createMockFile(content);

      const result = await getFirstRow(file, 'jsonl');

      expect(result).toEqual({ id: 1, name: 'First' });
    });
  });

  describe('Error cases', () => {
    it('should return null for empty file', async () => {
      const file = createMockFile('');

      expect(await getFirstRow(file, 'json')).toBeNull();
      expect(await getFirstRow(createMockFile(''), 'jsonl')).toBeNull();
    });

    it('should return null for empty JSON array', async () => {
      const file = createMockFile('[]');

      expect(await getFirstRow(file, 'json')).toBeNull();
    });

    it('should return null for invalid JSON/JSONL', async () => {
      const jsonFile = createMockFile('invalid json');
      const jsonlFile = createMockFile('invalid jsonl\n{"valid": true}');

      expect(await getFirstRow(jsonFile, 'json')).toBeNull();
      expect(await getFirstRow(jsonlFile, 'jsonl')).toBeNull();
    });

    it('should return null for whitespace-only file', async () => {
      const file = createMockFile('   \n  \t  ');

      expect(await getFirstRow(file, 'json')).toBeNull();
      expect(await getFirstRow(createMockFile('   \n  \t  '), 'jsonl')).toBeNull();
    });
  });
});

describe('extractUserFriendlyKeysFromRow', () => {
  it('should extract regular keys from simple object', () => {
    const item = {
      prompt: 'What is AI?',
      completion: 'AI is...',
      metadata: { source: 'test' },
    };

    const result = extractUserFriendlyKeysFromRow(item);

    expect(result).toEqual([
      { label: 'prompt', value: 'prompt' },
      { label: 'completion', value: 'completion' },
      { label: 'metadata', value: 'metadata' },
    ]);
  });

  it('should expand messages array into individual content paths with role labels', () => {
    const item = {
      messages: [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there' },
        { role: 'system', content: 'You are helpful' },
      ],
      id: '123',
    };

    const result = extractUserFriendlyKeysFromRow(item);

    expect(result).toEqual([
      { label: 'messages: user message', value: 'messages[0].content' },
      { label: 'messages: assistant message', value: 'messages[1].content' },
      { label: 'messages: system message', value: 'messages[2].content' },
      { label: 'id', value: 'id' },
    ]);
  });

  it('should use pre-computed messagesArrayResult when provided', () => {
    const item = {
      conversation: [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi' },
      ],
    };

    const messagesArrayResult = {
      key: 'conversation',
      value: [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi' },
      ],
    };

    const result = extractUserFriendlyKeysFromRow(item, messagesArrayResult);

    expect(result).toEqual([
      { label: 'conversation: user message', value: 'conversation[0].content' },
      { label: 'conversation: assistant message', value: 'conversation[1].content' },
    ]);
  });

  it('should skip messages without content or with empty content', () => {
    const item = {
      messages: [
        { role: 'user', content: '' }, // Empty content - skip
        { role: 'user', content: 'Valid message' },
        { role: 'assistant' }, // No content - skip
        { role: 'assistant', content: 'Valid response' },
      ],
    };

    const result = extractUserFriendlyKeysFromRow(item);

    expect(result).toEqual([
      { label: 'messages: user message', value: 'messages[1].content' },
      { label: 'messages: assistant message', value: 'messages[3].content' },
    ]);
  });

  it('should skip messages without role', () => {
    const item = {
      messages: [
        { content: 'No role' }, // No role - skip
        { role: 'user', content: 'Has role' },
      ],
    };

    const result = extractUserFriendlyKeysFromRow(item);

    expect(result).toEqual([{ label: 'messages: user message', value: 'messages[1].content' }]);
  });

  it('should handle mixed object with regular keys and messages', () => {
    const item = {
      id: 'abc',
      messages: [
        { role: 'user', content: 'Question' },
        { role: 'assistant', content: 'Answer' },
      ],
      metadata: { source: 'test' },
    };

    const result = extractUserFriendlyKeysFromRow(item);

    expect(result).toEqual([
      { label: 'id', value: 'id' },
      { label: 'messages: user message', value: 'messages[0].content' },
      { label: 'messages: assistant message', value: 'messages[1].content' },
      { label: 'metadata', value: 'metadata' },
    ]);
  });

  it('should handle explicit null for messagesArrayResult parameter', () => {
    const item = {
      prompt: 'Question',
      completion: 'Answer',
    };

    const result = extractUserFriendlyKeysFromRow(item, null);

    expect(result).toEqual([
      { label: 'prompt', value: 'prompt' },
      { label: 'completion', value: 'completion' },
    ]);
  });

  it('should treat array without messages structure as regular key', () => {
    const item = {
      tags: ['tag1', 'tag2'],
      scores: [1, 2, 3],
    };

    const result = extractUserFriendlyKeysFromRow(item);

    expect(result).toEqual([
      { label: 'tags', value: 'tags' },
      { label: 'scores', value: 'scores' },
    ]);
  });
});

describe('resolveKeyPath', () => {
  const row = {
    question: 'q',
    messages: [
      { role: 'system', content: 's' },
      { role: 'user', content: 'u1' },
      { role: 'assistant', content: 'a1' },
      { role: 'user', content: 'u2' },
      { role: 'assistant', content: 'a2' },
    ],
  };

  it('resolves plain and positional paths', () => {
    expect(resolveKeyPath(row, 'question')).toBe('q');
    expect(resolveKeyPath(row, 'messages[1].content')).toBe('u1');
    expect(resolveKeyPath(row, 'messages[9].content')).toBeUndefined();
  });

  it('selects the last element matching a predicate', () => {
    expect(resolveKeyPath(row, 'messages[role=user].content')).toBe('u2');
    expect(resolveKeyPath(row, 'messages[role=assistant].content')).toBe('a2');
    expect(resolveKeyPath(row, 'messages[role=system]')).toEqual({ role: 'system', content: 's' });
  });

  it('returns undefined when a predicate matches nothing', () => {
    expect(resolveKeyPath(row, 'messages[role=tool].content')).toBeUndefined();
  });

  it('returns undefined when a predicate is applied to a non-array', () => {
    expect(resolveKeyPath(row, 'question[role=user]')).toBeUndefined();
    expect(resolveKeyPath({ messages: { role: 'user' } }, 'messages[role=user]')).toBeUndefined();
  });

  it('matches string-valued fields only', () => {
    expect(resolveKeyPath({ items: [{ done: true }] }, 'items[done=true]')).toBeUndefined();
  });
});
