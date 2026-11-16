// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InputFileSchemaType } from '@nemo/common/src/types';
import {
  validateFileFormat,
  detectFileStructure,
  findMessageByRole,
} from '@nemo/common/src/utils/fileValidation';

// Mock File constructor for testing
const createMockFile = (content: string, name: string = 'test.json'): File => {
  const blob = new Blob([content], { type: 'application/json' });
  const file = new File([blob], name, { type: 'application/json' });
  file.text = vi.fn().mockResolvedValue(content);
  return file;
};

describe('validateFileFormat', () => {
  it('should detect JSON format for object and array', async () => {
    const objectFile = createMockFile('{"name": "test"}');
    const arrayFile = createMockFile('[{"name": "test"}]');
    expect(await validateFileFormat(objectFile)).toEqual({ isValid: true, format: 'json' });
    expect(await validateFileFormat(arrayFile)).toEqual({ isValid: true, format: 'json' });
  });

  it('should detect JSONL format', async () => {
    const file = createMockFile('{"name": "test1"}\n{"name": "test2"}');
    const result = await validateFileFormat(file);
    expect(result).toEqual({ isValid: true, format: 'jsonl' });
  });

  it('should handle whitespace correctly', async () => {
    const jsonWithWhitespace = createMockFile('  {"name": "test"}  \n');
    const jsonlWithWhitespace = createMockFile('  {"name": "test1"}  \n  {"name": "test2"}  ');
    expect(await validateFileFormat(jsonWithWhitespace)).toEqual({ isValid: true, format: 'json' });
    expect(await validateFileFormat(jsonlWithWhitespace)).toEqual({
      isValid: true,
      format: 'jsonl',
    });
  });

  it('should return error for invalid content', async () => {
    const invalidFile = createMockFile('invalid');
    const emptyFile = createMockFile('');
    const whitespaceFile = createMockFile('   \n  \t  ');

    expect((await validateFileFormat(invalidFile)).isValid).toBe(false);
    expect((await validateFileFormat(emptyFile)).isValid).toBe(false);
    expect((await validateFileFormat(whitespaceFile)).isValid).toBe(false);
  });

  it('should return error for malformed JSONL', async () => {
    const file = createMockFile('{"valid": "json"}\ninvalid line');
    const result = await validateFileFormat(file);
    expect(result.isValid).toBe(false);
    expect(result.error).toBeDefined();
  });
});

describe('findMessageByRole', () => {
  it('should find first message matching role with content', () => {
    const messages = [
      { role: 'system', content: 'System' },
      { role: 'user', content: 'First user' },
      { role: 'user', content: 'Second user' },
    ];
    expect(findMessageByRole(messages, 'user')).toEqual({
      index: 1,
      content: 'First user',
    });
  });

  it('should return null when role not found or content empty', () => {
    const messages = [{ role: 'user', content: '' }];
    expect(findMessageByRole(messages, 'user')).toBeNull();
    expect(findMessageByRole(messages, 'assistant')).toBeNull();
  });

  it('should find system message even without content', () => {
    const messages = [{ role: 'system' }];
    expect(findMessageByRole(messages, 'system')).toEqual({ index: 0, content: '' });
  });

  it('should skip invalid elements', () => {
    const messages = [null, 'string', { role: 'user', content: 'Hello' }] as Array<{
      role?: string;
      content?: string;
    }>;
    expect(findMessageByRole(messages, 'user')).toEqual({ index: 2, content: 'Hello' });
  });
});

describe('detectFileStructure', () => {
  describe('Messages format', () => {
    it('should detect complete messages format', async () => {
      const data = {
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi' },
          { role: 'system', content: 'System' },
        ],
      };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.CHAT_COMPLETION,
          detectedMessages: {
            user: { index: 0, selector: 'messages[0].content' },
            assistant: { index: 1, selector: 'messages[1].content' },
            system: { index: 2, selector: 'messages[2].content' },
          },
          messagesKey: 'messages',
          isComplete: true,
        })
      );
    });

    it('should detect partial messages format (incomplete)', async () => {
      const data = { messages: [{ role: 'user', content: 'Hello' }] };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.CHAT_COMPLETION,
          detectedMessages: {
            user: { index: 0, selector: 'messages[0].content' },
          },
          isComplete: false,
        })
      );
    });

    it('should find messages under any key name', async () => {
      const data = {
        conversation: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi' },
        ],
      };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.CHAT_COMPLETION,
          messagesKey: 'conversation',
          isComplete: true,
        })
      );
    });

    it('should skip messages without content', async () => {
      const data = {
        messages: [
          { role: 'user', content: '' },
          { role: 'user', content: 'Real question' },
          { role: 'assistant', content: 'Answer' },
        ],
      };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          detectedMessages: {
            user: { index: 1, selector: 'messages[1].content' },
            assistant: { index: 2, selector: 'messages[2].content' },
          },
        })
      );
    });
  });

  describe('Prompt-completion format', () => {
    it('should detect complete prompt-completion format', async () => {
      const data = { prompt: 'Question', completion: 'Answer' };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.COMPLETION,
          detectedFields: {
            prompt: 'prompt',
            completion: 'completion',
          },
          isComplete: true,
        })
      );
    });

    it('should detect alternative key names', async () => {
      const data = { question: 'Q', ideal_response: 'A' };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.COMPLETION,
          detectedFields: {
            prompt: 'question',
            completion: 'ideal_response',
          },
          isComplete: true,
        })
      );
    });

    it('should detect partial prompt-completion (only prompt)', async () => {
      const data = { prompt: 'Question', other: 'value' };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.COMPLETION,
          detectedFields: { prompt: 'prompt' },
          isComplete: false,
        })
      );
    });

    it('should detect partial prompt-completion (only completion)', async () => {
      const data = { completion: 'Answer', other: 'value' };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.COMPLETION,
          detectedFields: { completion: 'completion' },
          isComplete: false,
        })
      );
    });
  });

  describe('JSONL format', () => {
    it('should detect messages in JSONL', async () => {
      const data = {
        messages: [
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi' },
        ],
      };
      const file = createMockFile(JSON.stringify(data), 'test.jsonl');
      const result = await detectFileStructure(file, 'jsonl');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.CHAT_COMPLETION,
          isComplete: true,
        })
      );
    });

    it('should detect prompt-completion in JSONL', async () => {
      const data = { prompt: 'Q', completion: 'A' };
      const file = createMockFile(JSON.stringify(data), 'test.jsonl');
      const result = await detectFileStructure(file, 'jsonl');

      expect(result).toEqual(
        expect.objectContaining({
          schemaType: InputFileSchemaType.COMPLETION,
          isComplete: true,
        })
      );
    });
  });

  describe('Error cases', () => {
    it('should return null for empty file', async () => {
      const file = createMockFile('');
      expect(await detectFileStructure(file, 'json')).toBeNull();
    });

    it('should return null for invalid JSON', async () => {
      const file = createMockFile('invalid');
      expect(await detectFileStructure(file, 'json')).toBeNull();
    });

    it('should return unknown schema for unrecognized format', async () => {
      const data = { someField: 'value', anotherField: 'value' };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual({
        schemaType: null,
        firstRow: data,
      });
    });

    it('should return unknown schema for invalid messages structure', async () => {
      const data = { messages: 'not an array' };
      const file = createMockFile(JSON.stringify(data));
      const result = await detectFileStructure(file, 'json');

      expect(result).toEqual({
        schemaType: null,
        firstRow: data,
      });
    });
  });
});
