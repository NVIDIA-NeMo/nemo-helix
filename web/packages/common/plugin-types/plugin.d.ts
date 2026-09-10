// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import * as React$2 from "react";
import React$1, { CSSProperties, ChangeEvent, ComponentProps, ComponentPropsWithRef, ComponentPropsWithoutRef, ComponentType, ElementType, FC, ForwardRefExoticComponent, JSX as JSX$1, JSXElementConstructor, MouseEventHandler, PropsWithChildren, ReactElement, ReactNode, RefAttributes, RefObject, SVGProps } from "react";
import { ThreadMessageLike, ThreadPrimitive } from "@assistant-ui/react";
import { PlatformJobLog, PlatformJobStatus, PromptData } from "@nemo/sdk/generated/platform/schema";
import { VariantProps } from "class-variance-authority";
//#endregion
//#region src/components/AccessibleTitle/index.d.ts
interface AccessibleTitleProps {
  title?: string;
}
/**
 * AccessibleTitle is a small wrapper component that updates the document title, which
 * both makes it easier to keep track of if you have many tabs open, and also makes route
 * changes audible to screen readers.
 */
export declare const AccessibleTitle: FC<PropsWithChildren<AccessibleTitleProps>>;
//#endregion
//#region ../../node_modules/.pnpm/@types+unist@3.0.3/node_modules/@types/unist/index.d.ts
// ## Interfaces
/**
 * Info associated with nodes by the ecosystem.
 *
 * This space is guaranteed to never be specified by unist or specifications
 * implementing unist.
 * But you can use it in utilities and plugins to store data.
 *
 * This type can be augmented to register custom data.
 * For example:
 *
 * ```ts
 * declare module 'unist' {
 *   interface Data {
 *     // `someNode.data.myId` is typed as `number | undefined`
 *     myId?: number | undefined
 *   }
 * }
 * ```
 */
interface Data$1 {}
/**
 * One place in a source file.
 */
interface Point {
  /**
   * Line in a source file (1-indexed integer).
   */
  line: number;
  /**
   * Column in a source file (1-indexed integer).
   */
  column: number;
  /**
   * Character in a source file (0-indexed integer).
   */
  offset?: number | undefined;
}
/**
 * Position of a node in a source document.
 *
 * A position is a range between two points.
 */
interface Position {
  /**
   * Place of the first character of the parsed source region.
   */
  start: Point;
  /**
   * Place of the first character after the parsed source region.
   */
  end: Point;
}
/**
 * Abstract unist node.
 *
 * The syntactic unit in unist syntax trees are called nodes.
 *
 * This interface is supposed to be extended.
 * If you can use {@link Literal} or {@link Parent}, you should.
 * But for example in markdown, a `thematicBreak` (`***`), is neither literal
 * nor parent, but still a node.
 */
interface Node$1 {
  /**
   * Node type.
   */
  type: string;
  /**
   * Info from the ecosystem.
   */
  data?: Data$1 | undefined;
  /**
   * Position of a node in a source document.
   *
   * Nodes that are generated (not in the original source document) must not
   * have a position.
   */
  position?: Position | undefined;
}
//#endregion
//#region ../../node_modules/.pnpm/@types+hast@3.0.4/node_modules/@types/hast/index.d.ts
// ## Interfaces
/**
 * Info associated with hast nodes by the ecosystem.
 *
 * This space is guaranteed to never be specified by unist or hast.
 * But you can use it in utilities and plugins to store data.
 *
 * This type can be augmented to register custom data.
 * For example:
 *
 * ```ts
 * declare module 'hast' {
 *   interface Data {
 *     // `someNode.data.myId` is typed as `number | undefined`
 *     myId?: number | undefined
 *   }
 * }
 * ```
 */
interface Data extends Data$1 {}
/**
 * Info associated with an element.
 */
interface Properties {
  [PropertyName: string]: boolean | number | string | null | undefined | Array<string | number>;
}
// ## Content maps
/**
 * Union of registered hast nodes that can occur in {@link Element}.
 *
 * To register mote custom hast nodes, add them to {@link ElementContentMap}.
 * They will be automatically added here.
 */
type ElementContent = ElementContentMap[keyof ElementContentMap];
/**
 * Registry of all hast nodes that can occur as children of {@link Element}.
 *
 * For a union of all {@link Element} children, see {@link ElementContent}.
 */
interface ElementContentMap {
  comment: Comment;
  element: Element$1;
  text: Text$1;
}
/**
 * Union of registered hast nodes that can occur in {@link Root}.
 *
 * To register custom hast nodes, add them to {@link RootContentMap}.
 * They will be automatically added here.
 */
type RootContent = RootContentMap[keyof RootContentMap];
/**
 * Registry of all hast nodes that can occur as children of {@link Root}.
 *
 * > 👉 **Note**: {@link Root} does not need to be an entire document.
 * > it can also be a fragment.
 *
 * For a union of all {@link Root} children, see {@link RootContent}.
 */
interface RootContentMap {
  comment: Comment;
  doctype: Doctype;
  element: Element$1;
  text: Text$1;
}
// ## Abstract nodes
/**
 * Abstract hast node.
 *
 * This interface is supposed to be extended.
 * If you can use {@link Literal} or {@link Parent}, you should.
 * But for example in HTML, a `Doctype` is neither literal nor parent, but
 * still a node.
 *
 * To register custom hast nodes, add them to {@link RootContentMap} and other
 * places where relevant (such as {@link ElementContentMap}).
 *
 * For a union of all registered hast nodes, see {@link Nodes}.
 */
interface Node extends Node$1 {
  /**
   * Info from the ecosystem.
   */
  data?: Data | undefined;
}
/**
 * Abstract hast node that contains the smallest possible value.
 *
 * This interface is supposed to be extended if you make custom hast nodes.
 *
 * For a union of all registered hast literals, see {@link Literals}.
 */
interface Literal extends Node {
  /**
   * Plain-text value.
   */
  value: string;
}
/**
 * Abstract hast node that contains other hast nodes (*children*).
 *
 * This interface is supposed to be extended if you make custom hast nodes.
 *
 * For a union of all registered hast parents, see {@link Parents}.
 */
interface Parent extends Node {
  /**
   * List of children.
   */
  children: RootContent[];
}
// ## Concrete nodes
/**
 * HTML comment.
 */
interface Comment extends Literal {
  /**
   * Node type of HTML comments in hast.
   */
  type: "comment";
  /**
   * Data associated with the comment.
   */
  data?: CommentData | undefined;
}
/**
 * Info associated with hast comments by the ecosystem.
 */
interface CommentData extends Data {}
/**
 * HTML document type.
 */
interface Doctype extends Node$1 {
  /**
   * Node type of HTML document types in hast.
   */
  type: "doctype";
  /**
   * Data associated with the doctype.
   */
  data?: DoctypeData | undefined;
}
/**
 * Info associated with hast doctypes by the ecosystem.
 */
interface DoctypeData extends Data {}
/**
 * HTML element.
 */
interface Element$1 extends Parent {
  /**
   * Node type of elements.
   */
  type: "element";
  /**
   * Tag name (such as `'body'`) of the element.
   */
  tagName: string;
  /**
   * Info associated with the element.
   */
  properties: Properties;
  /**
   * Children of element.
   */
  children: ElementContent[];
  /**
   * When the `tagName` field is `'template'`, a `content` field can be
   * present.
   */
  content?: Root$1 | undefined;
  /**
   * Data associated with the element.
   */
  data?: ElementData | undefined;
}
/**
 * Info associated with hast elements by the ecosystem.
 */
interface ElementData extends Data {}
/**
 * Document fragment or a whole document.
 *
 * Should be used as the root of a tree and must not be used as a child.
 *
 * Can also be used as the value for the content field on a `'template'` element.
 */
interface Root$1 extends Parent {
  /**
   * Node type of hast root.
   */
  type: "root";
  /**
   * Children of root.
   */
  children: RootContent[];
  /**
   * Data associated with the hast root.
   */
  data?: RootData | undefined;
}
/**
 * Info associated with hast root nodes by the ecosystem.
 */
interface RootData extends Data {}
/**
 * HTML character data (plain text).
 */
interface Text$1 extends Literal {
  /**
   * Node type of HTML character data (plain text) in hast.
   */
  type: "text";
  /**
   * Data associated with the text.
   */
  data?: TextData | undefined;
}
/**
 * Info associated with hast texts by the ecosystem.
 */
interface TextData extends Data {}
//#endregion
//#region ../../node_modules/.pnpm/mdast-util-to-hast@13.2.1/node_modules/mdast-util-to-hast/index.d.ts
/**
 * Raw string of HTML embedded into HTML AST.
 */
interface Raw extends Literal {
  /**
   * Node type of raw.
   */
  type: 'raw';
  /**
   * Data associated with the hast raw.
   */
  data?: RawData | undefined;
}
/**
 * Info associated with hast raw nodes by the ecosystem.
 */
interface RawData extends Data {}
// Register nodes in content.
declare module 'hast' {
  interface ElementData {
    /**
     * Custom info relating to the node, if `<code>` in `<pre>`.
     *
     * Defined by `mdast-util-to-hast` (`remark-rehype`).
     */
    meta?: string | null | undefined;
  }
  interface ElementContentMap {
    /**
     * Raw string of HTML embedded into HTML AST.
     */
    raw: Raw;
  }
  interface RootContentMap {
    /**
     * Raw string of HTML embedded into HTML AST.
     */
    raw: Raw;
  }
}
// Register data on mdast.
declare module 'mdast' {
  interface Data {
    /**
     * Field supported by `mdast-util-to-hast` to signal that a node should
     * result in something with these children.
     *
     * When this is defined, when a parent is created, these children will
     * be used.
     */
    hChildren?: ElementContent[] | undefined;
    /**
     * Field supported by `mdast-util-to-hast` to signal that a node should
     * result in a particular element, instead of its default behavior.
     *
     * When this is defined, an element with the given tag name is created.
     * For example, when setting `hName` to `'b'`, a `<b>` element is created.
     */
    hName?: string | undefined;
    /**
     * Field supported by `mdast-util-to-hast` to signal that a node should
     * result in an element with these properties.
     *
     * When this is defined, when an element is created, these properties will
     * be used.
     */
    hProperties?: Properties | undefined;
  }
}
//#endregion
//#region ../../node_modules/.pnpm/react-markdown@9.1.0_@types+react@19.2.14_react@19.2.7/node_modules/react-markdown/lib/index.d.ts
/**
 * Extra fields we pass.
 */
type ExtraProps = {
  /**
   * passed when `passNode` is on.
   */
  node?: Element$1 | undefined;
};
/**
 * Map tag names to components.
 */
type Components$1 = { [Key in Extract<ElementType, string>]?: ElementType<ComponentProps<Key> & ExtraProps>; };
//#endregion
//#region ../../node_modules/.pnpm/react-markdown@9.1.0_@types+react@19.2.14_react@19.2.7/node_modules/react-markdown/index.d.ts
type Components = Components$1;
//#endregion
//#region src/components/Chat/MessageContent/types.d.ts
interface MarkdownTableOptions {
  expandableCells?: boolean;
}
interface MessageContentProps {
  content?: string | null;
  markdownLinkComponent?: Components['a'];
  markdownTableOptions?: MarkdownTableOptions;
  renderAsMarkdown?: boolean;
}
declare namespace shared_d_exports {
  export { AllModels, ChatModel, ComparisonFilter, CompoundFilter, ErrorObject, FunctionDefinition, FunctionParameters, Metadata, Reasoning, ReasoningEffort, ResponseFormatJSONObject, ResponseFormatJSONSchema, ResponseFormatText, ResponsesModel };
}
type AllModels = (string & {}) | ChatModel | 'o1-pro' | 'o1-pro-2025-03-19' | 'computer-use-preview' | 'computer-use-preview-2025-03-11';
type ChatModel = 'gpt-4.1' | 'gpt-4.1-mini' | 'gpt-4.1-nano' | 'gpt-4.1-2025-04-14' | 'gpt-4.1-mini-2025-04-14' | 'gpt-4.1-nano-2025-04-14' | 'o4-mini' | 'o4-mini-2025-04-16' | 'o3' | 'o3-2025-04-16' | 'o3-mini' | 'o3-mini-2025-01-31' | 'o1' | 'o1-2024-12-17' | 'o1-preview' | 'o1-preview-2024-09-12' | 'o1-mini' | 'o1-mini-2024-09-12' | 'gpt-4o' | 'gpt-4o-2024-11-20' | 'gpt-4o-2024-08-06' | 'gpt-4o-2024-05-13' | 'gpt-4o-audio-preview' | 'gpt-4o-audio-preview-2024-10-01' | 'gpt-4o-audio-preview-2024-12-17' | 'gpt-4o-mini-audio-preview' | 'gpt-4o-mini-audio-preview-2024-12-17' | 'gpt-4o-search-preview' | 'gpt-4o-mini-search-preview' | 'gpt-4o-search-preview-2025-03-11' | 'gpt-4o-mini-search-preview-2025-03-11' | 'chatgpt-4o-latest' | 'codex-mini-latest' | 'gpt-4o-mini' | 'gpt-4o-mini-2024-07-18' | 'gpt-4-turbo' | 'gpt-4-turbo-2024-04-09' | 'gpt-4-0125-preview' | 'gpt-4-turbo-preview' | 'gpt-4-1106-preview' | 'gpt-4-vision-preview' | 'gpt-4' | 'gpt-4-0314' | 'gpt-4-0613' | 'gpt-4-32k' | 'gpt-4-32k-0314' | 'gpt-4-32k-0613' | 'gpt-3.5-turbo' | 'gpt-3.5-turbo-16k' | 'gpt-3.5-turbo-0301' | 'gpt-3.5-turbo-0613' | 'gpt-3.5-turbo-1106' | 'gpt-3.5-turbo-0125' | 'gpt-3.5-turbo-16k-0613';
/**
 * A filter used to compare a specified attribute key to a given value using a
 * defined comparison operation.
 */
interface ComparisonFilter {
  /**
   * The key to compare against the value.
   */
  key: string;
  /**
   * Specifies the comparison operator: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`.
   *
   * - `eq`: equals
   * - `ne`: not equal
   * - `gt`: greater than
   * - `gte`: greater than or equal
   * - `lt`: less than
   * - `lte`: less than or equal
   */
  type: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte';
  /**
   * The value to compare against the attribute key; supports string, number, or
   * boolean types.
   */
  value: string | number | boolean;
}
/**
 * Combine multiple filters using `and` or `or`.
 */
interface CompoundFilter {
  /**
   * Array of filters to combine. Items can be `ComparisonFilter` or
   * `CompoundFilter`.
   */
  filters: Array<ComparisonFilter | unknown>;
  /**
   * Type of operation: `and` or `or`.
   */
  type: 'and' | 'or';
}
interface ErrorObject {
  code: string | null;
  message: string;
  param: string | null;
  type: string;
}
interface FunctionDefinition {
  /**
   * The name of the function to be called. Must be a-z, A-Z, 0-9, or contain
   * underscores and dashes, with a maximum length of 64.
   */
  name: string;
  /**
   * A description of what the function does, used by the model to choose when and
   * how to call the function.
   */
  description?: string;
  /**
   * The parameters the functions accepts, described as a JSON Schema object. See the
   * [guide](https://platform.openai.com/docs/guides/function-calling) for examples,
   * and the
   * [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
   * documentation about the format.
   *
   * Omitting `parameters` defines a function with an empty parameter list.
   */
  parameters?: FunctionParameters;
  /**
   * Whether to enable strict schema adherence when generating the function call. If
   * set to true, the model will follow the exact schema defined in the `parameters`
   * field. Only a subset of JSON Schema is supported when `strict` is `true`. Learn
   * more about Structured Outputs in the
   * [function calling guide](docs/guides/function-calling).
   */
  strict?: boolean | null;
}
/**
 * The parameters the functions accepts, described as a JSON Schema object. See the
 * [guide](https://platform.openai.com/docs/guides/function-calling) for examples,
 * and the
 * [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
 * documentation about the format.
 *
 * Omitting `parameters` defines a function with an empty parameter list.
 */
type FunctionParameters = Record<string, unknown>;
/**
 * Set of 16 key-value pairs that can be attached to an object. This can be useful
 * for storing additional information about the object in a structured format, and
 * querying for objects via API or the dashboard.
 *
 * Keys are strings with a maximum length of 64 characters. Values are strings with
 * a maximum length of 512 characters.
 */
type Metadata = Record<string, string>;
/**
 * **o-series models only**
 *
 * Configuration options for
 * [reasoning models](https://platform.openai.com/docs/guides/reasoning).
 */
interface Reasoning {
  /**
   * **o-series models only**
   *
   * Constrains effort on reasoning for
   * [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
   * supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
   * result in faster responses and fewer tokens used on reasoning in a response.
   */
  effort?: ReasoningEffort | null;
  /**
   * @deprecated **Deprecated:** use `summary` instead.
   *
   * A summary of the reasoning performed by the model. This can be useful for
   * debugging and understanding the model's reasoning process. One of `auto`,
   * `concise`, or `detailed`.
   */
  generate_summary?: 'auto' | 'concise' | 'detailed' | null;
  /**
   * A summary of the reasoning performed by the model. This can be useful for
   * debugging and understanding the model's reasoning process. One of `auto`,
   * `concise`, or `detailed`.
   */
  summary?: 'auto' | 'concise' | 'detailed' | null;
}
/**
 * **o-series models only**
 *
 * Constrains effort on reasoning for
 * [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
 * supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
 * result in faster responses and fewer tokens used on reasoning in a response.
 */
type ReasoningEffort = 'low' | 'medium' | 'high' | null;
/**
 * JSON object response format. An older method of generating JSON responses. Using
 * `json_schema` is recommended for models that support it. Note that the model
 * will not generate JSON without a system or user message instructing it to do so.
 */
interface ResponseFormatJSONObject {
  /**
   * The type of response format being defined. Always `json_object`.
   */
  type: 'json_object';
}
/**
 * JSON Schema response format. Used to generate structured JSON responses. Learn
 * more about
 * [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).
 */
interface ResponseFormatJSONSchema {
  /**
   * Structured Outputs configuration options, including a JSON Schema.
   */
  json_schema: ResponseFormatJSONSchema.JSONSchema;
  /**
   * The type of response format being defined. Always `json_schema`.
   */
  type: 'json_schema';
}
declare namespace ResponseFormatJSONSchema {
  /**
   * Structured Outputs configuration options, including a JSON Schema.
   */
  interface JSONSchema {
    /**
     * The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores
     * and dashes, with a maximum length of 64.
     */
    name: string;
    /**
     * A description of what the response format is for, used by the model to determine
     * how to respond in the format.
     */
    description?: string;
    /**
     * The schema for the response format, described as a JSON Schema object. Learn how
     * to build JSON schemas [here](https://json-schema.org/).
     */
    schema?: Record<string, unknown>;
    /**
     * Whether to enable strict schema adherence when generating the output. If set to
     * true, the model will always follow the exact schema defined in the `schema`
     * field. Only a subset of JSON Schema is supported when `strict` is `true`. To
     * learn more, read the
     * [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
     */
    strict?: boolean | null;
  }
}
/**
 * Default response format. Used to generate text responses.
 */
interface ResponseFormatText {
  /**
   * The type of response format being defined. Always `text`.
   */
  type: 'text';
}
type ResponsesModel = (string & {}) | ChatModel | 'o1-pro' | 'o1-pro-2025-03-19' | 'computer-use-preview' | 'computer-use-preview-2025-03-11';
//#endregion
//#region ../../node_modules/.pnpm/openai@4.104.0_ws@8.21.1_zod@3.25.76/node_modules/openai/resources/chat/completions/completions.d.ts
interface ChatCompletionTool {
  function: FunctionDefinition;
  /**
   * The type of the tool. Currently, only `function` is supported.
   */
  type: 'function';
}
//#endregion
//#region src/components/AssistantChat/types.d.ts
declare const ComposerMode: {
  readonly PER_PANEL: 'per-panel';
  readonly BROADCAST_ALL: 'broadcast-all';
};
type ComposerMode = (typeof ComposerMode)[keyof typeof ComposerMode];
interface AssistantChatThreadAttributes {
  ThreadViewport?: ComponentProps<typeof ThreadPrimitive.Viewport>;
}
type AssistantChatMessageContentProps = Pick<MessageContentProps, 'markdownLinkComponent'>;
interface AssistantChatProps {
  /**
   * The model name to route through inference gateway.
   */
  model: string;
  /**
   * Workspace used to build the default inference gateway URL.
   */
  workspace?: string;
  /**
   * Explicit OpenAI-compatible chat completions base URL. When omitted, `useChatCompletion`
   * resolves inference gateway routing from workspace and model.
   */
  baseURL?: string;
  /**
   * Optional prompt data used for system prompt and inference parameter defaults.
   */
  promptData?: PromptData;
  /**
   * Optional OpenAI-compatible tools for the request.
   */
  tools?: ChatCompletionTool[];
  /**
   * Display name used in the composer placeholder.
   */
  assistantName?: string;
  placeholder?: string;
  disabled?: boolean;
  showRunningIndicator?: boolean;
  attributes?: AssistantChatThreadAttributes;
  className?: string;
  initialMessages?: readonly ThreadMessageLike[];
  onError?: (error: Error) => void;
  /**
   * Called once per assistant message after the stream completes (or after
   * the non-stream completion lands). Surfaces per-message timing so callers
   * can render their own latency/throughput UI without owning the runtime.
   * Not invoked on cancellation or error.
   */
  onMessageComplete?: (info: AssistantMessageCompletion) => void;
  /**
   * Fires whenever the runtime's "is currently streaming" state changes.
   * Lets a parent (e.g. a page that hosts many AssistantChats) aggregate the
   * running state across instances — used by the Chat route to drive a global
   * Stop button in Compare mode.
   */
  onRunningChange?: (isRunning: boolean) => void;
  /**
   * Fires whenever the thread transitions between empty and non-empty. Lets a
   * parent derive seed-chip visibility from whether any messages exist.
   */
  onEmptyChange?: (isEmpty: boolean) => void;
  /**
   * Controls whether the internal composer is shown and how input is driven.
   * In `broadcast-all` mode the composer is suppressed; a page-level composer
   * drives every AssistantChat in parallel.
   * @default ComposerMode.PER_PANEL
   */
  composerMode?: ComposerMode;
  /**
   * External broadcast trigger. Whenever `seq` changes (excluding initial
   * mount), the runtime appends `text` as a new user message and runs a
   * completion — same code path as a user typing into the composer.
   */
  broadcast?: BroadcastSignal;
  /**
   * Monotonic counter — when it changes, the runtime aborts any in-flight
   * stream. Lets a parent cancel many AssistantChats at once.
   */
  stopCount?: number;
  /**
   * Content rendered immediately above the composer, inside the same outer
   * frame. Use for seed-prompt chips or any prefatory hint that should read
   * as part of the composer affordance rather than a separate block.
   */
  slotComposerStart?: ReactNode;
  emptyState?: {
    slotHeading?: string;
    slotSubheading?: string;
    slotAction?: ReactNode;
  };
  /** Overrides used when rendering Markdown inside chat messages. */
  messageContentProps?: AssistantChatMessageContentProps;
  composerOverride?: ReactNode;
  /**
   * @default true
   */
  enableImageAttachments?: boolean;
}
interface BroadcastSignal {
  /** Monotonically increasing sequence — on change, runtime fires a send. */
  seq: number;
  /** Text to inject as the user's next message. */
  text: string;
}
interface AssistantMessageCompletion {
  assistantMessageId: string;
  text: string;
  /** ms from request start to first delta (0 if non-stream). */
  ttftMs: number;
  /** ms from request start to final delta. */
  totalMs: number;
  /** Number of delta chunks (1 for non-stream). */
  chunkCount: number;
  /** Approximate; chars/4 fallback when the API doesn't return a usage block. */
  completionTokens: number;
  /** Completion tokens per second of streaming wall-time (excludes TTFT). */
  tokensPerSec: number;
}
//#endregion
//#region src/components/AssistantChat/index.d.ts
export declare const AssistantChat: FC<AssistantChatProps>;
//#endregion
//#region src/components/AccordionSection/index.d.ts
interface AccordionSectionProps {
  icon?: ReactNode;
  isDisabled?: boolean;
  title: string;
  value: string;
  className?: string;
  contentClassName?: string;
}
/**
 * AccordionItem implementation to support a custom icon in the header of an accordion section.
 */
export declare const AccordionSection: FC<PropsWithChildren<AccordionSectionProps>>;
declare namespace clsx_d_exports {
  export { ClassArray, ClassDictionary, ClassValue$1 as ClassValue, clsx, clsx as default };
}
type ClassValue$1 = ClassArray | ClassDictionary | string | number | bigint | null | boolean | undefined;
type ClassDictionary = Record<string, any>;
type ClassArray = ClassValue$1[];
declare function clsx(...inputs: ClassValue$1[]): string;
declare namespace types_d_exports {
  export { ClassProp, ClassPropKey, ClassValue, OmitUndefined, StringToBoolean };
}
type ClassPropKey = "class" | "className";
type ClassValue = ClassValue$1;
type ClassProp = {
  class: ClassValue;
  className?: never;
} | {
  class?: never;
  className: ClassValue;
} | {
  class?: never;
  className?: never;
};
type OmitUndefined<T> = T extends undefined ? never : T;
type StringToBoolean<T> = T extends "true" | "false" ? boolean : T;
//#endregion
//#region ../../node_modules/.pnpm/@radix-ui+react-primitive@2.1.4_@types+react-dom@19.2.3_@types+react@19.2.14__@types+re_79f9cc29726bbcca5df2cac469f5e931/node_modules/@radix-ui/react-primitive/dist/index.d.mts
type PrimitivePropsWithRef$1<E extends React$2.ElementType> = React$2.ComponentPropsWithRef<E> & {
  asChild?: boolean;
};
//#endregion
//#region ../../node_modules/.pnpm/@nvidia+foundations-react-core@1.7.0_@types+react-dom@19.2.3_@types+react@19.2.14__@typ_3c9beb47e01bfdc8313437db9c95ced2/node_modules/@nvidia/foundations-react-core/dist/index.d.ts
/**
 * Common attributes
 * @see {@link https://react.dev/reference/react-dom/components/common}
 */
declare const COMMON_ATTRIBUTES: readonly ["dangerouslySetInnerHTML", "suppressContentEditableWarning", "suppressHydrationWarning", "style", "accessKey", "autoCapitalize", "className", "contentEditable", "dir", "draggable", "enterKeyHint", "hidden", "id", "is", "inputMode", "itemProp", "lang", "onAnimationEnd", "onAnimationEndCapture", "onAnimationIteration", "onAnimationIterationCapture", "onAnimationStart", "onAnimationStartCapture", "onAuxClick", "onAuxClickCapture", "onBeforeInput", "onBeforeInputCapture", "onBlur", "onBlurCapture", "onClick", "onClickCapture", "onCompositionStart", "onCompositionStartCapture", "onCompositionEnd", "onCompositionEndCapture", "onCompositionUpdate", "onCompositionUpdateCapture", "onContextMenu", "onContextMenuCapture", "onCopy", "onCopyCapture", "onCut", "onCutCapture", "onDoubleClick", "onDoubleClickCapture", "onDrag", "onDragCapture", "onDragEnd", "onDragEndCapture", "onDragEnter", "onDragEnterCapture", "onDragOver", "onDragOverCapture", "onDragStart", "onDragStartCapture", "onDrop", "onDropCapture", "onFocus", "onFocusCapture", "onGotPointerCapture", "onGotPointerCaptureCapture", "onKeyDown", "onKeyDownCapture", "onKeyPress", "onKeyPressCapture", "onKeyUp", "onKeyUpCapture", "onLostPointerCapture", "onLostPointerCaptureCapture", "onMouseDown", "onMouseDownCapture", "onMouseEnter", "onMouseLeave", "onMouseMove", "onMouseMoveCapture", "onMouseOut", "onMouseOutCapture", "onMouseUp", "onMouseUpCapture", "onPointerCancel", "onPointerCancelCapture", "onPointerDown", "onPointerDownCapture", "onPointerEnter", "onPointerLeave", "onPointerMove", "onPointerMoveCapture", "onPointerOut", "onPointerOutCapture", "onPointerUp", "onPointerUpCapture", "onPaste", "onPasteCapture", "onScroll", "onScrollCapture", "onSelect", "onSelectCapture", "onTouchCancel", "onTouchCancelCapture", "onTouchEnd", "onTouchEndCapture", "onTouchMove", "onTouchMoveCapture", "onTouchStart", "onTouchStartCapture", "onTransitionEnd", "onTransitionEndCapture", "onWheel", "onWheelCapture", "role", "slot", "spellCheck", "tabIndex", "title", "translate", "onReset", "onResetCapture", "onSubmit", "onSubmitCapture", "onCancel", "onCancelCapture", "onClose", "onCloseCapture", "onToggle", "onToggleCapture", "onLoad", "onLoadCapture", "onError", "onErrorCapture", "onAbort", "onAbortCapture", "onCanPlay", "onCanPlayCapture", "onCanPlayThrough", "onCanPlayThroughCapture", "onDurationChange", "onDurationChangeCapture", "onEmptied", "onEmptiedCapture", "onEncrypted", "onEncryptedCapture", "onEnded", "onEndedCapture", "onLoadedData", "onLoadedDataCapture", "onLoadedMetadata", "onLoadedMetadataCapture", "onLoadStart", "onLoadStartCapture", "onPause", "onPauseCapture", "onPlay", "onPlayCapture", "onPlaying", "onPlayingCapture", "onProgress", "onProgressCapture", "onRateChange", "onRateChangeCapture", "onResize", "onResizeCapture", "onSeeked", "onSeekedCapture", "onSeeking", "onSeekingCapture", "onStalled", "onStalledCapture", "onSuspend", "onSuspendCapture", "onTimeUpdate", "onTimeUpdateCapture", "onVolumeChange", "onVolumeChangeCapture", "onWaiting", "onWaitingCapture"];
/**
 * Input attributes
 * @see {@link https://react.dev/reference/react-dom/components/input}
 */
declare const INPUT_ATTRIBUTES: readonly ["accept", "alt", "capture", "autoComplete", "autoFocus", "checked", "defaultChecked", "defaultValue", "dirname", "disabled", "form", "formAction", "formEncType", "formMethod", "formNoValidate", "formTarget", "height", "list", "max", "maxLength", "min", "minLength", "multiple", "name", "onChange", "onChangeCapture", "onInput", "onInputCapture", "onInvalid", "onInvalidCapture", "onSelect", "onSelectCapture", "pattern", "placeholder", "readOnly", "required", "size", "src", "step", "type", "value", "width", "aria-label", "aria-describedby", "aria-details", "aria-labelledby", "id"];
declare const ELEMENT_ATTRIBUTE_MAP: {
  readonly a: readonly ["href", "target", "rel", "download", "ping", "hrefLang", "referrerPolicy"];
  readonly form: readonly ["action"];
  readonly input: readonly Exclude<(typeof INPUT_ATTRIBUTES)[number], "dirname">[];
  readonly select: readonly ["autoComplete", "autoFocus", "children", "defaultValue", "disabled", "form", "multiple", "name", "onChange", "onChangeCapture", "onInput", "onInputCapture", "onInvalid", "onInvalidCapture", "required", "size", "value", "aria-describedby", "aria-details", "aria-labelledby", "aria-label", "id", "name"];
  readonly textarea: readonly ["autoComplete", "autoFocus", "cols", "defaultValue", "disabled", "form", "maxLength", "minLength", "name", "onChange", "onChangeCapture", "onInput", "onInputCapture", "onInvalid", "onInvalidCapture", "onSelect", "onSelectCapture", "placeholder", "readOnly", "required", "rows", "value", "wrap", "aria-describedby", "aria-details", "aria-labelledby", "aria-label", "id", "name"];
  readonly button: readonly ["type", "disabled", "form", "formAction", "formMethod", "formNoValidate", "formTarget", "name", "value", "aria-describedby", "aria-details", "aria-labelledby", "id"];
  readonly label: readonly ["form", "htmlFor"];
  readonly img: readonly ["src", "alt", "width", "height", "loading", "decoding", "srcSet", "sizes", "crossOrigin", "referrerPolicy", "fetchPriority"];
  readonly progress: readonly ["max", "value"];
};
type SupportedElement = keyof typeof ELEMENT_ATTRIBUTE_MAP;
type AttributesFor<TElement extends SupportedElement> = (typeof ELEMENT_ATTRIBUTE_MAP)[TElement][number];
type ComponentType$1<P = any, T = any> = ForwardRefExoticComponent<P & RefAttributes<T>> | JSXElementConstructor<P>;
/**
 * A tuple type representing an HTML element and its corresponding React component
 * @typeParam T - The HTML element type (e.g., "div", "input")
 * @typeParam C - The React component type
 *
 * @example
 * ```tsx
 * type DivRoot = ElementComponentPair<"div", typeof RootComponent>;
 * type InputField = ElementComponentPair<"input", typeof InputComponent>;
 * ```
 */
type ElementComponentPair<T extends keyof React$1.JSX.IntrinsicElements = keyof React$1.JSX.IntrinsicElements, C extends ComponentType$1 = ComponentType$1> = readonly [T, C];
/**
 * Type utility that allows mapping native HTML attributes to a component while ensuring type safety.
 * It filters out attributes that would conflict with the component's props and allows data attributes to be passed through.
 *
 * @typeParam T - The HTML element type (e.g., "div", "button")
 * @typeParam C - The component type to map attributes to
 *
 * @example
 * ```tsx
 * // Allow passing native div attributes to MyComponent
 * type Props = {
 *   attributes?: NativeElementAttributes<"div", typeof MyComponent>;
 * }
 * ```
 */
type NativeElementAttributes<T extends ElementComponentPair[0], C extends ElementComponentPair[1]> = { [K in keyof React$1.JSX.IntrinsicElements[T] as K extends "children" ? never : K extends keyof React$1.ComponentProps<C> ? React$1.ComponentProps<C>[K] extends React$1.JSX.IntrinsicElements[T][K] ? React$1.JSX.IntrinsicElements[T][K] extends React$1.ComponentProps<C>[K] ? K : never : never : K]: React$1.JSX.IntrinsicElements[T][K]; } & {
  [key: `data-${string}`]: string | number | boolean;
} & ("ref" extends keyof React$1.ComponentProps<C> ? {
  ref?: React$1.ComponentProps<C>["ref"];
} : Record<string, never>);
/**
 * Maps to our static attribute definitions from ELEMENT_ATTRIBUTE_MAP
 * @internal
 */
type AttributeMap = typeof ELEMENT_ATTRIBUTE_MAP;
/**
 * Recursively finds the first element that accepts an attribute and returns its type
 * @internal
 */
type GetAttributeTypeForIndex<K extends string, T extends readonly ElementComponentPair[]> = T extends readonly [infer First extends ElementComponentPair, ...infer Rest extends ElementComponentPair[]] ? K extends keyof React$1.JSX.IntrinsicElements[First[0]] ? React$1.JSX.IntrinsicElements[First[0]][K] : GetAttributeTypeForIndex<K, Rest> : never;
/**
 * Creates a type containing all valid HTML attributes that can be hoisted to components
 * based on our static attribute maps.
 *
 * @remarks
 * This type creates a union of all attributes from the provided element types,
 * using our static maps to determine valid attributes. When an attribute is present
 * on multiple elements, the type comes from the first element in the list that
 * accepts it.
 *
 * @example
 * ```tsx
 * type AvatarProps = MergedHoistedElementAttributes<[
 *   ["div", typeof AvatarRoot],
 *   ["img", typeof AvatarImage],
 *   ["div", typeof AvatarFallback]
 * ]>;
 * // Results in a type with all div and img attributes from our maps
 * // with types from the first element that accepts each attribute
 * ```
 */
type MergedHoistedElementAttributes<T extends readonly ElementComponentPair[]> = Partial<{ [K in T[number] extends readonly [infer E, unknown] ? E extends keyof AttributeMap ? (typeof ELEMENT_ATTRIBUTE_MAP)[E & keyof AttributeMap][number] : never : never]: K extends (typeof COMMON_ATTRIBUTES)[number] ? K extends keyof React$1.JSX.IntrinsicElements[T[0][0]] ? React$1.JSX.IntrinsicElements[T[0][0]][K] : never : GetAttributeTypeForIndex<K & string, T>; } & { [K in (typeof COMMON_ATTRIBUTES)[number]]: K extends keyof React$1.JSX.IntrinsicElements[T[0][0]] ? React$1.JSX.IntrinsicElements[T[0][0]][K] : never; } & {
  [key: `data-${string}`]: string | number | boolean;
  [key: `aria-${string}`]: string | number | boolean;
}>;
declare const primitiveStyles: (props?: ({
  gap?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
  padding?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
  paddingX?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
  paddingY?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
  paddingTop?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
  paddingRight?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
  paddingBottom?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
  paddingLeft?: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "14" | "16" | "18" | "20" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "52" | "56" | "60" | "64" | "72" | "80" | "96" | "250" | "px" | "0.25" | "0.5" | "0.75" | "1.5" | "2.5" | "3.5" | "density-xxs" | "density-xs" | "density-sm" | "density-md" | "density-lg" | "density-xl" | "density-2xl" | "density-3xl" | "density-4xl" | "density-5xl" | "inherit" | null | undefined;
} & ClassProp) | undefined) => string;
type PrimitiveVariantProps = VariantProps<typeof primitiveStyles>;
type PrimitiveProps<E extends React$1.ElementType> = PrimitivePropsWithRef$1<E>;
interface WithAsChild {
  /** Render-as-child slot (from Radix Primitive) */
  asChild?: boolean;
}
interface PrimitiveComponentProps extends WithAsChild {
  /**
   * Sets spacing between flex and grid items.
   */
  gap?: PrimitiveVariantProps["gap"];
  /**
   * Sets padding.
   */
  padding?: PrimitiveVariantProps["padding"];
  /**
   * Sets horizontal padding.
   */
  paddingX?: PrimitiveVariantProps["paddingX"];
  /**
   * Sets vertical padding.
   */
  paddingY?: PrimitiveVariantProps["paddingY"];
  /**
   * Sets top padding.
   */
  paddingTop?: PrimitiveVariantProps["paddingTop"];
  /**
   * Sets right padding.
   */
  paddingRight?: PrimitiveVariantProps["paddingRight"];
  /**
   * Sets bottom padding.
   */
  paddingBottom?: PrimitiveVariantProps["paddingBottom"];
  /**
   * Sets left padding.
   */
  paddingLeft?: PrimitiveVariantProps["paddingLeft"];
}
type PrimitivePropsWithRef<E extends React$1.ElementType> = React$1.ComponentPropsWithRef<E> & WithAsChild;
declare const text$1: (props?: ({
  fontFamily?: "sans" | "mono" | null | undefined;
  fontWeight?: "bold" | "light" | "regular" | "semibold" | null | undefined;
  fontStyle?: "normal" | "italic" | null | undefined;
  fontSize?: "10" | "12" | "14" | "16" | "18" | "20" | "22" | "24" | "28" | "32" | "36" | "40" | "44" | "48" | "50" | "56" | "60" | "64" | "72" | "80" | null | undefined;
  underline?: boolean | null | undefined;
  lineHeight?: "100" | "125" | "150" | "175" | null | undefined;
  kind?: "inherit" | "body/bold/2xl" | "body/bold/3xl" | "body/bold/lg" | "body/bold/md" | "body/bold/xl" | "body/bold/sm" | "body/bold/xs" | "body/regular/lg" | "body/regular/md" | "body/regular/sm" | "body/regular/xl" | "body/regular/2xl" | "body/regular/3xl" | "body/regular/xs" | "body/semibold/2xl" | "body/semibold/3xl" | "body/semibold/lg" | "body/semibold/md" | "body/semibold/sm" | "body/semibold/xl" | "body/semibold/xs" | "display/2xl" | "display/xl" | "display/lg" | "display/md" | "display/sm" | "display/xs" | "label/bold/2xl" | "label/bold/3xl" | "label/bold/lg" | "label/bold/md" | "label/bold/sm" | "label/bold/xl" | "label/bold/xs" | "label/light/lg" | "label/light/xl" | "label/light/2xl" | "label/light/3xl" | "label/light/md" | "label/light/sm" | "label/light/xs" | "label/regular/lg" | "label/regular/md" | "label/regular/sm" | "label/regular/xs" | "label/regular/xl" | "label/regular/2xl" | "label/regular/3xl" | "label/semibold/lg" | "label/semibold/md" | "label/semibold/sm" | "label/semibold/xl" | "label/semibold/2xl" | "label/semibold/3xl" | "label/semibold/xs" | "mono/md" | "mono/sm" | "mono/lg" | "mono/xl" | "mono/2xl" | "title/2xl" | "title/xl" | "title/lg" | "title/md" | "title/sm" | "title/xs" | null | undefined;
} & ClassProp) | undefined) => string;
type TextVariantProps = VariantProps<typeof text$1>;
interface TextProps extends PrimitivePropsWithRef<"span"> {
  /**
     * A semantic typography token combining family, weight, and size. Pass "inherit" to keep the parent's text style.
     Use "display" for the largest hero text, "title" for headings, "body" for paragraphs, "label" for short labels and UI text, and "mono" for code or technical content.
     * @defaultValue "label/regular/md"
     * @llm Common mappings: page heading `title/lg`, section heading `title/md`, sub-section `title/sm`, paragraph `body/regular/md`, card title `body/bold/xl`, metadata `label/regular/sm`, form label `label/regular/sm`, code `mono/sm` or `mono/md`. Use `display/*` only on hero or marketing surfaces.
     */
  kind?: TextVariantProps["kind"];
  /** Overrides the font weight inherited from `kind`. */
  fontWeight?: TextVariantProps["fontWeight"];
  /** Overrides the font family inherited from `kind`. */
  fontFamily?: TextVariantProps["fontFamily"];
  /** Sets the font style. */
  fontStyle?: TextVariantProps["fontStyle"];
  /** Overrides the font size inherited from `kind`, in pixels. */
  fontSize?: TextVariantProps["fontSize"];
  /** Overrides the line height inherited from `kind`, as a percentage of the font size. */
  lineHeight?: TextVariantProps["lineHeight"];
  /** Applies an underline to the text. */
  underline?: boolean;
}
/**
 * Renders text with the design system's typography tokens applied.
 * @param props - {@link TextProps}
 *
 * @llm Always pick a `kind` from the typography scale rather than overriding `fontSize` / `fontWeight` individually.
 * @llm Pair heading kinds (`title/*`, `display/*`) with `asChild` and a semantic `h1`–`h6` so screen readers can navigate by heading level.
 * @llm Do not render text in uppercase — no `uppercase` utility, no manually capitalized strings. Acronyms (API, GPU, URL) are the only exception.
 * @llm Text renders an inline `<span>`, so vertical margin utilities (`mt-*`, `mb-*`, `my-*`) are silently dropped — horizontal margins (`ml-*`, `mr-*`, `mx-*`) work fine. Inside a `Flex` or `Stack` this rarely surfaces because the parent `gap` handles spacing; it bites standalone `Text` in a vertical flow, where you should add `block` (or wrap in a block element).
 * @llm Do not hallucinate `kind` values — only use the ones defined in the type.
 * @llm When in doubt, map context to kind: page heading → `title/lg`; section heading → `title/md`; paragraph → `body/regular/md`; card title → `body/bold/xl`; card description → `body/regular/sm`; metadata → `label/regular/sm` with `text-secondary`; form label → `label/regular/sm`; hero headline → `display/lg`; code → `mono/sm` or `mono/md`.
 * @llm Do NOT render text in uppercase because all-caps reduces reading speed 10-15%
 * @llm `display/*` kinds are reserved for hero and marketing surfaces; use `title/*` for standard in-app headings.
 *
 * @example
 * <caption>Kinds Text</caption>
 * Pick a `kind` from one of the five families: `display/*` for the largest hero text, `title/*` for headings, `body/*` for prose, `label/*` for short UI labels, and `mono/*` for code or technical identifiers. Each family offers multiple sizes and weights — see the `kind` type for the full set.
 * ```tsx
 * <Stack gap="density-md">
 * 	<Text kind="display/lg">Display Heading</Text>
 * 	<Text kind="title/md">Section Title</Text>
 * 	<Text kind="body/regular/md">
 * 		The quick brown fox jumps over the lazy dog
 * 	</Text>
 * 	<Text kind="label/semibold/sm">Form Label</Text>
 * 	<Text kind="mono/md">console.log("hello")</Text>
 * </Stack>
 * ```
 *
 * @example
 * <caption>Custom Styled Text</caption>
 * Reach for the granular fontFamily, fontWeight, fontSize, lineHeight, and fontStyle props as an escape hatch when no predefined kind matches the design.
 * ```tsx
 * <Text
 * 	fontFamily="sans"
 * 	fontWeight="semibold"
 * 	fontSize="18"
 * 	lineHeight="150"
 * 	fontStyle="italic"
 * >
 * 	Custom typography
 * </Text>
 * ```
 *
 * @example
 * <caption>Underlined Text</caption>
 * Add the underline prop on top of any kind to flag inline emphasis like links or highlighted terms without changing the type style.
 * ```tsx
 * <Text kind="body/regular/md" underline>
 * 	Underlined passage
 * </Text>
 * ```
 *
 * @example
 * <caption>Inherit Kind Text</caption>
 * Use kind="inherit" when Text is nested inside an element that already defines the type style (e.g. inside another Text or a styled heading) and you only need the Text primitive's other props.
 * ```tsx
 * <Text kind="inherit">Inherits parent typography</Text>
 * ```
 *
 * @example
 * <caption>As Child Text</caption>
 * Use asChild to apply Text styling to a semantic HTML element like h1–h6 without adding an extra wrapper.
 * ```tsx
 * <Text asChild kind="title/lg">
 * 	<h1>Semantic Heading</h1>
 * </Text>
 * ```
 */
declare const Text: React$2.ForwardRefExoticComponent<Omit<TextProps, "ref"> & React$2.RefAttributes<HTMLSpanElement>>;
declare const anchor: (props?: ({
  kind?: "inline" | "standalone" | null | undefined;
  disabled?: boolean | null | undefined;
} & ClassProp) | undefined) => string;
type AnchorVariantProps = VariantProps<typeof anchor>;
interface AnchorProps extends PrimitivePropsWithRef<"a">, Pick<TextProps, "fontWeight" | "fontFamily" | "fontStyle" | "fontSize" | "lineHeight" | "underline"> {
  /**
   * The kind of anchor.
   * - "inline" - Embedded within prose; renders with an underline so the link remains distinguishable inside body text.
   * - "standalone" - Rendered outside of prose, such as a navigation link or call to action; renders without an underline.
   * @defaultValue "inline"
   */
  kind?: AnchorVariantProps["kind"];
  /**
   * Typography token applied to the link text.
   * @defaultValue "body/regular/md"
   * @llm For standalone anchors, prefer a label family token over the body default.
   */
  textKind?: TextProps["kind"];
  /**
   * Renders the anchor as a non-interactive `<span>` so it no longer navigates.
   * @defaultValue false
   * @llm When using `asChild`, prefer to manage the disabled state on the consuming component instead of passing `disabled` to Anchor.
   */
  disabled?: AnchorVariantProps["disabled"];
}
/**
 * Interactive text that navigates the user to another page, section, or resource.
 * @param props - {@link AnchorProps}
 *
 * @alias Link
 *
 * @llm Use Anchor for navigation (URL or route). For actions that do not navigate (submit, toggle, open a modal), use Button instead.
 * @llm When opening in a new tab, set `target="_blank"` and `rel="noopener"` (add `noreferrer` for untrusted destinations) to avoid the `window.opener` security issue.
 * @see {@link Button}
 * @see {@link Breadcrumbs}
 *
 * @example
 * <caption>Basic Anchor</caption>
 * ```tsx
 * <Flex gap="2">
 * 	<Anchor href="/" kind="inline">
 * 		Inline
 * 	</Anchor>
 * 	<Anchor href="/" kind="standalone">
 * 		Standalone
 * 	</Anchor>
 * </Flex>
 * ```
 *
 * @example
 * <caption>Custom Link Component</caption>
 * Use asChild to render framework specific link components with our styling
 * ```tsx
 * <Anchor asChild>
 * 	<NextLink href="/">Home</NextLink>
 * </Anchor>
 * ```
 *
 * @example
 * <caption>External Link Anchor</caption>
 * When opening in a new tab, pair `target="_blank"` with `rel="noopener"` (add `noreferrer` for untrusted destinations) to avoid leaking the `window.opener` reference.
 * ```tsx
 * <Anchor href="https://example.com" target="_blank" rel="noopener">
 * 	Open documentation
 * </Anchor>
 * ```
 *
 * @example
 * <caption>Disabled Anchor</caption>
 * A disabled anchor will render as a span instead of an anchor tag
 * ```tsx
 * <Anchor disabled href="/">
 * 	Disabled Link
 * </Anchor>
 * ```
 *
 * @example
 * <caption>As Button</caption>
 * For an Anchor that looks like a link but triggers an action
 * ```tsx
 * <Anchor asChild>
 * 	<button onClick={() => console.log("Button clicked")} type="button">
 * 		View Details
 * 	</button>
 * </Anchor>
 * ```
 */
declare const Anchor: React$2.ForwardRefExoticComponent<Omit<AnchorProps, "ref"> & React$2.RefAttributes<HTMLAnchorElement>>;
declare const button: (props?: ({
  size?: "small" | "medium" | "large" | "tiny" | null | undefined;
  kind?: "primary" | "secondary" | "tertiary" | null | undefined;
  color?: "brand" | "neutral" | "danger" | null | undefined;
} & ClassProp) | undefined) => string;
type ButtonVariantProps = VariantProps<typeof button>;
interface ButtonProps extends Omit<PrimitivePropsWithRef<"button">, "color"> {
  /**
   * The color variant of the button.
   * - "brand" - Prominent calls-to-action such as page-level CTAs and modal primary actions.
   * - "neutral" - Regular actions, suitable for most use cases.
   * - "danger" - Destructive actions that cannot be undone, such as delete or remove.
   * @defaultValue "neutral"
   */
  color?: ButtonVariantProps["color"];
  /**
   * Disables the button.
   */
  disabled?: boolean;
  /**
   * The kind of button.
   * - "primary" - The most important call-to-action on the page. Only one per context.
   * - "secondary" - Regular actions, suitable for most use cases.
   * - "tertiary" - Low-priority or supplemental actions.
   * @defaultValue "primary"
   */
  kind?: ButtonVariantProps["kind"];
  /**
   * The size of the button.
   * - "large" - The main call-to-action for a page or section.
   * - "medium" - Suitable for most use cases.
   * - "small" - Compact layouts with limited space or less significant actions.
   * - "tiny" - Dense layouts such as table cells where horizontal space is at a premium.
   * @defaultValue "medium"
   */
  size?: ButtonVariantProps["size"];
}
/**
 * A clickable element that triggers an action. Use specific verb + noun labels instead of vague text.
 * @param props - {@link ButtonProps}
 *
 * @llm Unlike native <button>, you will need to set type="submit" for it to act as a form submit button.
 * @llm For page-level CTAs, PageHeader slotActions, empty-state CTAs, and modal primary actions, use color="brand". A bare Button without color="brand" defaults to neutral, which is visually indistinguishable from secondary buttons.
 * @llm For destructive actions (Delete, Terminate, Revoke), use kind="primary" color="danger" and pair with a confirmation Modal.
 * @llm Status-changing actions that can be reversed (Decline, Reject, Archive, Dismiss) use kind="secondary" color="neutral", not color="danger". Danger styling on the secondary action fights the primary for attention; if the action can be undone or reversed, it is not destructive.
 * @llm Only one kind="primary" button per view (page, dialog, panel). Demote competing actions to secondary or tertiary (Hick's Law).
 * @llm Escalation path is Tertiary → Secondary → Primary. If a tertiary action is being missed, promote it to secondary — do not jump straight to primary.
 * @llm "Clear all" / "Clear filters" buttons should use kind="tertiary" and be placed alongside the filter chips they reset, not in the discovery toolbar.
 * @llm Stack order is context-dependent: commitment contexts (dialogs, form footers, page-level action bars) use `[tertiary] [secondary] [primary]` with primary trailing; CTA contexts (hero stacks, empty states, onboarding) use `[primary] [secondary] [tertiary]` with primary leading.
 * @llm For single action → Button; 2-4 related actions → ButtonGroup; navigation to URL → `Button asChild` wrapping `<a>`; inline prose navigation → Anchor.
 * @llm Use specific verb + noun labels ("Create cluster", "Delete workspace"); avoid generic verbs and keep to 2-3 words. If the action requires more than 3 words to describe, the surrounding UI should provide that context instead.
 * @llm Icon-only buttons must set `aria-label` describing the action. Reserve icon-only buttons for universally understood symbols (close, edit, delete, settings, kebab).
 * @llm Tertiary buttons need an affordance so they read as interactive. Options: trailing arrow/right caret for navigation-style actions, leading icon for utility verbs ("Edit", "Configure"), a color shift, or uppercase + letterspacing for dev-tool UIs. The default for hero contexts is the trailing arrow. A tertiary button with no icon, no color shift, and no label styling reads as a caption rather than an action.
 * @see {@link ButtonGroup}
 * @see {@link Anchor}
 *
 * @example
 * <caption>Basic Button</caption>
 * ```tsx
 * <Flex direction="col" gap="2">
 * 	<Button kind="primary" color="brand">
 * 		Button
 * 	</Button>
 * 	<Button kind="secondary" color="neutral">
 * 		Button
 * 	</Button>
 * 	<Button color="danger">Button</Button>
 * </Flex>
 * ```
 *
 * @example
 * <caption>With Icon Button</caption>
 * ```tsx
 * <Button>
 * 	<Document />
 * 	Add Document
 * </Button>
 * ```
 *
 * @example
 * <caption>Icon Only Button</caption>
 * ```tsx
 * <Button aria-label="Add document">
 * 	<Document />
 * </Button>
 * ```
 *
 * @example
 * <caption>Primary Action Button</caption>
 * Preferred default for primary actions: primary & brand. Use the primary action button for the most important action in a context. It should be used sparingly and only for the most important actions(submit, create, deploy, save, etc)
 * ```tsx
 * <Button color="brand">Create cluster</Button>
 * ```
 *
 * @example
 * <caption>Secondary Action Button</caption>
 * Preferred default for secondary actions: secondary & neutral. Use the secondary action button for actions that are not the most important in a context. It should be used for actions that are not the most important in a context(cancel, close, etc)
 * ```tsx
 * <Button kind="secondary">Deploy to GPU</Button>
 * ```
 *
 * @example
 * <caption>Tertiary Or Inline Action Button</caption>
 * Preferred default for tertiary/inline actions: tertiary & neutral. For example - edit, configure, kebab menu triggers, etc
 * ```tsx
 * <Button kind="tertiary">View logs</Button>
 * ```
 *
 * @example
 * <caption>Destructive Action Button</caption>
 * Preferred default for destructive actions: primary & danger. Use the destructive action button for actions that are destructive and cannot be undone. For status-changing or reversible actions (remove from list, unassign, archive), use secondary + neutral. Reserve `color="danger"` for irreversible destructive actions.
 * ```tsx
 * <Button color="danger">Delete cluster</Button>
 * ```
 *
 * @example
 * <caption>Button As Link</caption>
 * ```tsx
 * <Button asChild>
 * 	<a href="https://example.com">Link</a>
 * </Button>
 * ```
 *
 * @example
 * <caption>Button With Truncation</caption>
 * Button text should be short and concise, and if not text will wrap to the next line to ensure users can read the text. If you prefer to truncate the text you should set the `title` prop.
 * ```tsx
 * <Button title="Download the full results as a CSV file">
 * 	<span className="truncate">Download the full results as a CSV file</span>
 * </Button>
 * ```
 */
declare const Button: React$1.ForwardRefExoticComponent<Omit<ButtonProps, "ref"> & React$1.RefAttributes<HTMLButtonElement>>;
declare const badge: (props?: ({
  kind?: "solid" | "outline" | null | undefined;
  color?: "blue" | "gray" | "green" | "purple" | "red" | "teal" | "yellow" | null | undefined;
  size?: "small" | "medium" | "large" | null | undefined;
} & ClassProp) | undefined) => string;
type BadgeVariantProps = VariantProps<typeof badge>;
interface BadgeProps extends Omit<PrimitivePropsWithRef<"span">, "color"> {
  /**
   * Semantic color used to convey meaning at a glance.
   * @defaultValue "blue"
   */
  color?: BadgeVariantProps["color"];
  /**
   * Visual treatment of the badge.
   * - "outline" - De-emphasized contexts such as card metadata or secondary information.
   * - "solid" - Prominent status badges in detail views, side panels, and status bars.
   * @defaultValue "outline"
   */
  kind?: BadgeVariantProps["kind"];
  /**
   * Controls the badge's padding and text size. Reach for `small` in very dense layouts, and `large` for prominent status in detail views.
   * @defaultValue "medium"
   */
  size?: BadgeVariantProps["size"];
}
interface CardContentProps extends ComponentPropsWithRef<"div"> {}
/**
 * The body region of a composed card. Holds text, tags, and other primary content beneath any media.
 * @param props - {@link CardContentProps}
 */
declare const CardContent: React$1.ForwardRefExoticComponent<Omit<CardContentProps, "ref"> & React$1.RefAttributes<HTMLDivElement>>;
type SlottablePropsWithRef<E extends React$1.ElementType> = PrimitivePropsWithRef<E>;
declare const cardMedia: (props?: ({
  mediaTheme?: "dark" | "light" | null | undefined;
} & ClassProp) | undefined) => string;
interface CardMediaProps extends SlottablePropsWithRef<"div"> {
  /**
   * Forces the theme of overlaid `slotHeader` content so it contrasts with the media beneath.
   * - "dark" - For light or bright media; renders overlaid content in the dark theme.
   * - "light" - For dark media; renders overlaid content in the light theme.
   */
  mediaTheme?: VariantProps<typeof cardMedia>["mediaTheme"];
  /** Overlays content (typically the card title and actions) on top of the media. */
  slotHeader?: React$1.ReactNode;
}
/**
 * The media region of a composed card. Typically holds an image or video rendered above the content, with an optional overlaid header.
 * @param props - {@link CardMediaProps}
 */
declare const CardMedia: React$1.ForwardRefExoticComponent<Omit<CardMediaProps, "ref"> & React$1.RefAttributes<HTMLDivElement>>;
/**
 * Density variants for our density aware components.
 *
 * For any components that are density aware, you can use this constant to set the density of the
 * component. We should not assign a defaultVariant/defaultValue for density - `undefined` will allow
 * the component to inherit the density from the parent.
 *
 * @example
 * ```ts
 * const densityAwareComponentStyles = cva("nv-some-density-aware-component", {
 * 	variants: {
 * 		density: densityVariant,
 * 	},
 * });
 * ```
 */
declare const densityVariant: {
  compact: "nv-density-compact";
  standard: "nv-density-standard";
  spacious: "nv-density-spacious";
};
interface DensityVariantProps {
  /**
   * The "density" of the component. This affects the component padding. Set to `compact` for dense layouts, `standard` for general use, and `spacious` for marketing or onboarding surfaces.
   * @defaultValue "standard"
   */
  density?: keyof typeof densityVariant | null;
}
declare const cardRoot: (props?: ({
  density?: "compact" | "standard" | "spacious" | null | undefined;
  interactive?: boolean | null | undefined;
  kind?: "solid" | "float" | "gradient" | null | undefined;
  layout?: "horizontal" | "vertical" | null | undefined;
  selected?: boolean | null | undefined;
} & ClassProp) | undefined) => string;
interface CardRootProps extends PrimitivePropsWithRef<"div">, DensityVariantProps {
  /**
   * Adds hover and focus affordances to signal the card is clickable. Do not enable on non-clickable cards — it creates false affordance.
   * @defaultValue false
   */
  interactive?: boolean;
  /**
   * Visual treatment of the card.
   * - "solid" - General-purpose treatment with padded content and a hard border between media and content. Use for most cards.
   * - "gradient" - Same as "solid" but the media fades into the content. Use for promotional or hero-style cards.
   * - "float" - Pairs with `slotMedia` to render the media as a bordered tile while the content sits on the page background without padding. Use when the card should visually lift off the surface.
   * @defaultValue "solid"
   */
  kind?: VariantProps<typeof cardRoot>["kind"];
  /**
   * Orientation of the media and content.
   * - "vertical" - Media stacks above the content. Use when the media is the focal point and cards display in a grid.
   * - "horizontal" - Media sits to the left of the content. Use when the content outweighs the media and cards stack in a list.
   * @defaultValue "vertical"
   */
  layout?: VariantProps<typeof cardRoot>["layout"];
  /**
   * Applies the selected visual state, e.g. when the card represents the current choice in a list.
   * @defaultValue false
   * @llm Pair `selected` with a visible affordance for what selection means — a bulk action toolbar, a selection count indicator, or a primary action — so users understand the consequence of selecting a card.
   */
  selected?: boolean;
}
interface CardProps extends CardRootProps, Pick<CardMediaProps, "mediaTheme"> {
  /** Header content rendered above the body. When paired with `slotMedia`, it overlays the media instead. */
  slotHeader?: React$1.ReactNode;
  /** Media content rendered above the body, typically an image or video. Pair with `mediaTheme` when also using `slotHeader` so overlaid content remains readable. */
  slotMedia?: React$1.ReactNode;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    CardContent?: NativeElementAttributes<"div", typeof CardContent>;
    CardMedia?: NativeElementAttributes<"div", typeof CardMedia>;
  };
}
/**
 * A container that groups content, actions, and optional media about a single subject. Cards are fluid and grow to fill their container.
 * @param props - {@link CardProps}
 *
 * @llm The Card already has padding, so do not add padding to the Card component or to it's children.
 * @llm Card is for single-subject entity display (one cluster, one user, one model). Do not use Card as a general container on dashboards — use Panel instead. Card's interaction states and internal structure add unintended visual layering when used as a generic wrapper.
 * @llm Cards are fluid and grow to fill their container — do not set fixed widths. Control sizing through the parent layout (e.g. a responsive Grid). Apply `className="h-fit"` when a card should shrink to its content height.
 * @llm Identity test for Card vs Panel: if the container represents "a thing" with its own name and attributes, use Card; if it represents "a region of content", use Panel.
 * @llm Pair `selected` with a bulk-action toolbar or selection-count indicator so users understand what selection means.
 * @llm A Card is EITHER a single click target (set `interactive` and render via `asChild` as a link, button or label with containing hidden input) OR carries inline action buttons on its surface — never both. Combining them creates competing click targets and ambiguous focus order.
 * @llm When a Card is `interactive`, do not place other interactive elements (Button, Anchor, Menu, etc.) inside it. Nested click targets break the "whole card is one click" affordance.
 *
 * @see {@link Panel}
 * @see {@link Grid}
 *
 * @example
 * <caption>Basic Card That Fits Its Content</caption>
 * ```tsx
 * <Card className="h-fit">
 * 	<Badge kind="solid" color="gray">
 * 		Badge
 * 	</Badge>
 * 	<Text kind="body/bold/2xl">Header</Text>
 * 	<Text kind="body/regular/md">Lorem ipsum dolor sit amet</Text>
 * </Card>
 * ```
 *
 * @example
 * <caption>With Header Card</caption>
 * The header slot is rendered absolutely over the media. Use it to add a persistent label or badge above the body content. If there is no media, the header will be rendered above the body content. It is a flex container with a preset gap.
 * ```tsx
 * <Card
 * 	slotHeader={
 * 		<Flex gap="inherit">
 * 			<Badge>New</Badge>
 * 			<Badge>Alt</Badge>
 * 		</Flex>
 * 	}
 * >
 * 	Card body content
 * </Card>
 * ```
 *
 * @example
 * <caption>With Media Card</caption>
 * Use when the card needs a visual hero area like an image or video above or alongside the content. If the media rendered is dark or light, and you're using `slotHeader` in conjunction with it, you may want to set `mediaTheme` to ensure text and components are readable. This will set light/dark theme in the header slot to ensure sufficient contrast.
 * ```tsx
 * <Card
 * 	slotHeader={<span>Featured</span>}
 * 	slotMedia={<MediaImg />}
 * 	mediaTheme="light"
 * >
 * 	Card content below media
 * </Card>
 * ```
 *
 * @example
 * <caption>Model Card Example</caption>
 * Canonical browse-collection card anatomy for a single entity (model catalog, dataset gallery, template library, related-item strip). Slot order: header `Badge` (resource type) → publisher → title → description → topic `Tag` row (`outline` / `gray`) → footer stats (`text-placeholder`, left-aligned, never `justify="between"`). Every card in the collection must share this same slot structure — see the entity-cards pattern guide.
 * ```tsx
 * <Card slotHeader={<Badge kind="solid">Model</Badge>}>
 * 	<Flex direction="col" gap="density-sm">
 * 		<Text className="text-secondary" kind="label/regular/sm">
 * 			NVIDIA
 * 		</Text>
 * 		<Text kind="body/bold/xl">Nemotron 3 Super 120B</Text>
 * 		<Text kind="body/regular/md">
 * 			120B-parameter reasoning model optimized for enterprise RAG.
 * 		</Text>
 * 	</Flex>
 * 	<Flex gap="2" wrap="wrap">
 * 		<Tag kind="outline" color="gray" readOnly>
 * 			Reasoning
 * 		</Tag>
 * 		<Tag kind="outline" color="gray" readOnly>
 * 			English
 * 		</Tag>
 * 		<Tag kind="outline" color="gray" readOnly>
 * 			RAG
 * 		</Tag>
 * 	</Flex>
 * 	<Text className="text-placeholder" kind="label/regular/sm">
 * 		Updated 3d · 12.4k · 120B
 * 	</Text>
 * </Card>
 * ```
 *
 * @example
 * <caption>Interactive Card</caption>
 * Use when the entire card should be a clickable target, such as navigation or selection. Render as an `<a>` for navigation or a button for actions. Do not place other interactive elements (Button, Anchor, Menu, etc.) on the card surface — competing click targets are ambiguous. If you need inline actions per card instead, omit `interactive` and use the `With Actions` pattern below.
 * ```tsx
 * <Card asChild interactive>
 * 	<button type="button" onClick={handleClick}>
 * 		<Flex direction="col" gap="1">
 * 			<Text className="text-secondary" kind="label/bold/sm">
 * 				DeepSeek
 * 			</Text>
 * 			<Text kind="label/bold/md">DeepSeek-V3.1</Text>
 * 			<Text kind="body/regular/sm">
 * 				DeepSeek V3.1 Instruct is a hybrid AI model with fast reasoning, 128K
 * 				context, and strong tool use.
 * 			</Text>
 * 		</Flex>
 * 	</button>
 * </Card>
 * ```
 *
 * @example
 * <caption>With Actions</caption>
 * Use when the user needs to perform actions without navigating to a detail view, or when multiple distinct actions are available per card. The card itself is not a click target in this pattern — do not also enable `interactive`.
 * ```tsx
 * <Card>
 * 	Card content... Right aligned actions:
 * 	<Flex gap="2" justify="end" wrap="wrap">
 * 		<Button kind="tertiary">Export</Button>
 * 		<Button kind="secondary">Share</Button>
 * 	</Flex>
 * 	Or you can have actions stretch across the card:
 * 	<Flex gap="2" justify="end" wrap="wrap">
 * 		<Button kind="tertiary" style={{ flex: 1 }}>
 * 			Export
 * 		</Button>
 * 		<Button kind="secondary" style={{ flex: 1 }}>
 * 			Share
 * 		</Button>
 * 	</Flex>
 * 	Or for just a single action:
 * 	<Button color="brand" style={{ width: "100%" }}>
 * 		Share
 * 	</Button>
 * </Card>
 * ```
 *
 * @example
 * <caption>Cards In Responsive Grid</caption>
 * It's common to use cards in a responsive grid. Using the grid component, set a minimum column width and let the cards automatically resize to fill the available space.
 * ```tsx
 * <Grid colMinWidth="250px" gap="2">
 * 	<Card>Card 1</Card>
 * 	<Card>Card 2</Card>
 * 	<Card>Card 3</Card>
 * </Grid>
 * ```
 *
 * @example
 * <caption>Composed</caption>
 * ```tsx
 * <Flex direction="col" gap="1">
 * 	<CardRoot interactive={false} kind="solid">
 * 		<CardMedia mediaTheme="light" slotHeader={<Badge>Featured</Badge>}>
 * 			<MediaImg />
 * 		</CardMedia>
 * 		<CardContent>Composed card with media</CardContent>
 * 	</CardRoot>
 * 	<CardRoot>
 * 		<CardContent>
 * 			<div className="nv-card-content-header">
 * 				<Badge>Featured</Badge>
 * 			</div>
 * 			Composed card without media
 * 		</CardContent>
 * 	</CardRoot>
 * </Flex>
 * ```
 */
declare const Card$1: React$1.ForwardRefExoticComponent<Omit<CardProps, "ref"> & React$1.RefAttributes<HTMLDivElement>>;
declare const label$1: (props?: ({
  disabled?: boolean | null | undefined;
  size?: "small" | "medium" | "large" | null | undefined;
} & ClassProp) | undefined) => string;
interface LabelProps extends PrimitivePropsWithRef<"label"> {
  /** ID of the form control this label is associated with. */
  htmlFor?: string;
  /** Styles the label as disabled, for use only with disabled form controls. */
  disabled?: boolean;
  /**
   * Typographic size of the label.
   * @defaultValue "medium"
   */
  size?: VariantProps<typeof label$1>["size"];
}
/**
 * A text label for a form control. Prefer FormField for end-user forms; this primitive is intended for composing custom field layouts.
 * @param props - {@link LabelProps}
 *
 * @example
 * <caption>Basic Label</caption>
 * Use when labeling a form input to provide accessible context for the control.
 * ```tsx
 * <Label htmlFor="email">Email Address</Label>
 * ```
 *
 * @example
 * <caption>Disabled Label</caption>
 * Use when the associated input is disabled to visually communicate the inactive state.
 * ```tsx
 * <Label htmlFor="email" disabled aria-disabled="true">
 * 	Email Address
 * </Label>
 * ```
 *
 * @example
 * <caption>Small Label</caption>
 * Use in compact layouts or alongside small-sized inputs where space is limited.
 * ```tsx
 * <Label htmlFor="email" size="small">
 * 	Email Address
 * </Label>
 * ```
 *
 * @example
 * <caption>Label With Icon</caption>
 * Use when the label needs an inline help affordance. Label accepts arbitrary children so an icon or tooltip can sit alongside the text.
 * ```tsx
 * <Label htmlFor="email">
 * 	Email Address
 * 	<Tooltip slotContent="We use this to send account confirmations.">
 * 		<InfoCircle />
 * 	</Tooltip>
 * </Label>
 * ```
 */
declare const Label: React$1.ForwardRefExoticComponent<Omit<LabelProps, "ref"> & React$1.RefAttributes<HTMLLabelElement>>;
/** Checked state of a checkbox, including the tri-state `"indeterminate"` value. */
type CheckedState = boolean | "indeterminate";
interface CheckboxInputProps extends Omit<ComponentPropsWithRef<"input">, "defaultChecked" | "checked" | "type"> {
  /**
   * Initial checked state when the checkbox is uncontrolled.
   */
  defaultChecked?: boolean;
  /**
   * Controlled checked state; pair with `onCheckedChange` to handle updates.
   */
  checked?: CheckedState;
  /**
   * Called when the checked state changes.
   */
  onCheckedChange?: (checked: CheckedState) => void;
  /**
   * Renders the checkbox in an error state.
   */
  error?: boolean;
  /**
   * Disables interaction with the checkbox.
   */
  disabled?: boolean;
  /**
   * Form field name submitted with the checkbox value as a name/value pair.
   */
  name?: string;
  /**
   * Requires the checkbox to be checked for form submission.
   */
  required?: boolean;
  /**
   * ID of the form element to associate with, allowing the checkbox to be rendered outside that form.
   */
  form?: string;
}
/**
 * The `<input type="checkbox">` element of a composed checkbox, including indeterminate-state handling and form integration.
 * @param props - {@link CheckboxInputProps}
 */
declare const CheckboxInput: React$1.ForwardRefExoticComponent<Omit<CheckboxInputProps, "ref"> & React$1.RefAttributes<HTMLInputElement>>;
declare const checkboxRoot: (props?: ({
  labelSide?: "left" | "right" | null | undefined;
} & ClassProp) | undefined) => string;
interface CheckboxRootProps extends PrimitivePropsWithRef<"div"> {
  /**
   * Side of the checkbox the label is rendered on.
   * @defaultValue "right"
   */
  labelSide?: VariantProps<typeof checkboxRoot>["labelSide"];
}
interface PropsFromRoot$4 extends Omit<CheckboxRootProps, AttributesFor<"input"> | "ref"> {}
interface PropsFromInput$3 extends Pick<CheckboxInputProps, Extract<keyof CheckboxInputProps, AttributesFor<"input">> | "onCheckedChange" | "error"> {}
interface CheckboxProps extends PropsFromRoot$4, PropsFromInput$3 {
  /**
   * Label rendered next to the checkbox and automatically associated with it for clicks and assistive tech.
   */
  slotLabel?: React$1.ReactNode;
  /**
   * Native HTML attributes forwarded to the internal composed components.
   */
  attributes?: {
    CheckboxInput?: NativeElementAttributes<"input", typeof CheckboxInput>;
    Label?: NativeElementAttributes<"label", typeof Label>;
  };
}
declare const inputShell: (props?: ({
  kind?: "flat" | "floating" | null | undefined;
  layout?: "horizontal" | "vertical" | null | undefined;
  size?: "small" | "medium" | "large" | null | undefined;
  withValidation?: boolean | null | undefined;
} & ClassProp) | undefined) => string;
type InputShellVariantProps = VariantProps<typeof inputShell>;
interface InputShellProps extends ComponentPropsWithoutRef<"div"> {
  /** Render-as-child slot (from Radix Primitive) */
  asChild?: boolean;
  /**
   * When true, the input will not redirect focus to the input when clicked.
   * @defaultValue false
   */
  disableFocusRedirect?: boolean;
  /**
   * Visual treatment of the shell. `"flat"` has a border and background; `"floating"` is borderless and transparent.
   * @defaultValue "flat"
   */
  kind?: InputShellVariantProps["kind"];
  /**
   * Axis along which slotted content is arranged inside the shell. `"vertical"` stacks slots in a column with auto height block padding.
   * @defaultValue "horizontal"
   */
  layout?: InputShellVariantProps["layout"];
  /**
   * Overall height and typography of the shell.
   * @defaultValue "medium"
   */
  size?: InputShellVariantProps["size"];
  /** Surfaces success and error styling automatically based on the inner input's `:user-valid` and `:user-invalid` states. */
  withValidation?: boolean;
}
/**
 * Mixin interface for components whose prop types are intersected with input-shell status. Not a component or standalone prop — it is composed into other component prop types (for example, upload triggers) and is not imported or used directly.
 */
interface WithInputShellStatus {
  /**
   * The status of the input. Use `withValidation` to automatically apply success/error states based
   * on `:user-valid` and `:user-invalid` pseudo classes.
   */
  status?: "success" | "error";
}
/**
 * A clear-the-value button rendered inside dismissible inputs.
 * @param props - {@link ButtonProps}
 */
declare const InputDismissButton: React$2.ForwardRefExoticComponent<Omit<ButtonProps, "ref"> & React$2.RefAttributes<HTMLButtonElement>>;
declare const dividerElement: (props?: ({
  orientation?: "horizontal" | "vertical" | null | undefined;
  width?: "small" | "medium" | "large" | null | undefined;
} & ClassProp) | undefined) => string;
type DividerElementVariantProps = VariantProps<typeof dividerElement>;
interface DividerElementProps extends PrimitivePropsWithRef<"div"> {
  /**
   * Axis the separator runs along.
   * @defaultValue "horizontal"
   */
  orientation?: DividerElementVariantProps["orientation"];
  /**
   * Thickness of the separator line. Step up to `"medium"` only when separating major page sections that need extra visual weight
   * @defaultValue "small"
   */
  width?: DividerElementVariantProps["width"];
}
/**
 * A horizontal or vertical separator line with configurable thickness.
 * @param props - {@link DividerElementProps}
 */
declare const DividerElement: React$2.ForwardRefExoticComponent<Omit<DividerElementProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface DividerRootProps extends PrimitivePropsWithRef<"div">, Pick<PrimitiveComponentProps, "padding" | "paddingX" | "paddingY" | "paddingTop" | "paddingRight" | "paddingBottom" | "paddingLeft"> {}
/**
 * The outermost element of a composed divider. Applies padding tokens around the separator line.
 * @param props - {@link DividerRootProps}
 */
declare const DividerRoot: React$2.ForwardRefExoticComponent<Omit<DividerRootProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface DividerProps extends DividerElementProps, Pick<PrimitiveComponentProps, "asChild" | "padding" | "paddingX" | "paddingY" | "paddingTop" | "paddingRight" | "paddingBottom" | "paddingLeft"> {
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DividerRoot?: NativeElementAttributes<"div", typeof DividerRoot>;
    DividerElement?: NativeElementAttributes<"div", typeof DividerElement>;
  };
}
/**
 * A horizontal or vertical line that visually separates groups of content. Use to create breaks between sections, list items, or toolbar regions where a heading or whitespace alone would be too subtle.
 * @param props - {@link DividerProps}
 *
 * @llm Prefer spacing or a section heading over a Divider inside cards and between form sections — lines there read as visual noise. Reserve Divider for breaks between distinct regions (sidebar sections, list groups, panel areas). For form sections, replace dividers with a `Text kind="title/sm"` heading and spacing.
 * @llm Use `orientation="vertical"` only for toolbars or side-by-side layouts; default to horizontal.
 * @llm Default to `width="small"`; reserve `"medium"` for more visual weight when separating major page sections.
 *
 * @see {@link Stack}
 *
 * @example
 * <caption>Basic Divider</caption>
 * ```tsx
 * <Flex className="w-full" direction="col" gap="3">
 * 	<Divider />
 * 	<Divider width="medium" />
 * 	<Divider width="large" />
 * </Flex>
 * ```
 *
 * @example
 * <caption>With Text</caption>
 * ```tsx
 * <Flex align="center" gap="density-lg">
 * 	<Divider />
 * 	<Text kind="label/semibold/md">Text</Text>
 * 	<Divider />
 * </Flex>
 * ```
 *
 * @example
 * <caption>Vertical Divider</caption>
 * ```tsx
 * <Divider orientation="vertical" />
 * ```
 *
 * @example
 * <caption>Composed</caption>
 * ```tsx
 * <DividerRoot paddingY="2">
 * 	<DividerElement orientation="horizontal" width="medium" />
 * </DividerRoot>
 * ```
 */
declare const Divider: React$2.ForwardRefExoticComponent<Omit<DividerProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface RadioGroupInputProps extends Omit<React$1.ComponentPropsWithRef<"input">, "type" | "value"> {
  /** Value submitted with the form when this option is selected. */
  value: string;
  /** Marks this individual option as destructive. Pair with destructive copy on the label. */
  danger?: boolean;
  /** Renders this option in an error state. */
  error?: boolean;
  /**
   * Shows the radio indicator. When `false`, the input remains in the DOM but is visually hidden so tile-style items can convey the selected state themselves.
   * @defaultValue true
   */
  showIndicator?: boolean;
  /** Called with this option's value when it becomes the selected radio in the group. */
  onValueChange?: (value: string) => void;
}
/**
 * A native `<input type="radio">` styled with KUI tokens. Inherits the group's `name`, `value`, and shared attributes from its surrounding root.
 * @param props - {@link RadioGroupInputProps}
 */
declare const RadioGroupInput: React$1.ForwardRefExoticComponent<Omit<RadioGroupInputProps, "ref"> & React$1.RefAttributes<HTMLInputElement>>;
interface RadioGroupItemProps extends Omit<PrimitivePropsWithRef<"label">, "children" | "value" | "defaultValue"> {
  /** Label for the radio input. */
  children: React.ReactNode;
}
/**
 * A `<label>` that wraps a radio input and its visible content so clicking anywhere on the item toggles the input.
 * @param props - {@link RadioGroupItemProps}
 */
declare const RadioGroupItem: React$2.ForwardRefExoticComponent<Omit<RadioGroupItemProps, "ref"> & React$2.RefAttributes<HTMLLabelElement>>;
declare const radioGroupRoot: (props?: ({
  error?: boolean | null | undefined;
  kind?: "default" | "tile" | null | undefined;
  orientation?: "horizontal" | "vertical" | null | undefined;
} & ClassProp) | undefined) => string;
type RadioGroupRootVariantProps = VariantProps<typeof radioGroupRoot>;
interface RadioGroupRootProps extends PrimitivePropsWithRef<"div"> {
  /** Shared HTML `name` applied to each radio input. Falls back to the surrounding `FormField`'s `name` when omitted. */
  name?: string;
  /** Initial selected value when uncontrolled. */
  defaultValue?: string;
  /** Controlled selected value. When provided, the consumer is responsible for updating it in response to `onValueChange`. */
  value?: string;
  /** Called when the selected value changes. */
  onValueChange?: (value: string) => void;
  /** Disables every radio in the group. */
  disabled?: boolean;
  /** Marks the group as required so the surrounding form refuses to submit without a selection. */
  required?: boolean;
  /** Renders every radio in an error state. */
  error?: boolean;
  /**
   * Visual treatment for the group and its items.
   * @defaultValue "default"
   */
  kind?: RadioGroupRootVariantProps["kind"];
  /**
   * Layout direction. Determines both visual arrangement and arrow-key navigation order.
   * @defaultValue "vertical"
   */
  orientation?: "horizontal" | "vertical";
}
interface MenuCheckboxItemProps extends Omit<React$1.LabelHTMLAttributes<HTMLLabelElement>, "checked" | "defaultChecked" | "onSelect"> {
  /** Controlled checked state. Pair with `onCheckedChange`. */
  checked?: CheckedState;
  /** Called with the new checked state when the item is toggled. */
  onCheckedChange?: (checked: CheckedState) => void;
  /**
   * Initial checked state when uncontrolled.
   * @defaultValue false
   */
  defaultChecked?: CheckedState;
  /** Custom indicator element rendered in place of the default checkbox. */
  slotControl?: ReactNode;
  /** Content rendered before the label, typically an icon. */
  slotStart?: ReactNode;
  /** Content rendered after the label, such as a shortcut hint or trailing icon. */
  slotEnd?: ReactNode;
  /** Form `name` for the hidden input that submits with the surrounding form when the item is checked. */
  name?: string;
  /** Submit value for the hidden input, also used as a fallback for search filtering and the `onSelect` detail. */
  value?: string;
  /** Disables the item so it cannot be toggled or focused. */
  disabled?: boolean;
  /**
   * When true, applies error styling to the checkbox control.
   */
  error?: boolean;
  /** Applies the destructive visual treatment for actions such as delete or revoke. */
  danger?: boolean;
  /** Additional class names appended to the rendered element. */
  className?: string;
  /** Label content rendered next to the indicator. */
  children?: ReactNode;
  /** String used to match this item when the surrounding menu is filterable. Pass `null` to opt the item out of filtering. */
  filterValue?: string | null;
  /** Called when the item is activated via click, Enter, or Space. */
  onSelect?: (event: Event) => void;
  /**
   * Callback fired on click events. Note that `onSelect` is fired before `onClick` and
   * is the preferable event for handling selection logic.
   * @param event - The React mouse event.
   */
  onClick?: React$1.MouseEventHandler<HTMLLabelElement>;
}
declare const MenuCheckboxItem: React$1.ForwardRefExoticComponent<MenuCheckboxItemProps & React$1.RefAttributes<HTMLLabelElement>>;
interface MenuHeadingProps extends PrimitivePropsWithRef<"div"> {}
/**
 * A non-interactive heading that labels a group of items inside a menu.
 * @param props - {@link MenuHeadingProps}
 */
declare const MenuHeading: React$2.ForwardRefExoticComponent<Omit<MenuHeadingProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface MenuItemProps extends Omit<SlottablePropsWithRef<"button">, "onSelect" | "defaultChecked"> {
  /** Applies the destructive visual treatment for actions such as delete or revoke. */
  danger?: boolean;
  /** Disables the item so it cannot be selected or focused. */
  disabled?: boolean;
  /** String used to match this item when the surrounding menu is filterable. Pass `null` to opt the item out of filtering. */
  filterValue?: string | null;
  /** Called when the item is activated via click, Enter, or Space. */
  onSelect?: (event: Event) => void;
  /** Custom control element rendered before the label, replacing the default checkbox or radio indicator slot. */
  slotControl?: ReactNode;
  /** Content rendered before the label, typically an icon. If you set this to `true` it will take up the space an icon would take without rendering one, useful to align non-icon items with icon items */
  slotStart?: ReactNode;
  /** Content rendered after the label, such as a shortcut hint or trailing icon. */
  slotEnd?: ReactNode;
  /** Value associated with the item, surfaced in selection events and used as the form submit value when `formAction` is set. */
  value?: string;
  /** URL the item submits to when used as a form submit button. */
  formAction?: string;
  /**
   * HTTP method used when `formAction` is set.
   * @defaultValue "post"
   */
  formMethod?: "get" | "post";
}
/**
 * A single selectable item inside a menu. Use for actions, navigation targets, and form submissions.
 * @param props - {@link MenuItemProps}
 */
declare const MenuItem: React$2.ForwardRefExoticComponent<Omit<MenuItemProps, "ref"> & React$2.RefAttributes<HTMLButtonElement>>;
interface MenuRadioGroupProps extends PrimitivePropsWithRef<"fieldset"> {
  /** Shared form `name` for the radios in this group. Used as the fallback accessible name when no heading is provided. */
  name: string;
  /**
   * Selected value on first render when the group is uncontrolled.
   * @defaultValue ""
   */
  defaultValue?: string;
  /** Controlled selected value. Pair with `onValueChange`. */
  value?: string;
  /** Called with the new value when the selected radio changes. */
  onValueChange?: (value: string) => void;
  /** Disables every radio in the group. */
  disabled?: boolean;
  /** Marks the group as required so the owning form cannot be submitted until one radio is selected. */
  required?: boolean;
  /** Renders the group in an error visual state. */
  error?: boolean;
  /** Heading rendered above the radios, also serving as the group's accessible name. */
  slotHeading?: ReactNode;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    MenuHeading?: NativeElementAttributes<"div", typeof MenuHeading>;
  };
}
/**
 * A set of mutually exclusive options rendered inside a menu, of which one can be selected at a time.
 * @param props - {@link MenuRadioGroupProps}
 */
declare const MenuRadioGroup: React$2.ForwardRefExoticComponent<Omit<MenuRadioGroupProps, "ref"> & React$2.RefAttributes<HTMLFieldSetElement>>;
interface MenuRadioGroupItemProps {
  /** Unique value submitted with the form and used to track selection. Avoid using an empty string. */
  value: string;
  /** Disables the item so it cannot be selected or focused. */
  disabled?: boolean;
  /** Marks the item as required so the owning form cannot be submitted unless it is selected. */
  required?: boolean;
  /** Custom indicator element rendered in place of the default radio button. */
  slotControl?: ReactNode;
  /** Content rendered before the label, typically an icon. */
  slotStart?: ReactNode;
  /** Content rendered after the label, such as a shortcut hint or trailing icon. */
  slotEnd?: ReactNode;
  /** Additional class names appended to the rendered element. */
  className?: string;
  /** Label content rendered next to the indicator. */
  children?: ReactNode;
  /** String used to match this item when the surrounding menu is filterable. Pass `null` to opt the item out of filtering. */
  filterValue?: string | null;
  /** Called when the item is activated via click, Enter, or Space. */
  onSelect?: (event: Event) => void;
  /** Applies the destructive visual treatment for actions such as delete or revoke. */
  danger?: boolean;
}
/**
 * A radio option inside a menu radio group. Selecting it updates the group's value to this item's `value`.
 * @param props - {@link MenuRadioGroupItemProps}
 */
declare const MenuRadioGroupItem: React$2.ForwardRefExoticComponent<MenuRadioGroupItemProps & React$2.RefAttributes<HTMLInputElement>>;
interface MenuRootProps extends PrimitivePropsWithRef<"menu">, DensityVariantProps {
  /**
   * CSS selector used by the menu's roving focus to locate focusable items.
   * @defaultValue FOCUSABLE_MENU_ITEM_SELECTOR
   */
  itemSelector?: string;
  /** Called when the menu has been scrolled to the bottom. Use to lazy-load additional items. */
  onScrollToBottom?: () => void;
  /**
   * Axis along which the arrow keys move focus between items.
   * @defaultValue "vertical"
   */
  orientation?: "horizontal" | "vertical";
  /**
   * Whether arrow-key navigation wraps from the last item back to the first.
   * @defaultValue true
   */
  loop?: boolean;
  /**
   * Marks the menu as filterable so an empty state can be rendered when filtering excludes every item.
   * @defaultValue false
   */
  filterable?: boolean;
}
/**
 * The describing elements a control can reference. Used by the self-healing
 * registration so the control's ARIA only points at elements that actually mount.
 */
type FormFieldAriaSlot = "label" | "helper" | "info";
interface FormFieldContextType {
  /**
   * The ID of the form field (for connecting label and input). The value of this attribute must be unique.
   */
  id?: string;
  /**
   * Name of the element. Used to identify fields in form submits.
   */
  name?: string;
  status?: "success" | "error";
  /**
   * The `id` assigned to the field's label element. Composed label components (e.g. `FormFieldLabel`) read this to wire themselves to the control.
   */
  labelId?: string;
  /**
   * The `id` assigned to the field's helper element. `FormFieldHelper` reads this so the control's `aria-describedby` resolves to the helper text.
   */
  helperId?: string;
  /**
   * The `id` assigned to the field's supplementary info element (the popover content mirror used for `aria-details`).
   */
  infoId?: string;
  /**
   * The aria-describedby value for the form control. Identifies the element that describes the element on which the attribute is set.
   * You can customize this by passing an ID to the `FormFieldHelper` element via the attributes API.
   * @example
   * ```tsx
   * <FormField
   *   attributes={{
   *     FormFieldHelper: { id: "helper-id" },
   *   }}
   * ><TextInput /></FormField>
   * ```
   */
  "aria-describedby"?: string;
  /**
   * The aria-labelledby value for the form control. Identifies the element that labels the element it is applied to.
   * You can customize this by passing an ID to the `Label` element via the attributes API.
   * @example
   * ```tsx
   * <FormField
   *   attributes={{
   *     Label: { id: "label-id" },
   *   }}
   * ><TextInput /></FormField>
   * ```
   */
  "aria-labelledby"?: string;
  /**
   * The aria-details value for the form control. Identifies the element that provide additional information related to the object.
   * You can customize this by passing an ID to the `TooltipTrigger` element via the attributes API.
   * @example
   * ```tsx
   * <FormField
   *   attributes={{
   *     TooltipTrigger: { id: "details-id" },
   *   }}
   * ><TextInput /></FormField>
   * ```
   */
  "aria-details"?: string;
  /**
   * When true, indicates that the user is required to fill out this field. Used to determine whether or not the asterisk is shown next to the label.
   *
   * By default, this will automatically determine if the form field contains a `:required` input, otherwise it will be manually controlled.
   */
  required?: boolean;
  /**
   * @internal Lets a describing child (label, helper, info) report that it has
   * mounted with the given `id`. `FormFieldRoot` uses these registrations to
   * heal the control's `aria-*` after hydration so they only reference elements
   * that actually exist. Returns a cleanup that unregisters on unmount.
   */
  registerAria?: (slot: FormFieldAriaSlot, id: string) => () => void;
}
type SafeHrefProp<T extends {
  href?: string | null;
}> = Omit<T, "href"> & {
  href: string;
};
type ExcludedInputAttributes = Exclude<AttributesFor<"input">, "size">;
interface PropsFromRoot$3 extends WithInputShellStatus, Omit<InputShellProps, ExcludedInputAttributes> {}
interface PropsFromInput$2 extends Pick<ComponentPropsWithRef<"input">, ExcludedInputAttributes> {}
interface TextInputProps extends PropsFromRoot$3, PropsFromInput$2 {
  /**
   * Renders a dismiss button that clears the input and returns focus to it.
   * @llm Prefer `false` for form fields where the user edits in place.
   * @defaultValue `true` when `type="search"`, otherwise `false`
   */
  dismissible?: boolean;
  /** Overrides the default behavior of the dismiss button. */
  onDismiss?: MouseEventHandler<HTMLButtonElement>;
  /** Called with the new string value when the input content changes. Pair with `value` for controlled state. */
  onValueChange?: (value: string, event: ChangeEvent<HTMLInputElement>) => void;
  /**
   * Content rendered before the input. Moves above the input when `layout="vertical"`. For `type="time"`, `type="date"`, and `type="datetime-local"`, a picker button is always rendered first and this content renders after it.
   */
  slotStart?: ReactNode;
  /** Content rendered after the input. Moves below the input when `layout="vertical"`. */
  slotEnd?: ReactNode;
  /** Controlled input value. Pair with `onChange` or `onValueChange` to keep it editable. */
  value?: string;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    Input?: ComponentPropsWithRef<"input">;
    InputDismissButton?: React.ComponentProps<typeof InputDismissButton>;
  };
}
/**
 * A single-line input for text, numbers, or special symbols. Commonly used in forms; switch to TextArea when the response is likely to wrap.
 * @param props - {@link TextInputProps}
 *
 * @llm Pick the right primitive before building: TextInput for single-line free text, TextArea when users will regularly need multiple lines, Combobox for a text input paired with a selectable list, InputShell when composing a custom input from scratch. One-line test: if the expected input will almost always fit on a single line, use TextInput.
 * @llm When used as a search input, always include a MagnifyingGlass icon via slotStart for affordance.
 * @llm For search inputs, set `dismissible={true}` and run client-side filtering on keystroke; do not add a separate "Search" button next to the input unless the search is server-initiated. For form fields where the user edits in place, set `dismissible={false}`.
 * @llm Prefer wrapping in FormField (handles label, helper text, and status propagation) and reach for the `status` prop directly only on standalone inputs.
 * @llm Set `type` to match the data semantics (`"email"`, `"url"`, `"tel"`, `"password"`, `"search"`) so the browser provides the right keyboard, autofill hints, and validation.
 * @llm Prefer DatePicker for date selections
 * @llm Placeholder text is not a label substitute. Wrap in FormField with `slotLabel` or set `aria-label` for standalone inputs.
 * @llm Avoid placing interactive elements in `slotEnd` — use `dismissible` for clearing or render actions outside the input.
 * @see {@link FormField}
 * @see {@link TextArea}
 * @see {@link InputShell}
 *
 * @example
 * <caption>Basic Text Input</caption>
 * ```tsx
 * <Flex direction="col" gap="density-md">
 * 	<TextInput aria-label="Greeting" defaultValue="Hello, world!" />
 * 	<TextInput aria-label="Time" type="time" />
 * </Flex>
 * ```
 *
 * @example
 * <caption>With Slots Text Input</caption>
 * Use slotStart for affordance icons (e.g. a magnifying glass for search) and slotEnd for trailing hints like result counts or unit labels.
 * ```tsx
 * <TextInput
 * 	aria-label="Username"
 * 	slotStart={<Bell />}
 * 	slotEnd={<span>(7)</span>}
 * 	defaultValue="admin"
 * />
 * ```
 *
 * @example
 * <caption>Search Text Input</caption>
 * Standard search input: `type="search"` and `dismissible` so users can quickly clear the query. Pair with a leading magnifying-glass icon in `slotStart`. Provide `aria-label` when the input is not inside a FormField.
 * ```tsx
 * <TextInput
 * 	type="search"
 * 	aria-label="Search"
 * 	placeholder="Search..."
 * 	dismissible
 * 	slotStart={<SearchIcon />}
 * />
 * ```
 *
 * @example
 * <caption>Dismissible Text Input</caption>
 * Use when the user needs a quick way to clear the field, such as search inputs.
 * ```tsx
 * <TextInput
 * 	aria-label="Clearable value"
 * 	dismissible
 * 	defaultValue="Clearable value"
 * />
 * ```
 *
 * @example
 * <caption>Read Only Text Input</caption>
 * Use readOnly (instead of disabled) when the value should remain visible, focusable, and copyable but not editable — common in summary or review screens.
 * ```tsx
 * <TextInput aria-label="Read-only value" readOnly value="Read-only value" />
 * ```
 *
 * @example
 * <caption>Disabled Text Input</caption>
 * Use when the field value is locked by external conditions and cannot be changed.
 * ```tsx
 * <TextInput aria-label="Disabled value" disabled defaultValue="Cannot edit" />
 * ```
 *
 * @example
 * <caption>Size Text Input</caption>
 * Match the input size to surrounding controls — small for dense tables/toolbars, large for hero search bars or onboarding forms.
 * ```tsx
 * <Stack gap="density-md">
 * 	<TextInput size="small" placeholder="Small" />
 * 	<TextInput size="medium" placeholder="Medium" />
 * 	<TextInput size="large" placeholder="Large" />
 * </Stack>
 * ```
 *
 * @example
 * <caption>Status Text Input</caption>
 * Set `status` to drive validation styling manually — `error` for failed validation or `success` to confirm a passing value. Reach for `withValidation` when native HTML constraints can drive the state automatically.
 * ```tsx
 * <Stack gap="density-md">
 * 	<TextInput
 * 		aria-label="Invalid email"
 * 		status="error"
 * 		defaultValue="invalid-email"
 * 	/>
 * 	<TextInput
 * 		aria-label="Valid email"
 * 		status="success"
 * 		defaultValue="valid@example.com"
 * 	/>
 * </Stack>
 * ```
 *
 * @example
 * <caption>With Validation Text Input</caption>
 * Use withValidation to drive success/error styling automatically from native HTML constraints (`required`, `pattern`, `type="email"`, etc.) via `:user-valid` and `:user-invalid` — no controlled state needed.
 * ```tsx
 * <TextInput
 * 	withValidation
 * 	required
 * 	type="email"
 * 	placeholder="name@example.com"
 * />
 * ```
 *
 * @example
 * <caption>Date Text Input</caption>
 * Use type="date" or type="datetime-local" to render a calendar button that opens the browser's native picker via `showPicker()`.
 * ```tsx
 * <Stack gap="density-md">
 * 	<TextInput aria-label="Date" type="date" />
 * 	<TextInput aria-label="Date and time" type="datetime-local" />
 * </Stack>
 * ```
 *
 * @example
 * <caption>Time Text Input</caption>
 * Use for time selection with a native picker button. Set step={60} to hide seconds.
 * ```tsx
 * <TextInput aria-label="Time" type="time" step={60} />
 * ```
 *
 * @example
 * <caption>With Form Field Text Input</caption>
 * Wrap in FormField to attach a label, helper text, and required indicator. FormField pipes `id`, `name`, `required`, and validation status into the input via context.
 * ```tsx
 * <FormField
 * 	name="username"
 * 	slotLabel="Username"
 * 	slotHelp="3–20 characters, letters and numbers only."
 * 	required
 * >
 * 	<TextInput placeholder="Enter your username" />
 * </FormField>
 * ```
 *
 * @example
 * <caption>Controlled Text Input</caption>
 * Use controlled mode when state lives outside the input — e.g. when syncing with form libraries or URL state.
 * ```tsx
 * () => {
 * 	const [value, setValue] = useState("");
 * 	return (
 * 		<TextInput
 * 			value={value}
 * 			onValueChange={setValue}
 * 			placeholder="Type something"
 * 		/>
 * 	);
 * }
 * ```
 *
 * @example
 * <caption>Debounced Text Input</caption>
 * Combine with `useDebounce` to throttle expensive work like API calls or filtering driven by the input value.
 * ```tsx
 * () => {
 * 	const [value, setValue] = useState("");
 * 	const debounced = useDebounce(value, 500);
 * 	return (
 * 		<Stack gap="density-md">
 * 			<TextInput
 * 				value={value}
 * 				onValueChange={setValue}
 * 				placeholder="Search..."
 * 			/>
 * 			<Text>Debounced: {debounced}</Text>
 * 		</Stack>
 * 	);
 * }
 * ```
 *
 * @example
 * <caption>Composed</caption>
 * Drop down to InputShell + a raw `<input>` when you need full control over slot content, ARIA, or non-standard layouts that TextInput's API doesn't expose.
 * ```tsx
 * <InputShell>
 * 	<Bell />
 * 	<input type="text" placeholder="Composed text input" />
 * </InputShell>
 * ```
 */
declare const TextInput: React$2.ForwardRefExoticComponent<TextInputProps & React$2.RefAttributes<HTMLInputElement>>;
interface MenuSearchProps extends TextInputProps {}
/**
 * A search input rendered at the top of a filterable menu, wired up to the surrounding filter context.
 * @param props - {@link MenuSearchProps}
 */
declare const MenuSearch: React$2.ForwardRefExoticComponent<MenuSearchProps & React$2.RefAttributes<HTMLInputElement>>;
/**
 * The value for the menu search context.
 */
interface MenuSearchValueContextStore {
  matchFn: "disable" | ((matchTerm: string, value: string) => boolean);
  value: string | undefined;
}
interface MenuSearchProviderProps {
  /** Menu content placed under the search context. */
  children: React$1.ReactNode;
  /** Initial search value when the provider is uncontrolled. */
  defaultValue?: string;
  /** Custom matcher used to decide whether an item is visible for the current search value. Defaults to a case- and diacritic-insensitive substring match. */
  matchFn?: MenuSearchValueContextStore["matchFn"];
  /** Called with the new search value when it changes. */
  onValueChange?: (value: string) => void;
  /** Controlled search value. Pair with `onValueChange`. */
  value?: string;
}
interface MenuSectionProps extends PrimitivePropsWithRef<"div"> {
  /** Heading rendered above the section, also serving as the group's accessible name when set. */
  slotHeading?: ReactNode;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    MenuHeading?: NativeElementAttributes<"div", typeof MenuHeading>;
  };
}
/**
 * A labelled group of items inside a menu, used to cluster related actions or options under a shared heading.
 * @param props - {@link MenuSectionProps}
 */
declare const MenuSection: React$2.ForwardRefExoticComponent<Omit<MenuSectionProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface BaseMenuItem extends Pick<MenuItemProps, "children" | "filterValue" | "disabled" | "onSelect" | "slotStart" | "slotEnd" | "danger"> {
  /** String used to match this entry when the menu is filterable. Pass `null` to opt the entry out of filtering. */
  filterValue?: string | null;
}
interface MenuDefaultItemEntry extends BaseMenuItem, Pick<MenuItemProps, "slotStart" | "slotEnd" | "formAction" | "formMethod" | "value"> {
  /**
   * Discriminator selecting a standard selectable menu item.
   * @defaultValue "default"
   */
  kind?: "default";
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    MenuItem?: NativeElementAttributes<"button", typeof MenuItem>;
  };
}
interface MenuCheckboxItemEntry extends BaseMenuItem, Pick<MenuCheckboxItemProps, "defaultChecked" | "checked" | "onCheckedChange" | "name" | "value">, Pick<CheckboxInputProps, "error">, Pick<MenuCheckboxItemProps, "slotControl" | "slotStart" | "slotEnd"> {
  /** Discriminator selecting a checkbox menu item that toggles a boolean state. */
  kind: "checkbox";
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    MenuCheckboxItem?: NativeElementAttributes<"label", typeof MenuCheckboxItem>;
  };
}
interface MenuRadioItemEntry extends Omit<BaseMenuItem, "danger">, Pick<RadioGroupInputProps, "danger" | "value" | "required"> {
  /** Reserved for future entry kinds; radio items appear only inside a `MenuRadioGroupEntry`. */
  kind?: never;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    MenuRadioGroupItem?: NativeElementAttributes<"button", typeof MenuRadioGroupItem>;
  };
}
interface MenuRadioGroupEntry extends Pick<RadioGroupRootProps, "defaultValue" | "value" | "onValueChange" | "disabled" | "required" | "error"> {
  /** Shared form `name` for the radios in this group; required for form submission. */
  name: string;
  /** Discriminator selecting a radio group of mutually exclusive items. */
  kind: "radio";
  /**
   * Visual treatment of the selection indicator. `"radio"` renders radio buttons; `"check"` renders a checkmark beside the selected item.
   * @defaultValue "radio"
   */
  radioKind?: "radio" | "check";
  /** Heading rendered above the radio items, also serving as the group's accessible name. */
  slotHeading: ReactNode;
  /** Radio items rendered inside the group. */
  items: MenuRadioItemEntry[];
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    MenuRadioGroup?: NativeElementAttributes<"div", typeof MenuRadioGroup>;
    MenuHeading?: NativeElementAttributes<"div", typeof MenuHeading>;
  };
}
interface MenuSectionEntry {
  /** Discriminator selecting a grouping of menu items under a shared heading. */
  kind?: "section";
  /** Heading rendered above the section, also serving as the group's accessible name. */
  slotHeading?: ReactNode;
  /** Sections do not accept inline children; provide section contents via `items`. */
  children?: never;
  /** Items rendered inside the section. */
  items: (MenuDefaultItemEntry | MenuCheckboxItemEntry | MenuDividerItemEntry)[];
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    MenuSection?: NativeElementAttributes<"div", typeof MenuSection>;
    MenuHeading?: NativeElementAttributes<"div", typeof MenuHeading>;
  };
  /** Sections are not filterable on their own; filtering applies to items inside them. */
  filterValue?: never;
}
interface MenuDividerItemEntry {
  /** Discriminator selecting a non-interactive divider between items. */
  kind: "divider";
  /**
   * Thickness of the divider line.
   * @defaultValue "small"
   */
  width?: "small" | "medium" | "large";
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    Divider?: NativeElementAttributes<"div", typeof Divider>;
  };
  /** Dividers are not filterable. */
  filterValue?: never;
}
interface DropdownContentProps extends MenuRootProps {
  /** Called when focus moves back to the trigger after closing. Call `event.preventDefault` to suppress the focus shift. */
  onCloseAutoFocus?: (event: Event) => void;
  /** Called when focus moves into the content after opening. Call `event.preventDefault` to suppress the focus shift. */
  onOpenAutoFocus?: (event: Event) => void;
  /** Called when the Escape key is pressed. Call `event.preventDefault` to prevent the default close behavior. */
  onEscapeKeyDown?: (event: KeyboardEvent) => void;
  /** Called when a pointer event occurs outside the content bounds. Call `event.preventDefault` to keep the dropdown open. */
  onPointerDownOutside?: (event: Event) => void;
  /** Called when any interaction occurs outside the content bounds. Call `event.preventDefault` to keep the dropdown open. */
  onInteractOutside?: (event: Event) => void;
  /**
   * Alignment of the content relative to the trigger.
   * @defaultValue "start"
   */
  align?: "start" | "end" | "center";
  /**
   * Preferred side of the trigger to render against. Flips automatically when there is not enough space.
   * @defaultValue "bottom"
   */
  side?: "top" | "bottom" | "left" | "right";
  /** CSS anchor name this content positions against. Match the `anchorName` on a custom `DropdownTrigger` when not relying on the auto-generated anchor. */
  positionAnchor?: string;
}
/**
 * The popover surface of a composed dropdown. Anchors to the trigger and hosts the menu items.
 * @param props - {@link DropdownContentProps}
 */
declare const DropdownContent: React$2.ForwardRefExoticComponent<Omit<DropdownContentProps, "ref"> & React$2.RefAttributes<HTMLMenuElement>>;
interface DropdownItemProps extends Omit<MenuItemProps, "onSelect"> {
  /** Called when the user selects the item. Call `event.preventDefault` to keep the dropdown open after selection. */
  onSelect?: (event: Event) => void;
}
/**
 * A selectable item inside a composed dropdown. Activates the associated action and closes the menu by default.
 * @param props - {@link DropdownItemProps}
 */
declare const DropdownItem: React$2.ForwardRefExoticComponent<Omit<DropdownItemProps, "ref"> & React$2.RefAttributes<HTMLButtonElement>>;
interface DropdownRootProps extends PropsWithChildren {
  /** Stable identifier used to derive the popover id and CSS anchor name. Provide one to keep generated ids consistent across server and client renders. */
  id?: string;
  /**
   * Initial open state when uncontrolled.
   * @defaultValue false
   */
  defaultOpen?: boolean;
  /** Controlled open state. Pair with `onOpenChange`. */
  open?: boolean;
  /** Called when the open state changes. */
  onOpenChange?: (open: boolean) => void;
  /**
   * Renders the menu as a modal that blocks interaction with the rest of the page.
   * @defaultValue false
   */
  modal?: boolean;
  /**
   * Size applied to the dropdown and its trigger.
   * @defaultValue "medium"
   */
  size?: "tiny" | "small" | "medium" | "large";
}
interface DropdownSubProps extends PropsWithChildren {
  /**
   * Initial open state of the submenu when uncontrolled.
   * @defaultValue false
   */
  defaultOpen?: boolean;
  /** Controlled open state of the submenu. Pair with `onOpenChange`. */
  open?: boolean;
  /** Called when the submenu's open state changes. */
  onOpenChange?: (open: boolean) => void;
}
interface DropdownSubContentProps extends MenuRootProps {
  /** Distance in pixels from the trigger's side edge. */
  sideOffset?: number;
  /** Distance in pixels from the trigger's aligned edge. */
  alignOffset?: number;
}
/**
 * The popover surface of a submenu inside a composed dropdown. Anchors to the submenu trigger and hosts its items.
 * @param props - {@link DropdownSubContentProps}
 */
declare const DropdownSubContent: React$2.ForwardRefExoticComponent<Omit<DropdownSubContentProps, "ref"> & React$2.RefAttributes<HTMLMenuElement>>;
interface DropdownSubTriggerProps extends Omit<ComponentPropsWithRef<"button">, "ref"> {
  /** Disables interaction with the submenu trigger. */
  disabled?: boolean;
  /** Renders the trigger with destructive styling. */
  danger?: boolean;
  /** Value matched against the dropdown's search filter. Pass `null` to opt out of filtering. */
  filterValue?: string | null;
  /** Content rendered before the trigger label. */
  slotStart?: ReactNode;
  /** Content rendered after the trigger label. Defaults to a right-chevron icon. */
  slotEnd?: ReactNode;
}
/**
 * The interactive row that opens a submenu inside a composed dropdown.
 * @param props - {@link DropdownSubTriggerProps}
 */
declare const DropdownSubTrigger: React$2.ForwardRefExoticComponent<DropdownSubTriggerProps & React$2.RefAttributes<HTMLButtonElement>>;
interface DropdownTriggerProps extends ButtonProps {
  /**
   * Renders a chevron that reflects the current open state. Set to `false` for icon-only kebab/overflow triggers — the icon already signals a menu. Keep `true` for standalone text triggers.
   * @defaultValue true
   */
  showChevron?: boolean;
  /** CSS anchor name used for popover positioning. Auto-generated by default; override to coordinate with a matching `positionAnchor` on a custom `DropdownContent`. Must be a valid CSS custom property name (e.g. `--my-dropdown-anchor`). */
  anchorName?: string;
}
/**
 * The button that opens a composed dropdown. The dropdown content anchors to it by default.
 * @param props - {@link DropdownTriggerProps}
 */
declare const DropdownTrigger: React$1.ForwardRefExoticComponent<Omit<DropdownTriggerProps, "ref"> & React$1.RefAttributes<HTMLButtonElement>>;
interface BaseDropdownItem extends BaseMenuItem {}
interface DropdownDefaultItemEntry extends Omit<MenuDefaultItemEntry, "attributes"> {
  /** URL to navigate to when the item is activated. Renders the item as an `<a>` by default; use `renderLink` to swap in a framework-specific link component. */
  href?: string;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DropdownItem?: NativeElementAttributes<"button", typeof DropdownItem>;
  };
}
interface DropdownCheckboxItemEntry extends Omit<MenuCheckboxItemEntry, "attributes"> {
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DropdownCheckboxItem?: NativeElementAttributes<"button", typeof MenuCheckboxItem>;
  };
}
interface DropdownRadioItemEntry extends Omit<MenuRadioItemEntry, "attributes"> {
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DropdownRadioGroupItem?: NativeElementAttributes<"button", typeof MenuRadioGroupItem>;
  };
}
interface DropdownDividerItemEntry extends Omit<MenuDividerItemEntry, "attributes"> {
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    Divider?: NativeElementAttributes<"div", typeof Divider>;
  };
}
interface DropdownRadioGroupEntry extends Omit<MenuRadioGroupEntry, "attributes" | "items"> {
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DropdownRadioGroup?: NativeElementAttributes<"div", typeof MenuRadioGroup>;
    DropdownHeading?: NativeElementAttributes<"div", typeof MenuHeading>;
  };
  /** Items rendered inside the radio group. */
  items: (string | DropdownRadioItemEntry)[];
}
interface DropdownSubSection extends BaseDropdownItem, Pick<DropdownSubProps, "open" | "defaultOpen" | "onOpenChange">, Pick<DropdownSubContentProps, "density" | "sideOffset" | "alignOffset"> {
  /** Identifies this entry as a submenu. */
  kind: "sub";
  /** Content rendered inside the submenu trigger. */
  children: ReactNode;
  /** Items rendered inside the submenu. */
  items: (DropdownDefaultItemEntry | DropdownCheckboxItemEntry | DropdownRadioGroupEntry | DropdownDividerItemEntry | (Omit<DropdownSectionEntry, "items"> & {
    items: (DropdownDefaultItemEntry | DropdownCheckboxItemEntry | DropdownDividerItemEntry)[];
  }))[];
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DropdownSubTrigger?: NativeElementAttributes<"button", typeof DropdownSubTrigger>;
    DropdownSubContent?: NativeElementAttributes<"menu", typeof DropdownSubContent>;
  };
}
interface DropdownSectionEntry extends Omit<MenuSectionEntry, "items" | "attributes"> {
  /** Items rendered inside the section. */
  items: (DropdownDefaultItemEntry | DropdownCheckboxItemEntry | DropdownSubSection | DropdownDividerItemEntry)[];
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DropdownSection?: NativeElementAttributes<"div", typeof MenuSection>;
    DropdownHeading?: NativeElementAttributes<"div", typeof MenuHeading>;
  };
}
type DropdownEntry = string | DropdownDefaultItemEntry | DropdownCheckboxItemEntry | DropdownRadioGroupEntry | DropdownSubSection | DropdownSectionEntry | DropdownDividerItemEntry;
type DropdownRenderLinkItem = SafeHrefProp<DropdownDefaultItemEntry>;
interface DropdownProps extends PropsWithChildren<Pick<DropdownRootProps, "id" | "open" | "defaultOpen" | "modal" | "onOpenChange" | "size"> & Pick<DropdownContentProps, "density" | "onPointerDownOutside" | "onInteractOutside" | "onEscapeKeyDown" | "align" | "side" | "onCloseAutoFocus"> & Pick<DropdownTriggerProps, "asChild" | "showChevron"> & NativeElementAttributes<"button", typeof DropdownTrigger>> {
  /** Custom renderer for items with an `href`. Use to swap in a framework-specific link component such as Next.js `<Link>`. */
  renderLink?: (item: DropdownRenderLinkItem) => ReactNode;
  /** Initial value of the dropdown search input when filterable. */
  defaultFilterValue?: string;
  /**
   * Renders a search input that filters the items.
   * @defaultValue false
   */
  filterable?: boolean;
  /** Controlled value of the search input. Pair with `onFilterChange`. */
  filterValue?: string;
  /** Overrides the default substring matching used to filter items. */
  filterMatchFn?: MenuSearchProviderProps["matchFn"];
  /** Called when the search value changes. */
  onFilterChange?: (value: string) => void;
  /** Disables interaction with the dropdown. */
  disabled?: boolean;
  /** Entries rendered inside the dropdown. */
  items: DropdownEntry[];
  /** Called when a checkbox item's checked state changes. */
  onItemCheckedChange?: (item: DropdownCheckboxItemEntry, checked: CheckedState) => void;
  /** Called when an item is selected via mouse or keyboard. */
  onItemSelect?: (event: Event, item: DropdownDefaultItemEntry | DropdownCheckboxItemEntry | DropdownRadioItemEntry) => void;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    DropdownContent?: NativeElementAttributes<"menu", typeof DropdownContent>;
    MenuSearch?: NativeElementAttributes<"input", typeof MenuSearch>;
  };
}
interface PopoverContentProps extends PrimitiveProps<"div"> {
  /** Called when focus returns to the trigger after the popover closes. Call `event.preventDefault()` to skip the default focus restoration. */
  onCloseAutoFocus?: (event: Event) => void;
  /** Called when focus first moves into the popover after it opens. Call `event.preventDefault()` to skip the default focus behavior. */
  onOpenAutoFocus?: (event: Event) => void;
  /** Called when Escape is pressed inside the popover. Call `event.preventDefault()` to keep the popover open. */
  onEscapeKeyDown?: (event: KeyboardEvent) => void;
  /** Called when a pointer event occurs outside the popover. Call `event.preventDefault()` to keep the popover open. */
  onPointerDownOutside?: (event: Event) => void;
  /** Called when any interaction occurs outside the popover. Call `event.preventDefault()` to keep the popover open. */
  onInteractOutside?: (event: Event) => void;
  /**
   * Alignment of the popover relative to its anchor.
   * @defaultValue "center"
   */
  align?: "start" | "end" | "center";
  /**
   * Preferred side of the anchor to render against. Flips automatically when the popover would overflow the viewport.
   * @defaultValue "bottom"
   */
  side?: "top" | "bottom" | "left" | "right";
  /** Overrides the CSS anchor name the popover positions against. Use to pair with a custom anchor name set on `PopoverTrigger` or `PopoverAnchor`. */
  positionAnchor?: string;
}
interface PopoverRootProps extends PropsWithChildren {
  /** Stable identifier used to derive the popover's content id and CSS anchor name. Provide a stable value when rendering in SSR-sensitive trees. */
  id?: string;
  /**
   * Initial open state when uncontrolled.
   * @defaultValue false
   */
  defaultOpen?: boolean;
  /** Controlled open state. When provided, the consumer is responsible for updating it in response to `onOpenChange`. */
  open?: boolean;
  /** Called when the open state changes. */
  onOpenChange?: (open: boolean) => void;
  /**
   * Disables interaction with the rest of the page while the popover is open and hides it from assistive tech.
   * @defaultValue false
   */
  modal?: boolean;
}
interface PopoverTriggerProps extends PrimitivePropsWithRef<"button"> {
  /** Disables the popover trigger, preventing it from opening the popover. */
  disabled?: boolean;
  /** Overrides the auto-generated CSS anchor name. Must be a CSS custom property (e.g. `--my-popover`) matching `positionAnchor` on the corresponding `PopoverContent`. */
  anchorName?: string;
}
/**
 * Button that toggles the popover. Uses the native `popovertarget` attribute so the popover still toggles when JavaScript is unavailable.
 * @param props - {@link PopoverTriggerProps}
 */
declare const PopoverTrigger: React$2.ForwardRefExoticComponent<Omit<PopoverTriggerProps, "ref"> & React$2.RefAttributes<HTMLButtonElement>>;
interface PopoverProps extends PopoverContentProps, Pick<PopoverRootProps, "id" | "open" | "defaultOpen" | "onOpenChange" | "modal">, Pick<PopoverTriggerProps, "disabled"> {
  /** Content rendered inside the popover panel. */
  slotContent: React$1.ReactNode;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    PopoverTrigger?: NativeElementAttributes<"button", typeof PopoverTrigger>;
  };
}
/**
 * A floating panel that displays rich content, options, or actions anchored to a trigger.
 * @param props - {@link PopoverProps}
 *
 * @llm For single-string hints, use Tooltip — click-to-reveal for simple text adds an unnecessary interaction step.
 * @llm The interaction test: Tooltip = hover + text-only, Popover = click + rich content, Dropdown = click + item selection, Modal = click + blocking. Use Popover only when the surface is click-triggered and the content is richer than plain text.
 * @llm Do not use Popover as a filter panel or to host complex multi-step forms — reach for Dropdown for action menus, or SidePanel for complex interactive content.
 * @llm Render the trigger as a Button or other clearly interactive element so users recognize the surface is clickable. Plain text triggers offer no affordance.
 * @llm Do not nest Popovers — flatten the information hierarchy instead.
 * @llm Use Popover for onboarding guidance and task walkthroughs (e.g. "Step 1 of 3: Configure your cluster") anchored to the relevant trigger.
 * @llm Use `align="start"` or `align="end"` when the popover content is wide enough to overflow the viewport if centered.
 * @llm Set `modal={true}` only when the content needs focus trapping (e.g. an inline form inside the popover).
 * @llm Top-level props are split by element kind: recognized `<button>` attributes go to `PopoverTrigger`; every other top-level prop (`className`, `onPointerEnter`, `data-*`, …) is spread onto `PopoverContent`. Use `attributes.PopoverTrigger` when something must target the trigger.
 * @llm Popover content must be self-contained — users should be able to understand and act on it without navigating elsewhere. Escalate to SidePanel or Modal if the content needs significant scrolling or multi-step interaction.
 *
 * @example
 * <caption>Basic Popover</caption>
 * Use when you need to show supplementary information triggered by a button click without navigating away.
 * ```tsx
 * <Popover slotContent={<p>Additional context about this item.</p>}>
 * 	<Button>More Info</Button>
 * </Popover>
 * ```
 *
 * @example
 * <caption>Positioning Popover</caption>
 * Combine `side` (`top` | `bottom` | `left` | `right`) and `align` (`start` | `center` | `end`) to control where the popover renders relative to its trigger. Override the defaults when the trigger sits near a viewport edge or sibling content would otherwise overlap.
 * ```tsx
 * <Popover
 * 	side="right"
 * 	align="end"
 * 	slotContent={<p>Right side, end aligned</p>}
 * >
 * 	<Button>Open</Button>
 * </Popover>
 * ```
 *
 * @example
 * <caption>With Chevron Popover</caption>
 * Add `AnimatedChevron` inside the trigger to give a visual hint that the button toggles an open/closed surface.
 * ```tsx
 * <Popover slotContent={<p>Additional context about this item.</p>}>
 * 	<Button kind="secondary">
 * 		Open Popover
 * 		<AnimatedChevron />
 * 	</Button>
 * </Popover>
 * ```
 *
 * @example
 * <caption>With Rich Content Popover</caption>
 * Compose layout primitives inside `slotContent` to render rich, structured content like profile cards. Keep the content focused — avoid overcrowding the popover.
 * ```tsx
 * <Popover
 * 	side="right"
 * 	align="start"
 * 	slotContent={
 * 		<Stack gap="density-md" className="w-[260px] p-4">
 * 			<Stack direction="row" gap="density-md" align="center">
 * 				<Avatar fallback="JD" size="large" />
 * 				<Stack gap="density-sm">
 * 					<Stack direction="row" gap="density-sm" align="center">
 * 						<Text kind="label/bold/md">John Doe</Text>
 * 						<Badge color="green" kind="solid">
 * 							Active
 * 						</Badge>
 * 					</Stack>
 * 					<Text kind="body/regular/sm">Product Designer</Text>
 * 				</Stack>
 * 			</Stack>
 * 			<Stack direction="row" gap="density-sm">
 * 				<Button kind="secondary" size="small">
 * 					View Profile
 * 				</Button>
 * 				<Button kind="primary" color="brand" size="small">
 * 					Message
 * 				</Button>
 * 			</Stack>
 * 		</Stack>
 * 	}
 * >
 * 	<Button>View Team Member</Button>
 * </Popover>
 * ```
 *
 * @example
 * <caption>Modal Popover</caption>
 * Use when the popover content requires focused interaction and should prevent access to the rest of the page.
 * ```tsx
 * <Popover
 * 	modal
 * 	slotContent={
 * 		<div>
 * 			<p>This popover traps focus and dims the background.</p>
 * 			<Button>Action</Button>
 * 		</div>
 * 	}
 * >
 * 	<Button>Open Modal Popover</Button>
 * </Popover>
 * ```
 *
 * @example
 * <caption>With Anchor Popover</caption>
 * Use `PopoverAnchor` when the popover should position against an element other than the trigger — for example, a row in a table where the action button lives in a different cell.
 * ```tsx
 * <PopoverRoot>
 * 	<Stack direction="row" gap="density-xl" align="center">
 * 		<PopoverTrigger asChild>
 * 			<Button>Trigger</Button>
 * 		</PopoverTrigger>
 * 		<PopoverAnchor>
 * 			<Text kind="body/regular/md">Content anchors here</Text>
 * 		</PopoverAnchor>
 * 	</Stack>
 * 	<PopoverContent>
 * 		<p>This popover positions against the anchor, not the trigger.</p>
 * 	</PopoverContent>
 * </PopoverRoot>
 * ```
 *
 * @example
 * <caption>Composed</caption>
 * Use composed primitives when you need full control over the popover trigger and content layout.
 * ```tsx
 * <PopoverRoot>
 * 	<PopoverTrigger asChild>
 * 		<Button>Trigger</Button>
 * 	</PopoverTrigger>
 * 	<PopoverContent>
 * 		<p>Composed popover content using primitives directly.</p>
 * 	</PopoverContent>
 * </PopoverRoot>
 * ```
 */
declare const Popover: React$1.ForwardRefExoticComponent<Omit<PopoverProps, "ref"> & React$1.RefAttributes<HTMLDivElement>>;
declare const flex: (props?: ({
  align?: "center" | "end" | "start" | "baseline" | "stretch" | null | undefined;
  direction?: "col" | "row" | "row-reverse" | "col-reverse" | "column" | "column-reverse" | null | undefined;
  justify?: "center" | "end" | "start" | "stretch" | "normal" | "between" | "around" | "evenly" | null | undefined;
  wrap?: "wrap" | "nowrap" | "wrap-reverse" | null | undefined;
} & ClassProp) | undefined) => string;
type FlexVariantProps = VariantProps<typeof flex>;
interface FlexProps extends React$1.ComponentPropsWithRef<"div">, PrimitiveComponentProps {
  /**
   * Alignment of items along the cross axis. Maps to CSS `align-items`.
   * @defaultValue "stretch"
   * @see https://developer.mozilla.org/en-US/docs/Web/CSS/align-items
   */
  align?: FlexVariantProps["align"];
  /**
   * Direction items flow along the main axis. Maps to CSS `flex-direction`.
   * @defaultValue "row"
   * @see https://developer.mozilla.org/en-US/docs/Web/CSS/flex-direction
   */
  direction?: FlexVariantProps["direction"];
  /**
   * Distribution of space along the main axis. Maps to CSS `justify-content`.
   * @defaultValue "start"
   * @see https://developer.mozilla.org/en-US/docs/Web/CSS/justify-content
   */
  justify?: FlexVariantProps["justify"];
  /**
   * Wrapping behavior of the container. Maps to CSS `flex-wrap`.
   * @defaultValue "nowrap"
   * @see https://developer.mozilla.org/en-US/docs/Web/CSS/flex-wrap
   */
  wrap?: FlexVariantProps["wrap"];
}
interface FormFieldContentGroupProps extends PrimitivePropsWithRef<"div"> {}
/**
 * Wraps the input controls and helper text within a form field.
 * @param props - {@link FormFieldContentGroupProps}
 */
declare const FormFieldContentGroup: React$2.ForwardRefExoticComponent<Omit<FormFieldContentGroupProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface FormFieldControlGroupProps extends PrimitivePropsWithRef<"div"> {}
/**
 * Wraps the input control(s) within a form field, grouping them for layout alongside affixes and helper text.
 * @param props - {@link FormFieldControlGroupProps}
 */
declare const FormFieldControlGroup: React$2.ForwardRefExoticComponent<Omit<FormFieldControlGroupProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
declare const formFieldHelper: (props?: ({
  kind?: "error" | "info" | "success" | null | undefined;
} & ClassProp) | undefined) => string;
type FormFieldHelperVariantProps = VariantProps<typeof formFieldHelper>;
interface FormFieldHelperProps extends PrimitivePropsWithRef<"div"> {
  /** Visual treatment for the helper text. Defaults to the surrounding field's status when omitted. */
  kind?: FormFieldHelperVariantProps["kind"];
}
/**
 * Helper text rendered below a form field, used for instructions, error messages, or success confirmations.
 * @param props - {@link FormFieldHelperProps}
 */
declare const FormFieldHelper: React$2.ForwardRefExoticComponent<Omit<FormFieldHelperProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface FormFieldInfoProps {
  /** Supplementary content shown inside the info popover and mirrored off-screen so the control's `aria-details` resolves to it. */
  children: ReactNode;
  /**
   * Accessible label for the info trigger button.
   * @defaultValue "More information"
   */
  triggerLabel?: string;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    Popover?: NativeElementAttributes<"div", typeof Popover> & Pick<PopoverProps, "side" | "align">;
    PopoverTrigger?: NativeElementAttributes<"button", typeof PopoverTrigger> & Pick<PopoverTriggerProps, "anchorName">;
  };
}
interface FormFieldLabelGroupProps extends PrimitivePropsWithRef<"div"> {}
/**
 * Wraps the label and any adjacent affordances such as the info icon within a form field.
 * @param props - {@link FormFieldLabelGroupProps}
 */
declare const FormFieldLabelGroup: React$2.ForwardRefExoticComponent<Omit<FormFieldLabelGroupProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
declare const formFieldRoot: (props?: ({
  labelPosition?: "left" | "top" | null | undefined;
  required?: boolean | null | undefined;
} & ClassProp) | undefined) => string;
type FormFieldRootVariantProps = VariantProps<typeof formFieldRoot>;
interface FormFieldRootProps extends Omit<PrimitivePropsWithRef<"div">, "id"> {
  /** Identifier used to associate the label, helper text, and control. Falls back to an auto-generated id. */
  id?: string;
  /** Form control name submitted with the field's value. */
  name?: string;
  /** Validation state shared with descendants to drive styling and messaging. */
  status?: FormFieldContextType["status"];
  /** Marks the field as required, rendering the required indicator and propagating to the control. */
  required?: boolean;
  /**
   * Id assigned to the label element. Defaults to `${id}-label`. The label is referenced in the
   * server-rendered `aria-labelledby` only when this is an explicit string; composed `FormFieldLabel`
   * children otherwise wire it up automatically once mounted. Pass `null` to omit the id entirely.
   */
  labelId?: string | null;
  /**
   * Id assigned to the helper element. Defaults to `${id}-helper`. Referenced in the server-rendered
   * `aria-describedby` only when this is an explicit string; a composed `FormFieldHelper` otherwise
   * wires it up automatically once mounted. Pass `null` to omit the id entirely.
   */
  helperId?: string | null;
  /**
   * Id assigned to the supplementary info element. Defaults to `${id}-info`. Referenced in the
   * server-rendered `aria-describedby` / `aria-details` only when this is an explicit string; a
   * composed `FormFieldInfo` otherwise wires it up automatically once mounted. Pass `null` to omit
   * the id entirely.
   */
  infoId?: string | null;
  /**
   * @deprecated Prefer using the individual props
   */
  context?: FormFieldContextType;
  /**
   * Placement of the label relative to the input.
   * - "top" - Standard creation flows and most forms. Supports variable-width inputs and is the fastest position for completion.
   * - "left" - Dense settings panels or key-value layouts with short, consistent labels and constrained vertical space.
   * @defaultValue "top"
   * @llm Use one `labelPosition` for every field in the same form — switching mid-form disrupts the user's scanning pattern.
   */
  labelPosition?: FormFieldRootVariantProps["labelPosition"];
}
interface FormFieldProps extends Omit<PrimitivePropsWithRef<"div">, "children">, Pick<FormFieldRootProps, "labelPosition">, Pick<FormFieldContextType, "required"> {
  /** Unique identifier used to associate the label and helper text with the underlying input. Falls back to an auto-generated ID. */
  id?: string;
  /** Validation state that drives styling and selects which helper message is shown. */
  status?: "success" | "error";
  /** Form control name submitted with the field's value. */
  name?: string;
  /** Label content rendered above or beside the input. */
  slotLabel?: ReactNode;
  /** Content shown inside the popover triggered by the info icon next to the label. */
  slotInfo?: ReactNode;
  /** Message rendered below the input when `status` is `"error"`. */
  slotError?: ReactNode;
  /** Message rendered below the input when `status` is `"success"`. */
  slotSuccess?: ReactNode;
  /** Helper message rendered below the input when no validation status is set. */
  slotHelp?: ReactNode;
  /** Input element(s) wrapped by the field, or a render function that receives the field's accessibility context. */
  children?: ReactNode | ((args: FormFieldContextType) => ReactElement);
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    Label?: NativeElementAttributes<"label", typeof Label>;
    FormFieldLabelGroup?: NativeElementAttributes<"div", typeof FormFieldLabelGroup>;
    FormFieldContentGroup?: NativeElementAttributes<"div", typeof FormFieldContentGroup>;
    FormFieldControlGroup?: NativeElementAttributes<"div", typeof FormFieldControlGroup>;
    FormFieldHelper?: NativeElementAttributes<"div", typeof FormFieldHelper>;
  } & FormFieldInfoProps["attributes"];
}
declare const tabsContent: (props?: ({
  padding?: "default" | "none" | null | undefined;
} & ClassProp) | undefined) => string;
type TabsContentVariantProps = VariantProps<typeof tabsContent>;
interface TabsContentProps extends PrimitivePropsWithRef<"div">, TabsContentVariantProps {
  /** Keeps the panel mounted even when its tab is not active. Useful for preserving state across switches. */
  forceMount?: true;
  /**
   * Padding applied to the panel content.
   * @defaultValue "default"
   */
  padding?: TabsContentVariantProps["padding"];
  /** Removes the default flex layout styles so the panel inherits no enforced layout. */
  unstyled?: true;
  /** Value that pairs this panel with its trigger. */
  value: string;
}
/**
 * A panel of content displayed when its paired tab is active.
 * @param props - {@link TabsContentProps}
 */
declare const TabsContent: React$1.ForwardRefExoticComponent<Omit<TabsContentProps, "ref"> & React$1.RefAttributes<HTMLDivElement>>;
interface TabsListProps extends SlottablePropsWithRef<"div"> {
  /**
   * The visual hierarchy of the tabs.
   * - "primary" - Default style with bottom-border indicators, used for the top-level sections of a view.
   * - "secondary" - Pill-shaped tabs with a filled background, used to divide content within a primary section.
   * - "tertiary" - Minimal, text-only style for optional or supplementary content.
   * @defaultValue "primary"
   */
  kind?: "primary" | "secondary" | "tertiary";
  /**
   * Hides the chevron buttons rendered when the list overflows horizontally.
   * @defaultValue false
   */
  hideOverflowButtons?: boolean;
  /**
   * Restricts the visible triggers to the given child indices, inserting an ellipsis for each gap. For example, `[1,2,3,8,9,10]` shows items 1-3, an ellipsis, then 8-10.
   */
  visibleRange?: number[];
  /**
   * Internal prop. Removes all nv-tab classes from the component.
   * @internal
   */
  unstyled?: true;
}
/**
 * The `role="tablist"` container that groups tab triggers and manages keyboard navigation between them.
 * @param props - {@link TabsListProps}
 */
declare const TabsList: React$1.ForwardRefExoticComponent<Omit<TabsListProps, "ref"> & React$1.RefAttributes<HTMLDivElement>>;
type TabsActivationMode = "automatic" | "manual";
interface TabsRootProps extends PrimitivePropsWithRef<"div"> {
  /**
   * Whether tabs activate as keyboard focus moves to them, or only when Enter or Space is pressed.
   * - "manual" - Arrow keys move focus; Enter or Space activates the focused tab. Use when switching panels is expensive.
   * - "automatic" - Arrow keys both move focus and activate the focused tab. Use when switching is instant.
   * @defaultValue "manual"
   */
  activationMode?: TabsActivationMode;
  /** Controlled active tab value. Pair with `onValueChange`. */
  value?: string;
  /** Initial active tab value when uncontrolled. */
  defaultValue?: string;
  /** Called when the active tab changes. */
  onValueChange?: (value: string) => void;
  /**
   * Set of tab values that have an associated panel mounted in the DOM. Used to wire `aria-controls` on triggers whose panel exists.
   */
  panelValues?: Set<string>;
  /**
   * Internal prop. Removes all nv-tab classes from the component.
   * @internal
   */
  unstyled?: true;
}
/**
 * The outermost element of composed tabs. Establishes the active tab state and shares it with the list, triggers, and panels.
 * @param props - {@link TabsRootProps}
 */
declare const TabsRoot: React$1.ForwardRefExoticComponent<Omit<TabsRootProps, "ref"> & React$1.RefAttributes<HTMLDivElement>>;
interface TabsTriggerProps extends SlottablePropsWithRef<"button"> {
  /**
   * Renders a hidden duplicate of the label sized to the bold weight so the trigger does not shift width when activated.
   * @defaultValue true
   */
  renderSpacingElement?: boolean;
  /** Value that pairs this trigger with its panel. */
  value: string;
  /** Disables the trigger so it cannot be activated or focused. */
  disabled?: boolean;
}
/**
 * A `role="tab"` control that activates its paired panel when selected.
 * @param props - {@link TabsTriggerProps}
 */
declare const TabsTrigger: React$1.ForwardRefExoticComponent<Omit<TabsTriggerProps, "ref"> & React$1.RefAttributes<HTMLButtonElement>>;
interface TabItem {
  /** Renders the trigger into the element passed as `children` instead of a `<button>` wrapper. */
  asChild?: boolean;
  /** Label rendered inside the tab trigger. */
  children: React$1.ReactNode;
  /** Content rendered when the tab is active. Omit when rendering tab content yourself. */
  slotContent?: React$1.ReactNode;
  /** Unique value identifying this tab and its panel. */
  value: string;
  /** Disables the tab so it cannot be activated or focused. */
  disabled?: boolean;
  /** URL the tab navigates to. Providing `href` on any item renders the tabs as a `<nav>` landmark. */
  href?: string;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    TabsTrigger?: NativeElementAttributes<"button", typeof TabsTrigger>;
    TabsContent?: NativeElementAttributes<"div", typeof TabsContent>;
  };
}
type TabRenderLinkItem = SafeHrefProp<TabItem>;
interface TabsPropsBase extends Pick<TabsRootProps, "value" | "defaultValue" | "onValueChange" | "activationMode">, Pick<TabsListProps, "kind" | "visibleRange" | "hideOverflowButtons"> {
  /** Tab definitions rendered in order. */
  items: TabItem[];
  /**
   * Renders a custom link element for each tab with an `href`. When items have `href`, the root element becomes a `<nav>` landmark.
   */
  renderLink?: (item: TabRenderLinkItem) => React$1.ReactNode;
  /** Content rendered before the tab triggers inside the list. */
  slotStart?: React$1.ReactNode;
  /** Content rendered after the tab triggers inside the list. */
  slotEnd?: React$1.ReactNode;
  /**
   * Padding applied to generated tab panels.
   * @defaultValue "default"
   */
  contentPadding?: TabsContentProps["padding"];
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    TabsList?: NativeElementAttributes<"div", typeof TabsList>;
  };
}
interface TabsPropsDivRoot extends TabsPropsBase, NativeElementAttributes<"div", typeof TabsRoot> {
  renderLink?: undefined;
}
interface TabsPropsNavRoot extends TabsPropsBase, NativeElementAttributes<"nav", typeof TabsRoot> {
  renderLink: NonNullable<TabsPropsBase["renderLink"]>;
}
type TabsProps = TabsPropsDivRoot | TabsPropsNavRoot;
interface PageMeta {
  /**
   * The first page in the pagination component.
   * @defaultValue 1
   */
  first: number;
  /**
   * The last page in the pagination component.
   * @defaultValue 	Math.min(page * pageSize, totalItems)
   */
  last: number;
  /**
   * The total number of pages in the pagination component.
   * @defaultValue Math.max(1, Math.ceil(totalItems / pageSize))
   */
  total: number;
}
interface RangeMeta {
  /**
   * The index of the first item on the current page.
   * @defaultValue (current - 1) * pageSize
   */
  firstItemIndex: number;
  /**
   * The index of the last item on the current page.
   * @defaultValue Math.min(firstItemIndex + pageSize - 1, totalItems - 1)
   */
  lastItemIndex: number;
  /**
   * The total number of items in the pagination component.
   * @defaultValue Math.max(1, Math.ceil(totalItems / pageSize))
   */
  totalItems: number;
  /**
   * The size of the page.
   */
  pageSize: number;
}
type InternalSelectItemProps = "aria-checked" | "aria-selected" | "role" | "slotControl";
interface PropsFromMenuItem extends Omit<MenuItemProps, InternalSelectItemProps> {}
interface PropsFromMenuCheckboxItem extends Omit<MenuCheckboxItemProps, InternalSelectItemProps | keyof PropsFromMenuItem> {}
interface SelectItemProps extends PropsFromMenuItem, PropsFromMenuCheckboxItem {
  /** Unique value used to identify the selection and submit it when the Select participates in a form. */
  value: string;
}
/**
 * A selectable option rendered inside the Select listbox. Renders as a checkbox option when the Select is in multi-select mode.
 * @param props - {@link SelectItemProps}
 */
declare const SelectItem: React$1.ForwardRefExoticComponent<Omit<SelectItemProps, "ref"> & React$1.RefAttributes<HTMLElement>>;
interface SelectDefaultItem extends Pick<SelectItemProps, "children" | "value" | "filterValue" | "danger" | "disabled" | "onSelect" | "slotStart" | "slotEnd"> {
  /**
   * The native HTML attributes to apply to the internal composed components.
   */
  attributes?: {
    SelectItem?: NativeElementAttributes<"li", typeof SelectItem>;
  };
}
interface SelectSectionEntry extends Omit<MenuSectionEntry, "items"> {
  /**
   * The items to render in the section.
   */
  items: (string | SelectDefaultItem)[];
}
type SelectEntry = string | SelectDefaultItem | SelectSectionEntry;
interface SingleSelectProps {
  /**
   * The value of the select when initially rendered. Use when you do not need to control the
   * state of the select.
   *
   * Use a string for single value selects and an array of strings for multiple value selects.
   */
  defaultValue?: string;
  /**
   * @remarks When `mutiple` is true the Select allows for multiple values to be selected.
   */
  multiple?: false;
  /**
   * Callback when the selected value of the select changes.
   *
   * @param value - The `value` of the selected item. For multiple value selects, this will
   * be an array of `value`s.
   */
  onValueChange?: (value: string) => void;
  /**
   * Controls the rendering of the selected select value in the trigger.
   *
   * By default the select will render the children of the selected item for a single value select.
   * For a multiple value select, the select renders the count of selected items.
   */
  renderValue?: (value: string, setValue: (value: string | ((prev: string) => string)) => void) => React$1.ReactNode;
  /**
   * The controlled value of the select. Must be used in conjunction with `onValueChange`.
   */
  value?: string;
}
/**
 * The props that overlap between the single and multiple value selects but differ in type. This
 * union allows us to ensure that props are not mixed and matched in types between the two.
 */
type SelectSingleAndMultipleProps = SingleSelectProps | {
  defaultValue?: string[];
  multiple: true;
  onValueChange?: (value: string[]) => void;
  renderValue?: (value: string[], setValue: (value: string[] | ((prev: string[]) => string[])) => void) => React$1.ReactNode;
  value?: string[];
};
interface SelectContentProps extends Omit<React$1.ComponentPropsWithoutRef<"div">, "role"> {
  /**
   * Whether focus returns to the trigger when the listbox closes.
   * @defaultValue true
   */
  autoFocusOnHide?: boolean;
  /**
   * Whether pressing `Escape` closes the listbox.
   * @defaultValue true
   */
  hideOnEscape?: boolean;
}
/**
 * The popover surface that holds the listbox when the Select is open. Anchored to the trigger via the native popover top layer, so it renders correctly inside modals and overflow containers.
 * @param props - {@link SelectContentProps}
 */
declare const SelectContent: React$1.ForwardRefExoticComponent<SelectContentProps & React$1.RefAttributes<HTMLDivElement>>;
interface SelectListboxProps extends Omit<MenuRootProps, "asChild"> {}
/**
 * The list of selectable options inside the Select popover.
 * @param props - {@link SelectListboxProps}
 */
declare const SelectListbox: React$2.ForwardRefExoticComponent<Omit<SelectListboxProps, "ref"> & React$2.RefAttributes<HTMLMenuElement>>;
interface SelectRootProps extends React$1.PropsWithChildren, Pick<SelectSingleAndMultipleProps, "multiple" | "defaultValue" | "onValueChange" | "value"> {
  /**
   * In multi-select mode, allows pressing `Backspace` on the focused trigger to remove the last selected value. Opt in only when the visible selection (e.g. via `renderValue`) makes the target value obvious to the user.
   * @defaultValue false
   */
  allowBackspaceRemoval?: boolean;
  /**
   * Whether the listbox is open on first render. Use for uncontrolled open state.
   * @defaultValue false
   */
  defaultOpen?: boolean;
  /** Prevents the user from opening the listbox or changing the value. */
  disabled?: boolean;
  /** Element id applied to the trigger. */
  id?: string;
  /** Fires when the listbox opens or closes. */
  onOpenChange?: (open: boolean) => void;
  /** Controlled open state of the listbox. */
  open?: boolean;
  /** Locks the current value and prevents the listbox from opening, while keeping the trigger focusable. */
  readOnly?: boolean;
  /**
   * Preferred side relative to the trigger to position the listbox.
   * @defaultValue "bottom"
   */
  side?: "bottom" | "top" | "left" | "right";
  /**
   * Visual size of the trigger. Drives the listbox density unless `density` is set explicitly on the listbox.
   * @defaultValue "medium"
   */
  size?: "small" | "medium" | "large";
}
type HandledAttributes = "children" | "defaultValue" | "disabled" | "id" | "multiple" | "onChange" | "onChangeCapture" | "onInput" | "onInputCapture" | "onInvalid" | "onInvalidCapture" | "onKeyDown" | "required" | "value";
type SelectAttributes = Exclude<AttributesFor<"select">, HandledAttributes | "size">;
interface PropsFromInputShell extends WithInputShellStatus, Omit<InputShellProps, "asChild" | HandledAttributes | "disableFocusRedirect" | SelectAttributes> {}
interface PropsFromSelect extends Pick<ComponentPropsWithoutRef<"select">, SelectAttributes> {}
interface SelectTriggerProps extends PropsFromInputShell, PropsFromSelect {
  /**
   * Renders a clear button on the trigger that resets the selection.
   * @llm Reserve for optional fields. A required field must always carry a value, so the clear button creates a validation conflict.
   */
  dismissible?: boolean;
  /**
   * Keyboard event handler invoked on the trigger before the Select's built-in shortcuts run. Call `event.preventDefault()` to cancel the default ArrowDown/ArrowUp open and (when enabled) Backspace removal behavior.
   */
  onKeyDown?: React$1.KeyboardEventHandler<HTMLButtonElement>;
  /** Override for the display rendered inside the trigger. Receives the current value and a setter; defaults to the matching item's label (or `"N item(s) selected"` for multi-select). */
  renderValue?: (value: string | string[] | undefined, setValue: ((nextValue: string | string[]) => void) | ((nextValueFunc: (prevValue: string | string[]) => string | string[]) => void)) => React$1.ReactNode;
  /** Name applied to the hidden native `<select>` so the value participates in form submission. */
  name?: string;
  /** Text shown when no value is selected. */
  placeholder?: React$1.ReactNode;
  /** Marks the Select as required for form validation. */
  required?: boolean;
  /** Forwarded ref to the hidden native `<select>` used for form integration. Populated only when `name` is set. */
  selectRef?: React$1.Ref<HTMLSelectElement>;
  /** Content rendered before the value, typically an icon. */
  slotStart?: React$1.ReactNode;
  /** Content rendered after the value, before the dismiss/chevron controls. */
  slotEnd?: React$1.ReactNode;
}
/**
 * The button surface that displays the current Select value and opens the listbox. Renders a hidden native `<select>` for form submission when `name` is provided.
 * @param props - {@link SelectTriggerProps}
 */
declare const SelectTrigger: React$1.ForwardRefExoticComponent<SelectTriggerProps & React$1.RefAttributes<HTMLButtonElement>>;
interface BaseSelectProps extends Pick<SelectContentProps, "autoFocusOnHide" | "hideOnEscape">, Pick<SelectListboxProps, "density" | "onScrollToBottom">, Pick<SelectRootProps, "allowBackspaceRemoval" | "defaultOpen" | "defaultValue" | "disabled" | "readOnly" | "onOpenChange" | "open" | "side" | "size" | "value">, Pick<SelectTriggerProps, "name" | "form" | "dismissible" | "placeholder" | "required" | "selectRef" | "slotStart" | "slotEnd" | "status">, Omit<MergedHoistedElementAttributes<[["div", typeof SelectTrigger], ["div", typeof SelectContent]]>, "defaultValue" | "onKeyDown" | "value"> {
  /**
   * Keyboard event handler invoked on the trigger before the Select's built-in shortcuts run. Call `event.preventDefault()` to cancel the default ArrowDown/ArrowUp open and (when enabled) Backspace removal behavior.
   */
  onKeyDown?: React$1.KeyboardEventHandler<HTMLButtonElement>;
  /**
   * Visual treatment of the trigger. Use `"floating"` to drop the background and border for placement on busy surfaces.
   * @defaultValue "flat"
   */
  triggerKind?: "floating" | "flat";
  /** Options rendered in the listbox. Accepts plain strings, item objects, or section entries that group items under a heading. */
  items: SelectEntry[];
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    SelectTrigger?: NativeElementAttributes<"button", typeof SelectTrigger>;
    SelectContent?: NativeElementAttributes<"div", typeof SelectContent>;
    SelectListbox?: NativeElementAttributes<"menu", typeof SelectListbox>;
  };
}
type SelectProps = BaseSelectProps & SelectSingleAndMultipleProps;
/**
 * An input that lets the user pick one or more values from a predefined list.
 * @param props - {@link SelectProps}
 *
 * @llm Use Select for bounded, predefined option lists. Below five options, showing everything inline (RadioGroup or Checkbox group) reduces cognitive load because the user doesn't need to remember what's hidden behind the dropdown.
 * @llm Reach for Combobox when the list is long enough that filtering helps, or when the option count is unknown or could grow at runtime. The Select/Combobox boundary isn't a hard count — favor Combobox for many options, Select for a small bounded set.
 * @llm Select is for picking values, not triggering actions — use Dropdown for action menus.
 * @llm Write descriptive placeholders ("Select a region", "Choose a GPU type") instead of generic "Select..." text.
 * @llm Pre-select a default only when one option is clearly the common choice; an arbitrary default implies a recommendation that doesn't exist. For required fields with a clearly common choice, pre-selecting removes one interaction step for the common case.
 * @llm Group items only when there are 10+ options with natural categories; keep groups to 3–5. Alphabetical ordering optimizes for lookup, not selection — order groups by usage frequency when selection is the primary task.
 * @llm Enable `dismissible` on optional fields and leave it off on required fields (a required field must always have a value).
 * @llm Single-select closes the listbox on selection; multi-select keeps it open and accumulates picks.
 * @llm Top-level props are split by element kind: recognized `<button>` attributes go to `SelectToggle`; every other top-level prop (`className`, `onPointerEnter`, `data-*`, …) is spread onto `SelectTrigger`'s `InputShell` (the field chrome). Use `attributes.SelectContent` when something must target the popover.
 * @llm Default to `size="medium"` to match NVIDIA's uniform control-height rule. Use `size="small"` only when an explicit dense-layout requirement calls for it.
 * @see {@link Combobox}
 * @see {@link RadioGroup}
 * @see {@link Dropdown}
 * @see {@link FormField}
 *
 * @example
 * <caption>Basic Select</caption>
 * Use object items for more control over the displayed label and value. String items are convenient for simple cases.
 * ```tsx
 * <Flex direction="col" gap="density-md">
 * 	<Select
 * 		items={["Apple", "Banana", "Cherry"]}
 * 		defaultValue="Banana"
 * 		placeholder="Choose a fruit"
 * 	/>
 * 	<Select
 * 		items={["Apple", { children: "Banana", value: "b" }, "Cherry"]}
 * 		defaultValue={["Apple", "b"]}
 * 		multiple
 * 		placeholder="Choose multiple fruits"
 * 	/>
 * </Flex>
 * ```
 *
 * @example
 * <caption>Rich Item Content Select</caption>
 * Pass `slotStart` / `slotEnd` on an item for leading or trailing icons, or use `children` directly for richer multi-line content. Set `filterValue` whenever `children` is non-string so type-ahead still works. Use slotStart and slotEnd on the Select itself to add leading/trailing icons.
 * ```tsx
 * <Select
 * 	items={[
 * 		{
 * 			value: "document",
 * 			children: "Document",
 * 			slotStart: <Document />,
 * 		},
 * 		{
 * 			value: "schedule",
 * 			children: "Schedule",
 * 			slotStart: <Calendar />,
 * 		},
 * 		{
 * 			value: "report-q1",
 * 			filterValue: "Q1 Report.pdf",
 * 			children: (
 * 				<Flex direction="col">
 * 					<Flex gap="1" align="center">
 * 						<DocumentLine />
 * 						Q1 Report.pdf
 * 					</Flex>
 * 					<Text kind="label/regular/sm">Updated 2 days ago</Text>
 * 				</Flex>
 * 			),
 * 		},
 * 	]}
 * 	placeholder="Choose an option"
 * 	slotStart={<Filter />}
 * 	slotEnd={<InfoCircle />}
 * />
 * ```
 *
 * @example
 * <caption>Custom Render Value Select</caption>
 * Use renderValue when the trigger needs richer or differently formatted content than the matching item's children.
 * ```tsx
 * <Select
 * 	items={["Apple", "Banana", "Cherry"]}
 * 	defaultValue="Apple"
 * 	renderValue={(value) =>
 * 		value ? `Selected: ${value}` : "No value selected"
 * 	}
 * />
 * ```
 *
 * @example
 * <caption>Grouped Items Select</caption>
 * Use when options logically belong to categories that help users scan the list.
 * ```tsx
 * <Select
 * 	items={[
 * 		{ slotHeading: "Fruits", items: ["Apple", "Banana"] },
 * 		{ slotHeading: "Vegetables", items: ["Carrot", "Lettuce"] },
 * 	]}
 * 	placeholder="Choose an item"
 * />
 * ```
 *
 * @example
 * <caption>Multi Tag Select</caption>
 * Pair multiple + allowBackspaceRemoval with a tag-based renderValue when each selected value should appear as a clearly removable chip. The Backspace shortcut is opt-in because the default `N selected` summary has no per-item target.
 * ```tsx
 * <Select
 * 	allowBackspaceRemoval
 * 	multiple
 * 	defaultValue={["Apple", "Banana"]}
 * 	items={["Apple", "Banana", "Cherry", "Orange", "Pineapple"]}
 * 	renderValue={(value, setValue) =>
 * 		value.length > 0 &&
 * 		value.map((v) => (
 * 			<Tag
 * 				key={v}
 * 				color="gray"
 * 				density="compact"
 * 				onClick={(e) => {
 * 					e.stopPropagation();
 * 					setValue(value.filter((p) => p !== v));
 * 				}}
 * 			>
 * 				{v}
 * 				<Close />
 * 			</Tag>
 * 		))
 * 	}
 * />
 * ```
 *
 * @example
 * <caption>Floating Trigger Select</caption>
 * Use triggerKind='floating' when the Select sits on top of dense content (e.g. a toolbar) and should not render the default input border or background.
 * ```tsx
 * <Select
 * 	items={["Apple", "Banana", "Cherry"]}
 * 	defaultValue="Banana"
 * 	triggerKind="floating"
 * />
 * ```
 *
 * @example
 * <caption>Size Select</caption>
 * Default to `medium` to match NVIDIA's uniform control-height rule. Use `size="small"` only when an explicit dense-layout requirement calls for it. The menu density derives from size by default; override it with the density prop when needed.
 * ```tsx
 * <Flex direction="col" gap="2">
 * 	<Select size="small" items={["Apple", "Banana"]} placeholder="Small" />
 * 	<Select size="medium" items={["Apple", "Banana"]} placeholder="Medium" />
 * 	<Select size="large" items={["Apple", "Banana"]} placeholder="Large" />
 * </Flex>
 * ```
 *
 * @example
 * <caption>Read Only Select</caption>
 * Use readOnly (instead of disabled) when the value should remain visible and copyable but the user cannot open the menu or change the selection — common in review or summary screens.
 * ```tsx
 * <Select items={["Apple", "Banana"]} readOnly defaultValue="Apple" />
 * ```
 *
 * @example
 * <caption>With Form Field Select</caption>
 * Wrap Select in FormField to attach a label, helper text, and validation status that stay in sync with the trigger via context.
 * ```tsx
 * <FormField
 * 	slotLabel="Contact Method"
 * 	slotHelp="How would you like us to reach you?"
 * >
 * 	<Select items={["Phone", "Email", "Both"]} placeholder="Select" />
 * </FormField>
 * ```
 *
 * @example
 * <caption>In Form Select</caption>
 * Pass a name prop so the Select renders a hidden native <select> that participates in standard form submission — works without JavaScript.
 * ```tsx
 * <form>
 * 	<Flex direction="col" gap="density-md">
 * 		<Select
 * 			name="fruit"
 * 			items={["Apple", "Banana", "Cherry"]}
 * 			placeholder="Pick a fruit"
 * 		/>
 * 		<Button type="submit">Submit</Button>
 * 	</Flex>
 * </form>
 * ```
 *
 * @example
 * <caption>Controlled Select</caption>
 * Use controlled mode when the selection state lives outside the Select — for example, when syncing with form libraries, URL state, or another component.
 * ```tsx
 * () => {
 * 	const [value, setValue] = useState("Apple");
 * 	return (
 * 		<Select
 * 			items={["Apple", "Banana", "Cherry"]}
 * 			value={value}
 * 			onValueChange={setValue}
 * 		/>
 * 	);
 * }
 * ```
 *
 * @example
 * <caption>Composed</caption>
 * Use the composed primitives when you need full control over trigger rendering, content layout, or custom items.
 * ```tsx
 * <SelectRoot defaultValue="b">
 * 	<SelectTrigger placeholder="Pick one" />
 * 	<SelectContent>
 * 		<SelectItem value="a">Option A</SelectItem>
 * 		<SelectItem value="b">Option B</SelectItem>
 * 		<SelectItem value="c">Option C</SelectItem>
 * 	</SelectContent>
 * </SelectRoot>
 * ```
 */
declare const Select: React$1.ForwardRefExoticComponent<SelectProps & React$1.RefAttributes<HTMLButtonElement>>;
type PaginationAction = "form";
interface PaginationRootProps extends PrimitivePropsWithRef<"div"> {
  /** Total number of items in the underlying dataset. Used to derive the page count and item range. */
  totalItems: number;
  /**
   * Initial page when uncontrolled.
   * @defaultValue 1
   */
  defaultPage?: number;
  /** Controlled current page. When provided, the consumer is responsible for updating it in response to `onPageChange`. */
  page?: number;
  /** Called when the current page changes. */
  onPageChange?: (page: number) => void;
  /**
   * Initial page size when uncontrolled.
   * @defaultValue 10
   */
  defaultPageSize?: number;
  /** Controlled page size. When provided, the consumer is responsible for updating it in response to `onPageSizeChange`. */
  pageSize?: number;
  /** Called when the page size changes. */
  onPageSizeChange?: (pageSize: number) => void;
  /** Overrides the computed page metadata. Use when the source of truth lives outside the component (e.g. cursor-based APIs). */
  pageMeta?: PageMeta;
  /** Overrides the computed item-range metadata. Use when the visible range cannot be derived from `page` and `pageSize`. */
  rangeMeta?: RangeMeta;
  /**
   * Selectable page sizes shown in the page-size select.
   * @defaultValue [10, 25, 50, 100]
   */
  pageSizeOptions?: number[];
  /**
   * Submission mode for server-driven pagination. When set to `"form"`, controls render as submit buttons that submit `page` and `pageSize` to the surrounding form instead of updating internal state.
   */
  action?: PaginationAction;
}
interface StatusMessageFooterProps extends PrimitivePropsWithRef<"div"> {}
/**
 * The action area of a status message. Aligns and spaces call-to-action buttons that help the user recover from or act on the state.
 * @param props - {@link StatusMessageFooterProps}
 */
declare const StatusMessageFooter: React$2.ForwardRefExoticComponent<Omit<StatusMessageFooterProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface StatusMessageHeaderProps extends PrimitivePropsWithRef<"div"> {}
/**
 * Wraps the heading and subheading in a status message and controls the spacing between them.
 * @param props - {@link StatusMessageHeaderProps}
 */
declare const StatusMessageHeader: React$2.ForwardRefExoticComponent<Omit<StatusMessageHeaderProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface StatusMessageHeadingProps extends PrimitivePropsWithRef<"div"> {}
/**
 * The primary line of text in a status message that names the state being communicated.
 * @param props - {@link StatusMessageHeadingProps}
 */
declare const StatusMessageHeading: React$2.ForwardRefExoticComponent<Omit<StatusMessageHeadingProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface StatusMessageMediaProps extends PrimitivePropsWithRef<"div"> {}
/**
 * The icon or illustration rendered above a status message's heading. Handles SVG sizing and color automatically.
 * @param props - {@link StatusMessageMediaProps}
 */
declare const StatusMessageMedia: React$2.ForwardRefExoticComponent<Omit<StatusMessageMediaProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
declare const statusMessageRoot: (props?: ({
  size?: "small" | "medium" | null | undefined;
} & ClassProp) | undefined) => string;
type StatusMessageRootVariantProps = VariantProps<typeof statusMessageRoot>;
interface StatusMessageRootProps extends PrimitivePropsWithRef<"div"> {
  /**
   * Visual scale of the message.
   * - "small" - Use within constrained areas such as cards or side panels.
   * - "medium" - Use for full-page empty states like 404s or no-results pages.
   * @defaultValue "medium"
   */
  size?: StatusMessageRootVariantProps["size"];
}
interface StatusMessageSubheadingProps extends PrimitivePropsWithRef<"div"> {}
/**
 * The supporting description rendered below the heading. Use it to explain the state and hint at next steps.
 * @param props - {@link StatusMessageSubheadingProps}
 */
declare const StatusMessageSubheading: React$2.ForwardRefExoticComponent<Omit<StatusMessageSubheadingProps, "ref"> & React$2.RefAttributes<HTMLDivElement>>;
interface StatusMessageProps extends StatusMessageRootProps {
  /** Action area rendered below the message, typically containing call-to-action buttons. */
  slotFooter?: ReactNode;
  /** Supporting description rendered below the heading. Default `body/regular/md` typography styles */
  slotSubheading?: ReactNode;
  /** Primary message that names the status. Default `body/bold/2xl` or `xl` typography styles (based on size) */
  slotHeading: ReactNode;
  /** Icon or image rendered above the heading. SVG sizing and color are handled automatically. */
  slotMedia?: ReactNode;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    StatusMessageMedia?: NativeElementAttributes<"div", typeof StatusMessageMedia>;
    StatusMessageHeader?: NativeElementAttributes<"div", typeof StatusMessageHeader>;
    StatusMessageHeading?: NativeElementAttributes<"div", typeof StatusMessageHeading>;
    StatusMessageSubheading?: NativeElementAttributes<"div", typeof StatusMessageSubheading>;
    StatusMessageFooter?: NativeElementAttributes<"div", typeof StatusMessageFooter>;
  };
}
declare const tableRoot: (props?: ({
  density?: "compact" | "standard" | "spacious" | null | undefined;
  layout?: "fixed" | "auto" | null | undefined;
  align?: "center" | "left" | "right" | null | undefined;
  hoverableRows?: boolean | null | undefined;
} & ClassProp) | undefined) => string;
type TableRootVariantProps = VariantProps<typeof tableRoot>;
interface TableRootProps extends Omit<ComponentPropsWithRef<"table">, "align">, DensityVariantProps {
  /**
   * Controls the CSS `table-layout` algorithm.
   * @defaultValue "fixed"
   * @see {@link https://developer.mozilla.org/en-US/docs/Web/CSS/table-layout}
   */
  layout?: TableRootVariantProps["layout"];
  /**
   * Horizontal alignment applied to every cell in the table.
   * @defaultValue "left"
   * @llm Use "left" for textual data and "right" for numerical data. Avoid mixing alignments across columns when it would hurt readability.
   */
  align?: TableRootVariantProps["align"];
  /**
   * Applies a hover style to body rows on pointer-over.
   * @defaultValue false
   * @llm Enable only when rows are interactive (e.g. `onRowSelect` is set). On purely informational tables, hover is false affordance.
   */
  hoverableRows?: TableRootVariantProps["hoverableRows"];
}
interface TableToolbarProps extends React$1.ComponentPropsWithoutRef<"section"> {
  /** Renders the toolbar into the child element passed via `children` instead of a `<section>` wrapper. */
  asChild?: boolean;
  /**
   * Swaps the default toolbar content with the bulk actions toolbar. Set this to true when rows are selected.
   * @defaultValue false
   */
  showBulkActionsToolbar?: boolean;
  /** Content rendered inside the bulk actions toolbar when `showBulkActionsToolbar` is true. */
  slotBulkActions?: React$1.ReactNode;
}
declare const tagRoot: (props?: ({
  color?: "blue" | "gray" | "green" | "purple" | "red" | "teal" | "yellow" | null | undefined;
  density?: "compact" | "standard" | "spacious" | null | undefined;
  kind?: "solid" | "outline" | null | undefined;
  selected?: boolean | null | undefined;
} & ClassProp) | undefined) => string;
type TagRootVariants = VariantProps<typeof tagRoot>;
interface TagProps extends Omit<PrimitivePropsWithRef<"button">, "color">, Omit<TagRootVariants, "density">, DensityVariantProps {
  /**
   * The visual style of the tag.
   * @defaultValue "solid"
   * @llm Prefer `kind="outline"` for most Tag usage — the outline treatment visually distinguishes Tags (interactive) from solid Badges (read-only status). Use `kind="solid"` when distinguishing entity type (Model vs Blueprint vs Service) across a row of Tags.
   */
  kind?: TagRootVariants["kind"];
  /**
   * Accent color of the tag.
   * @defaultValue "blue"
   * @llm The framework default is `"blue"`, but pass `color="gray"` explicitly whenever the tag has no semantic meaning — a uniform neutral palette across the app keeps the status-vs-category signal intact. Semantic colors (green/red/yellow) are reserved for Badge; do not assign a distinct color per categorical value (rainbow coding).
   */
  color?: TagRootVariants["color"];
  /**
   * Renders the tag as a non-interactive `<span>` instead of a `<button>`. Use when placing a tag inside another interactive element like a clickable card.
   */
  readOnly?: boolean;
  /** Marks the tag as selected, applying the selected style and removing the hover state. */
  selected?: TagRootVariants["selected"];
}
declare const textAreaElement: (props?: ({
  resizeable?: "auto" | "manual" | null | undefined;
} & ClassProp) | undefined) => string;
interface TextAreaElementProps extends ComponentPropsWithRef<"textarea"> {
  /**
   * Controls how the textarea can be resized.
   * - "manual" - The user can drag a handle to resize vertically.
   * - "auto" - Grows to fit content up to `--max-auto-height` (defaults to 400px).
   * @llm Prefer "auto" for most forms; reach for "manual" only when the layout requires a fixed-height field with a drag handle.
   * @llm Leave unset to keep the textarea at its fixed initial height.
   */
  resizeable?: VariantProps<typeof textAreaElement>["resizeable"];
}
type ExcludedTextAreaAttributes = Exclude<AttributesFor<"textarea">, "size">;
interface PropsFromRoot extends WithInputShellStatus, Omit<InputShellProps, ExcludedTextAreaAttributes> {}
interface PropsFromTextArea extends Pick<TextAreaElementProps, ExcludedTextAreaAttributes | "resizeable"> {}
interface TextAreaProps extends PropsFromRoot, PropsFromTextArea {
  /** Called with the new string value when the textarea content changes. Pair with `value` for controlled state. */
  onValueChange?: (value: string, event: ChangeEvent<HTMLTextAreaElement>) => void;
  /** Content rendered before the textarea. Moves above the textarea when `layout="vertical"`. */
  slotStart?: ReactNode;
  /** Content rendered after the textarea. Moves below the textarea when `layout="vertical"`. */
  slotEnd?: ReactNode;
  /** Controlled textarea value. Pair with `onChange` or `onValueChange` to keep it editable. */
  value?: string;
  /** Native HTML attributes forwarded to the internal composed components. */
  attributes?: {
    TextAreaElement?: TextAreaElementProps;
  };
}
/**
 * A multi-line text input for free-form responses such as descriptions, comments, or notes.
 * @param props - {@link TextAreaProps}
 *
 * @llm Reach for TextInput when the response will almost always fit on a single line. Use TextArea when multi-line content is expected (descriptions, comments, messages).
 * @llm Prefer `resizeable="auto"` so the field grows with content instead of forcing the user to scroll inside a fixed box.
 * @llm Omitting `resizeable` produces a fixed-height textarea that scrolls internally — this is rarely the right choice; `auto` is almost always preferred for usability.
 * @llm Placeholder text is not a label substitute. Wrap in FormField with `slotLabel` or set `aria-label` for standalone fields.
 * @see {@link TextInput}
 * @see {@link FormField}
 *
 * @example
 * <caption>Basic Text Area</caption>
 * ```tsx
 * <TextArea
 * 	aria-label="Description"
 * 	defaultValue="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
 * />
 * ```
 *
 * @example
 * <caption>Controlled Text Area</caption>
 * Controlled mode lets you manage value externally, useful for syncing with form libraries, character counters, or external validation.
 * ```tsx
 * () => {
 * 	const [value, setValue] = useState("");
 * 	return (
 * 		<TextArea
 * 			value={value}
 * 			onValueChange={setValue}
 * 			placeholder="Type something"
 * 		/>
 * 	);
 * }
 * ```
 *
 * @example
 * <caption>With Placeholder Text Area</caption>
 * Use placeholder to hint at the expected format. Do not use it as a substitute for a label or required helper text.
 * ```tsx
 * <TextArea placeholder="Enter a description..." />
 * ```
 *
 * @example
 * <caption>Auto Resize Text Area</caption>
 * Use when the textarea should grow with content up to a max height, avoiding scrollbars for short entries. Override `--max-auto-height` (default 400px) to change the cap.
 * ```tsx
 * <TextArea resizeable="auto" placeholder="Start typing..." />
 * ```
 *
 * @example
 * <caption>Manual Resize Text Area</caption>
 * Use when the user should control the textarea height manually via a drag handle in the bottom-right corner.
 * ```tsx
 * <TextArea resizeable="manual" placeholder="Drag to resize" />
 * ```
 *
 * @example
 * <caption>Size Text Area</caption>
 * Default to `medium`. Reserve `small` for dense-table or toolbar contexts where vertical space is constrained — avoid it in regular forms. Use `large` only when surrounding controls already step up to a larger scale.
 * ```tsx
 * <Flex direction="col" gap="density-md">
 * 	<TextArea size="small" placeholder="Small" />
 * 	<TextArea size="medium" placeholder="Medium" />
 * 	<TextArea size="large" placeholder="Large" />
 * </Flex>
 * ```
 *
 * @example
 * <caption>Status Text Area</caption>
 * Set `status` to drive validation styling manually — `error` for failed validation (pair with helper text on the wrapping FormField) or `success` to confirm a passing value. Reach for `withValidation` when native HTML constraints can drive the state automatically.
 * ```tsx
 * <Flex direction="col" gap="density-md">
 * 	<TextArea
 * 		aria-label="Invalid description"
 * 		status="error"
 * 		defaultValue="Invalid content"
 * 	/>
 * 	<TextArea
 * 		aria-label="Valid description"
 * 		status="success"
 * 		defaultValue="Looks good"
 * 	/>
 * </Flex>
 * ```
 *
 * @example
 * <caption>With Validation Text Area</caption>
 * Set `withValidation` to drive status styling from the browser's native `:user-valid` / `:user-invalid` pseudo-classes — no JS state required. Combine with `required`, `minLength`, `maxLength`, etc.
 * ```tsx
 * <TextArea
 * 	withValidation
 * 	required
 * 	placeholder="Will turn red on blur if empty"
 * />
 * ```
 *
 * @example
 * <caption>Disabled Text Area</caption>
 * Use when the field is unavailable due to external state (e.g. permissions). Prefer `readOnly` when the value should still be selectable.
 * ```tsx
 * <TextArea
 * 	aria-label="Disabled description"
 * 	disabled
 * 	defaultValue="Read-only content"
 * />
 * ```
 *
 * @example
 * <caption>Read Only Text Area</caption>
 * Use when the value should remain visible and copyable but cannot be edited — common in review or summary screens.
 * ```tsx
 * <TextArea
 * 	aria-label="Read-only description"
 * 	readOnly
 * 	defaultValue="Read-only value"
 * />
 * ```
 *
 * @example
 * <caption>With Slots Text Area</caption>
 * Use `slotStart` / `slotEnd` to attach affordances like icons or action buttons. Use `mb-auto` / `mt-auto` utilities to anchor slot content to the top or bottom of a multi-line field.
 * ```tsx
 * <TextArea
 * 	placeholder="Add a comment..."
 * 	slotStart={<Document className="mb-auto" />}
 * 	slotEnd={
 * 		<Button className="mt-auto" color="brand" size="small" aria-label="Send">
 * 			<ChevronRight />
 * 		</Button>
 * 	}
 * />
 * ```
 *
 * @example
 * <caption>Vertical Layout Text Area</caption>
 * Use `layout="vertical"` to stack slot content above and below the textarea — ideal for chat composers, comment boxes, or toolbars that need full-width controls.
 * ```tsx
 * <TextArea
 * 	layout="vertical"
 * 	placeholder="Compose your message..."
 * 	slotStart={
 * 		<Badge color="green" kind="solid">
 * 			Active
 * 		</Badge>
 * 	}
 * 	slotEnd={
 * 		<Flex gap="2" className="w-full">
 * 			<Button kind="tertiary" size="small" aria-label="Attach">
 * 				<Document />
 * 			</Button>
 * 			<Button
 * 				className="ml-auto"
 * 				color="brand"
 * 				size="small"
 * 				aria-label="Send"
 * 			>
 * 				<ChevronRight />
 * 			</Button>
 * 		</Flex>
 * 	}
 * />
 * ```
 *
 * @example
 * <caption>With Form Field Text Area</caption>
 * Wrap TextArea in FormField to attach a label, helper text, and validation status that stay in sync with the field via context.
 * ```tsx
 * <FormField
 * 	slotLabel="Description"
 * 	slotHelp="Your description will be used to generate a report."
 * >
 * 	<TextArea placeholder="Enter your description" required />
 * </FormField>
 * ```
 *
 * @example
 * <caption>Composed</caption>
 * Use the composed primitives when you need full control over the shell and textarea element layout — for example, to mix the textarea with sibling buttons inside the same shell.
 * ```tsx
 * <InputShell>
 * 	<TextAreaElement defaultValue="Composed primitives" resizeable="auto" />
 * </InputShell>
 * ```
 */
declare const TextArea: React$2.ForwardRefExoticComponent<TextAreaProps & React$2.RefAttributes<HTMLTextAreaElement>>;
declare module "react" {
  interface CSSProperties {
    [key: `--${string}`]: string | number;
  }
}
//#endregion
//#region src/components/LoadingButton/index.d.ts
interface Props$6 extends ComponentProps<typeof Button> {
  loading?: boolean;
  height?: number;
}
/**
 * Button for displaying a spinner to convey an in-progress button state
 */
export declare const LoadingButton: FC<PropsWithChildren<Props$6>>;
//#endregion
//#region src/components/FormModal/index.d.ts
interface FormModalProps {
  open: boolean;
  title: ReactNode;
  instruction?: string;
  submitButtonText: string;
  errorText?: string | null;
  cancelButtonText?: string;
  /**
   * Disables everything, the submit button, the cancel button, and the ability to close the modal at all.
   * Use this to prevent the user from editing/closing/submitting the modal while request is in flight.
   */
  disabled?: boolean;
  loading?: boolean;
  /**
   * Only disables the submit button.
   * Use this to prevent the user from submitting until the form is valid and dirty.
   * */
  submitDisabled?: boolean;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  onClose: () => void;
  styles?: React.CSSProperties;
  className?: string;
  slotFooterLeft?: ReactNode;
  slotFooterRight?: ReactNode;
  attributes?: {
    CancelButton?: ComponentProps<typeof Button>;
    SubmitButton?: ComponentProps<typeof LoadingButton>;
    Form?: ComponentPropsWithoutRef<'form'>;
  };
}
export declare const FormModal: FC<PropsWithChildren<FormModalProps>>;
//#endregion
//#region src/providers/toast/types.d.ts
interface MessageFnOptions {
  durationMs?: number | false;
}
type NotifyType = 'success' | 'error' | 'info' | 'warning';
/**
 * Provider-independent notification sink. Components exposed across the plugin
 * boundary take this as a prop so a plugin can route them at Studio's toaster
 * without sharing a ToastContext module instance.
 */
type NotifyFn = (message: string, type?: NotifyType, options?: MessageFnOptions) => void;
//#endregion
//#region src/components/ConfirmationModal/index.d.ts
type SubmitButtonColor = NonNullable<ComponentProps<typeof LoadingButton>['color']>;
interface ConfirmationModalProps extends Pick<FormModalProps, 'open' | 'onClose' | 'cancelButtonText'> {
  /** Return true when the action succeeded (shows success toast); false shows error toast. */
  onConfirm: () => boolean | Promise<boolean>;
  title: string;
  /** Body copy shown above the optional confirmation field. */
  description: string;
  /** When set with `simpleConfirm === false`, user must type this exact string to enable submit. */
  confirmationText?: string;
  /** If true (default), only a single button click is required. Set `false` to require typing `confirmationText`. */
  simpleConfirm?: boolean;
  successText?: string;
  errorText?: string;
  submitButtonText?: string;
  /** Passed to the submit button; omit for default (non-destructive) styling. */
  submitButtonColor?: SubmitButtonColor;
  /** When true, success and error toasts from this modal are skipped (caller handles feedback). */
  suppressResultToasts?: boolean;
  /** Where result messages go. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}
export declare const ConfirmationModal: FC<ConfirmationModalProps>;
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/helpers/typeAliases.d.cts
type Primitive$1 = string | number | symbol | bigint | boolean | null | undefined;
type Scalars = Primitive$1 | Primitive$1[];
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/helpers/util.d.cts
declare namespace util {
  type AssertEqual<T, U> = (<V>() => V extends T ? 1 : 2) extends (<V>() => V extends U ? 1 : 2) ? true : false;
  export type isAny<T> = 0 extends 1 & T ? true : false;
  export const assertEqual: <A, B>(_: AssertEqual<A, B>) => void;
  export function assertIs<T>(_arg: T): void;
  export function assertNever(_x: never): never;
  export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;
  export type OmitKeys<T, K extends string> = Pick<T, Exclude<keyof T, K>>;
  export type MakePartial<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
  export type Exactly<T, X> = T & Record<Exclude<keyof X, keyof T>, never>;
  export type InexactPartial<T> = { [k in keyof T]?: T[k] | undefined; };
  export const arrayToEnum: <T extends string, U extends [T, ...T[]]>(items: U) => { [k in U[number]]: k; };
  export const getValidEnumValues: (obj: any) => any[];
  export const objectValues: (obj: any) => any[];
  export const objectKeys: ObjectConstructor["keys"];
  export const find: <T>(arr: T[], checker: (arg: T) => any) => T | undefined;
  export type identity<T> = objectUtil.identity<T>;
  export type flatten<T> = objectUtil.flatten<T>;
  export type noUndefined<T> = T extends undefined ? never : T;
  export const isInteger: NumberConstructor["isInteger"];
  export function joinValues<T extends any[]>(array: T, separator?: string): string;
  export const jsonStringifyReplacer: (_: string, value: any) => any;
  export {};
}
declare namespace objectUtil {
  export type MergeShapes<U, V> = keyof U & keyof V extends never ? U & V : { [k in Exclude<keyof U, keyof V>]: U[k]; } & V;
  type optionalKeys<T extends object> = { [k in keyof T]: undefined extends T[k] ? k : never; }[keyof T];
  type requiredKeys<T extends object> = { [k in keyof T]: undefined extends T[k] ? never : k; }[keyof T];
  export type addQuestionMarks<T extends object, _O = any> = { [K in requiredKeys<T>]: T[K]; } & { [K in optionalKeys<T>]?: T[K]; } & { [k in keyof T]?: unknown; };
  export type identity<T> = T;
  export type flatten<T> = identity<{ [k in keyof T]: T[k]; }>;
  export type noNeverKeys<T> = { [k in keyof T]: [T[k]] extends [never] ? never : k; }[keyof T];
  export type noNever<T> = identity<{ [k in noNeverKeys<T>]: k extends keyof T ? T[k] : never; }>;
  export const mergeShapes: <U, T>(first: U, second: T) => T & U;
  export type extendShape<A extends object, B extends object> = keyof A & keyof B extends never ? A & B : { [K in keyof A as K extends keyof B ? never : K]: A[K]; } & { [K in keyof B]: B[K]; };
  export {};
}
declare const ZodParsedType: {
  string: "string";
  nan: "nan";
  number: "number";
  integer: "integer";
  float: "float";
  boolean: "boolean";
  date: "date";
  bigint: "bigint";
  symbol: "symbol";
  function: "function";
  undefined: "undefined";
  null: "null";
  array: "array";
  object: "object";
  unknown: "unknown";
  promise: "promise";
  void: "void";
  never: "never";
  map: "map";
  set: "set";
};
type ZodParsedType = keyof typeof ZodParsedType;
declare const getParsedType: (data: any) => ZodParsedType;
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/ZodError.d.cts
type allKeys<T> = T extends any ? keyof T : never;
type inferFlattenedErrors<T extends ZodType<any, any, any>, U = string> = typeToFlattenedError<TypeOf<T>, U>;
type typeToFlattenedError<T, U = string> = {
  formErrors: U[];
  fieldErrors: { [P in allKeys<T>]?: U[]; };
};
declare const ZodIssueCode: {
  invalid_type: "invalid_type";
  invalid_literal: "invalid_literal";
  custom: "custom";
  invalid_union: "invalid_union";
  invalid_union_discriminator: "invalid_union_discriminator";
  invalid_enum_value: "invalid_enum_value";
  unrecognized_keys: "unrecognized_keys";
  invalid_arguments: "invalid_arguments";
  invalid_return_type: "invalid_return_type";
  invalid_date: "invalid_date";
  invalid_string: "invalid_string";
  too_small: "too_small";
  too_big: "too_big";
  invalid_intersection_types: "invalid_intersection_types";
  not_multiple_of: "not_multiple_of";
  not_finite: "not_finite";
};
type ZodIssueCode = keyof typeof ZodIssueCode;
type ZodIssueBase = {
  path: (string | number)[];
  message?: string | undefined;
};
interface ZodInvalidTypeIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_type;
  expected: ZodParsedType;
  received: ZodParsedType;
}
interface ZodInvalidLiteralIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_literal;
  expected: unknown;
  received: unknown;
}
interface ZodUnrecognizedKeysIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.unrecognized_keys;
  keys: string[];
}
interface ZodInvalidUnionIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_union;
  unionErrors: ZodError[];
}
interface ZodInvalidUnionDiscriminatorIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_union_discriminator;
  options: Primitive$1[];
}
interface ZodInvalidEnumValueIssue extends ZodIssueBase {
  received: string | number;
  code: typeof ZodIssueCode.invalid_enum_value;
  options: (string | number)[];
}
interface ZodInvalidArgumentsIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_arguments;
  argumentsError: ZodError;
}
interface ZodInvalidReturnTypeIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_return_type;
  returnTypeError: ZodError;
}
interface ZodInvalidDateIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_date;
}
type StringValidation = "email" | "url" | "emoji" | "uuid" | "nanoid" | "regex" | "cuid" | "cuid2" | "ulid" | "datetime" | "date" | "time" | "duration" | "ip" | "cidr" | "base64" | "jwt" | "base64url" | {
  includes: string;
  position?: number | undefined;
} | {
  startsWith: string;
} | {
  endsWith: string;
};
interface ZodInvalidStringIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_string;
  validation: StringValidation;
}
interface ZodTooSmallIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.too_small;
  minimum: number | bigint;
  inclusive: boolean;
  exact?: boolean;
  type: "array" | "string" | "number" | "set" | "date" | "bigint";
}
interface ZodTooBigIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.too_big;
  maximum: number | bigint;
  inclusive: boolean;
  exact?: boolean;
  type: "array" | "string" | "number" | "set" | "date" | "bigint";
}
interface ZodInvalidIntersectionTypesIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.invalid_intersection_types;
}
interface ZodNotMultipleOfIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.not_multiple_of;
  multipleOf: number | bigint;
}
interface ZodNotFiniteIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.not_finite;
}
interface ZodCustomIssue extends ZodIssueBase {
  code: typeof ZodIssueCode.custom;
  params?: {
    [k: string]: any;
  };
}
type DenormalizedError = {
  [k: string]: DenormalizedError | string[];
};
type ZodIssueOptionalMessage = ZodInvalidTypeIssue | ZodInvalidLiteralIssue | ZodUnrecognizedKeysIssue | ZodInvalidUnionIssue | ZodInvalidUnionDiscriminatorIssue | ZodInvalidEnumValueIssue | ZodInvalidArgumentsIssue | ZodInvalidReturnTypeIssue | ZodInvalidDateIssue | ZodInvalidStringIssue | ZodTooSmallIssue | ZodTooBigIssue | ZodInvalidIntersectionTypesIssue | ZodNotMultipleOfIssue | ZodNotFiniteIssue | ZodCustomIssue;
type ZodIssue = ZodIssueOptionalMessage & {
  fatal?: boolean | undefined;
  message: string;
};
declare const quotelessJson: (obj: any) => string;
type recursiveZodFormattedError<T> = T extends [any, ...any[]] ? { [K in keyof T]?: ZodFormattedError<T[K]>; } : T extends any[] ? {
  [k: number]: ZodFormattedError<T[number]>;
} : T extends object ? { [K in keyof T]?: ZodFormattedError<T[K]>; } : unknown;
type ZodFormattedError<T, U = string> = {
  _errors: U[];
} & recursiveZodFormattedError<NonNullable<T>>;
type inferFormattedError<T extends ZodType<any, any, any>, U = string> = ZodFormattedError<TypeOf<T>, U>;
declare class ZodError<T = any> extends Error {
  issues: ZodIssue[];
  get errors(): ZodIssue[];
  constructor(issues: ZodIssue[]);
  format(): ZodFormattedError<T>;
  format<U>(mapper: (issue: ZodIssue) => U): ZodFormattedError<T, U>;
  static create: (issues: ZodIssue[]) => ZodError<any>;
  static assert(value: unknown): asserts value is ZodError;
  toString(): string;
  get message(): string;
  get isEmpty(): boolean;
  addIssue: (sub: ZodIssue) => void;
  addIssues: (subs?: ZodIssue[]) => void;
  flatten(): typeToFlattenedError<T>;
  flatten<U>(mapper?: (issue: ZodIssue) => U): typeToFlattenedError<T, U>;
  get formErrors(): typeToFlattenedError<T, string>;
}
type stripPath<T extends object> = T extends any ? util.OmitKeys<T, "path"> : never;
type IssueData = stripPath<ZodIssueOptionalMessage> & {
  path?: (string | number)[];
  fatal?: boolean | undefined;
};
type ErrorMapCtx = {
  defaultError: string;
  data: any;
};
type ZodErrorMap = (issue: ZodIssueOptionalMessage, _ctx: ErrorMapCtx) => {
  message: string;
};
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/locales/en.d.cts
declare const errorMap: ZodErrorMap;
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/errors.d.cts
declare function setErrorMap(map: ZodErrorMap): void;
declare function getErrorMap(): ZodErrorMap;
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/helpers/parseUtil.d.cts
declare const makeIssue: (params: {
  data: any;
  path: (string | number)[];
  errorMaps: ZodErrorMap[];
  issueData: IssueData;
}) => ZodIssue;
type ParseParams = {
  path: (string | number)[];
  errorMap: ZodErrorMap;
  async: boolean;
};
type ParsePathComponent = string | number;
type ParsePath = ParsePathComponent[];
declare const EMPTY_PATH: ParsePath;
interface ParseContext {
  readonly common: {
    readonly issues: ZodIssue[];
    readonly contextualErrorMap?: ZodErrorMap | undefined;
    readonly async: boolean;
  };
  readonly path: ParsePath;
  readonly schemaErrorMap?: ZodErrorMap | undefined;
  readonly parent: ParseContext | null;
  readonly data: any;
  readonly parsedType: ZodParsedType;
}
type ParseInput = {
  data: any;
  path: (string | number)[];
  parent: ParseContext;
};
declare function addIssueToContext(ctx: ParseContext, issueData: IssueData): void;
type ObjectPair = {
  key: SyncParseReturnType<any>;
  value: SyncParseReturnType<any>;
};
declare class ParseStatus {
  value: "aborted" | "dirty" | "valid";
  dirty(): void;
  abort(): void;
  static mergeArray(status: ParseStatus, results: SyncParseReturnType<any>[]): SyncParseReturnType;
  static mergeObjectAsync(status: ParseStatus, pairs: {
    key: ParseReturnType<any>;
    value: ParseReturnType<any>;
  }[]): Promise<SyncParseReturnType<any>>;
  static mergeObjectSync(status: ParseStatus, pairs: {
    key: SyncParseReturnType<any>;
    value: SyncParseReturnType<any>;
    alwaysSet?: boolean;
  }[]): SyncParseReturnType;
}
interface ParseResult {
  status: "aborted" | "dirty" | "valid";
  data: any;
}
type INVALID = {
  status: "aborted";
};
declare const INVALID: INVALID;
type DIRTY<T> = {
  status: "dirty";
  value: T;
};
declare const DIRTY: <T>(value: T) => DIRTY<T>;
type OK<T> = {
  status: "valid";
  value: T;
};
declare const OK: <T>(value: T) => OK<T>;
type SyncParseReturnType<T = any> = OK<T> | DIRTY<T> | INVALID;
type AsyncParseReturnType<T> = Promise<SyncParseReturnType<T>>;
type ParseReturnType<T> = SyncParseReturnType<T> | AsyncParseReturnType<T>;
declare const isAborted: (x: ParseReturnType<any>) => x is INVALID;
declare const isDirty: <T>(x: ParseReturnType<T>) => x is OK<T> | DIRTY<T>;
declare const isValid: <T>(x: ParseReturnType<T>) => x is OK<T>;
declare const isAsync: <T>(x: ParseReturnType<T>) => x is AsyncParseReturnType<T>;
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/helpers/enumUtil.d.cts
declare namespace enumUtil {
  type UnionToIntersectionFn<T> = (T extends unknown ? (k: () => T) => void : never) extends ((k: infer Intersection) => void) ? Intersection : never;
  type GetUnionLast<T> = UnionToIntersectionFn<T> extends (() => infer Last) ? Last : never;
  type UnionToTuple<T, Tuple extends unknown[] = []> = [T] extends [never] ? Tuple : UnionToTuple<Exclude<T, GetUnionLast<T>>, [GetUnionLast<T>, ...Tuple]>;
  type CastToStringTuple<T> = T extends [string, ...string[]] ? T : never;
  export type UnionToTupleString<T> = CastToStringTuple<UnionToTuple<T>>;
  export {};
}
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/helpers/errorUtil.d.cts
declare namespace errorUtil {
  type ErrMessage = string | {
    message?: string | undefined;
  };
  const errToObj: (message?: ErrMessage) => {
    message?: string | undefined;
  };
  const toString: (message?: ErrMessage) => string | undefined;
}
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/helpers/partialUtil.d.cts
declare namespace partialUtil {
  type DeepPartial<T extends ZodTypeAny> = T extends ZodObject<ZodRawShape> ? ZodObject<{ [k in keyof T["shape"]]: ZodOptional<DeepPartial<T["shape"][k]>>; }, T["_def"]["unknownKeys"], T["_def"]["catchall"]> : T extends ZodArray<infer Type, infer Card> ? ZodArray<DeepPartial<Type>, Card> : T extends ZodOptional<infer Type> ? ZodOptional<DeepPartial<Type>> : T extends ZodNullable<infer Type> ? ZodNullable<DeepPartial<Type>> : T extends ZodTuple<infer Items> ? { [k in keyof Items]: Items[k] extends ZodTypeAny ? DeepPartial<Items[k]> : never; } extends (infer PI) ? PI extends ZodTupleItems ? ZodTuple<PI> : never : never : T;
}
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/standard-schema.d.cts
/**
 * The Standard Schema interface.
 */
type StandardSchemaV1<Input = unknown, Output = Input> = {
  /**
   * The Standard Schema properties.
   */
  readonly "~standard": StandardSchemaV1.Props<Input, Output>;
};
declare namespace StandardSchemaV1 {
  /**
   * The Standard Schema properties interface.
   */
  export interface Props<Input = unknown, Output = Input> {
    /**
     * The version number of the standard.
     */
    readonly version: 1;
    /**
     * The vendor name of the schema library.
     */
    readonly vendor: string;
    /**
     * Validates unknown input values.
     */
    readonly validate: (value: unknown) => Result<Output> | Promise<Result<Output>>;
    /**
     * Inferred types associated with the schema.
     */
    readonly types?: Types<Input, Output> | undefined;
  }
  /**
   * The result interface of the validate function.
   */
  export type Result<Output> = SuccessResult<Output> | FailureResult;
  /**
   * The result interface if validation succeeds.
   */
  export interface SuccessResult<Output> {
    /**
     * The typed output value.
     */
    readonly value: Output;
    /**
     * The non-existent issues.
     */
    readonly issues?: undefined;
  }
  /**
   * The result interface if validation fails.
   */
  export interface FailureResult {
    /**
     * The issues of failed validation.
     */
    readonly issues: ReadonlyArray<Issue>;
  }
  /**
   * The issue interface of the failure output.
   */
  export interface Issue {
    /**
     * The error message of the issue.
     */
    readonly message: string;
    /**
     * The path of the issue, if any.
     */
    readonly path?: ReadonlyArray<PropertyKey | PathSegment> | undefined;
  }
  /**
   * The path segment interface of the issue.
   */
  export interface PathSegment {
    /**
     * The key representing a path segment.
     */
    readonly key: PropertyKey;
  }
  /**
   * The Standard Schema types interface.
   */
  export interface Types<Input = unknown, Output = Input> {
    /**
     * The input type of the schema.
     */
    readonly input: Input;
    /**
     * The output type of the schema.
     */
    readonly output: Output;
  }
  /**
   * Infers the input type of a Standard Schema.
   */
  export type InferInput<Schema extends StandardSchemaV1> = NonNullable<Schema["~standard"]["types"]>["input"];
  /**
   * Infers the output type of a Standard Schema.
   */
  export type InferOutput<Schema extends StandardSchemaV1> = NonNullable<Schema["~standard"]["types"]>["output"];
  export {};
}
//#endregion
//#region ../../node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/types.d.cts
interface RefinementCtx {
  addIssue: (arg: IssueData) => void;
  path: (string | number)[];
}
type ZodRawShape = {
  [k: string]: ZodTypeAny;
};
type ZodTypeAny = ZodType<any, any, any>;
type TypeOf<T extends ZodType<any, any, any>> = T["_output"];
type input<T extends ZodType<any, any, any>> = T["_input"];
type output<T extends ZodType<any, any, any>> = T["_output"];
type CustomErrorParams = Partial<util.Omit<ZodCustomIssue, "code">>;
interface ZodTypeDef {
  errorMap?: ZodErrorMap | undefined;
  description?: string | undefined;
}
type RawCreateParams = {
  errorMap?: ZodErrorMap | undefined;
  invalid_type_error?: string | undefined;
  required_error?: string | undefined;
  message?: string | undefined;
  description?: string | undefined;
} | undefined;
type ProcessedCreateParams = {
  errorMap?: ZodErrorMap | undefined;
  description?: string | undefined;
};
type SafeParseSuccess<Output> = {
  success: true;
  data: Output;
  error?: never;
};
type SafeParseError<Input> = {
  success: false;
  error: ZodError<Input>;
  data?: never;
};
type SafeParseReturnType<Input, Output> = SafeParseSuccess<Output> | SafeParseError<Input>;
declare abstract class ZodType<Output = any, Def extends ZodTypeDef = ZodTypeDef, Input = Output> {
  readonly _type: Output;
  readonly _output: Output;
  readonly _input: Input;
  readonly _def: Def;
  get description(): string | undefined;
  "~standard": StandardSchemaV1.Props<Input, Output>;
  abstract _parse(input: ParseInput): ParseReturnType<Output>;
  _getType(input: ParseInput): string;
  _getOrReturnCtx(input: ParseInput, ctx?: ParseContext | undefined): ParseContext;
  _processInputParams(input: ParseInput): {
    status: ParseStatus;
    ctx: ParseContext;
  };
  _parseSync(input: ParseInput): SyncParseReturnType<Output>;
  _parseAsync(input: ParseInput): AsyncParseReturnType<Output>;
  parse(data: unknown, params?: util.InexactPartial<ParseParams>): Output;
  safeParse(data: unknown, params?: util.InexactPartial<ParseParams>): SafeParseReturnType<Input, Output>;
  "~validate"(data: unknown): StandardSchemaV1.Result<Output> | Promise<StandardSchemaV1.Result<Output>>;
  parseAsync(data: unknown, params?: util.InexactPartial<ParseParams>): Promise<Output>;
  safeParseAsync(data: unknown, params?: util.InexactPartial<ParseParams>): Promise<SafeParseReturnType<Input, Output>>;
  /** Alias of safeParseAsync */
  spa: (data: unknown, params?: util.InexactPartial<ParseParams>) => Promise<SafeParseReturnType<Input, Output>>;
  refine<RefinedOutput extends Output>(check: (arg: Output) => arg is RefinedOutput, message?: string | CustomErrorParams | ((arg: Output) => CustomErrorParams)): ZodEffects<this, RefinedOutput, Input>;
  refine(check: (arg: Output) => unknown | Promise<unknown>, message?: string | CustomErrorParams | ((arg: Output) => CustomErrorParams)): ZodEffects<this, Output, Input>;
  refinement<RefinedOutput extends Output>(check: (arg: Output) => arg is RefinedOutput, refinementData: IssueData | ((arg: Output, ctx: RefinementCtx) => IssueData)): ZodEffects<this, RefinedOutput, Input>;
  refinement(check: (arg: Output) => boolean, refinementData: IssueData | ((arg: Output, ctx: RefinementCtx) => IssueData)): ZodEffects<this, Output, Input>;
  _refinement(refinement: RefinementEffect<Output>["refinement"]): ZodEffects<this, Output, Input>;
  superRefine<RefinedOutput extends Output>(refinement: (arg: Output, ctx: RefinementCtx) => arg is RefinedOutput): ZodEffects<this, RefinedOutput, Input>;
  superRefine(refinement: (arg: Output, ctx: RefinementCtx) => void): ZodEffects<this, Output, Input>;
  superRefine(refinement: (arg: Output, ctx: RefinementCtx) => Promise<void>): ZodEffects<this, Output, Input>;
  constructor(def: Def);
  optional(): ZodOptional<this>;
  nullable(): ZodNullable<this>;
  nullish(): ZodOptional<ZodNullable<this>>;
  array(): ZodArray<this>;
  promise(): ZodPromise<this>;
  or<T extends ZodTypeAny>(option: T): ZodUnion<[this, T]>;
  and<T extends ZodTypeAny>(incoming: T): ZodIntersection<this, T>;
  transform<NewOut>(transform: (arg: Output, ctx: RefinementCtx) => NewOut | Promise<NewOut>): ZodEffects<this, NewOut>;
  default(def: util.noUndefined<Input>): ZodDefault<this>;
  default(def: () => util.noUndefined<Input>): ZodDefault<this>;
  brand<B extends string | number | symbol>(brand?: B): ZodBranded<this, B>;
  catch(def: Output): ZodCatch<this>;
  catch(def: (ctx: {
    error: ZodError;
    input: Input;
  }) => Output): ZodCatch<this>;
  describe(description: string): this;
  pipe<T extends ZodTypeAny>(target: T): ZodPipeline<this, T>;
  readonly(): ZodReadonly<this>;
  isOptional(): boolean;
  isNullable(): boolean;
}
type IpVersion = "v4" | "v6";
type ZodStringCheck = {
  kind: "min";
  value: number;
  message?: string | undefined;
} | {
  kind: "max";
  value: number;
  message?: string | undefined;
} | {
  kind: "length";
  value: number;
  message?: string | undefined;
} | {
  kind: "email";
  message?: string | undefined;
} | {
  kind: "url";
  message?: string | undefined;
} | {
  kind: "emoji";
  message?: string | undefined;
} | {
  kind: "uuid";
  message?: string | undefined;
} | {
  kind: "nanoid";
  message?: string | undefined;
} | {
  kind: "cuid";
  message?: string | undefined;
} | {
  kind: "includes";
  value: string;
  position?: number | undefined;
  message?: string | undefined;
} | {
  kind: "cuid2";
  message?: string | undefined;
} | {
  kind: "ulid";
  message?: string | undefined;
} | {
  kind: "startsWith";
  value: string;
  message?: string | undefined;
} | {
  kind: "endsWith";
  value: string;
  message?: string | undefined;
} | {
  kind: "regex";
  regex: RegExp;
  message?: string | undefined;
} | {
  kind: "trim";
  message?: string | undefined;
} | {
  kind: "toLowerCase";
  message?: string | undefined;
} | {
  kind: "toUpperCase";
  message?: string | undefined;
} | {
  kind: "jwt";
  alg?: string;
  message?: string | undefined;
} | {
  kind: "datetime";
  offset: boolean;
  local: boolean;
  precision: number | null;
  message?: string | undefined;
} | {
  kind: "date";
  message?: string | undefined;
} | {
  kind: "time";
  precision: number | null;
  message?: string | undefined;
} | {
  kind: "duration";
  message?: string | undefined;
} | {
  kind: "ip";
  version?: IpVersion | undefined;
  message?: string | undefined;
} | {
  kind: "cidr";
  version?: IpVersion | undefined;
  message?: string | undefined;
} | {
  kind: "base64";
  message?: string | undefined;
} | {
  kind: "base64url";
  message?: string | undefined;
};
interface ZodStringDef extends ZodTypeDef {
  checks: ZodStringCheck[];
  typeName: ZodFirstPartyTypeKind.ZodString;
  coerce: boolean;
}
declare function datetimeRegex(args: {
  precision?: number | null;
  offset?: boolean;
  local?: boolean;
}): RegExp;
declare class ZodString extends ZodType<string, ZodStringDef, string> {
  _parse(input: ParseInput): ParseReturnType<string>;
  protected _regex(regex: RegExp, validation: StringValidation, message?: errorUtil.ErrMessage): ZodEffects<this, string, string>;
  _addCheck(check: ZodStringCheck): ZodString;
  email(message?: errorUtil.ErrMessage): ZodString;
  url(message?: errorUtil.ErrMessage): ZodString;
  emoji(message?: errorUtil.ErrMessage): ZodString;
  uuid(message?: errorUtil.ErrMessage): ZodString;
  nanoid(message?: errorUtil.ErrMessage): ZodString;
  cuid(message?: errorUtil.ErrMessage): ZodString;
  cuid2(message?: errorUtil.ErrMessage): ZodString;
  ulid(message?: errorUtil.ErrMessage): ZodString;
  base64(message?: errorUtil.ErrMessage): ZodString;
  base64url(message?: errorUtil.ErrMessage): ZodString;
  jwt(options?: {
    alg?: string;
    message?: string | undefined;
  }): ZodString;
  ip(options?: string | {
    version?: IpVersion;
    message?: string | undefined;
  }): ZodString;
  cidr(options?: string | {
    version?: IpVersion;
    message?: string | undefined;
  }): ZodString;
  datetime(options?: string | {
    message?: string | undefined;
    precision?: number | null;
    offset?: boolean;
    local?: boolean;
  }): ZodString;
  date(message?: string): ZodString;
  time(options?: string | {
    message?: string | undefined;
    precision?: number | null;
  }): ZodString;
  duration(message?: errorUtil.ErrMessage): ZodString;
  regex(regex: RegExp, message?: errorUtil.ErrMessage): ZodString;
  includes(value: string, options?: {
    message?: string;
    position?: number;
  }): ZodString;
  startsWith(value: string, message?: errorUtil.ErrMessage): ZodString;
  endsWith(value: string, message?: errorUtil.ErrMessage): ZodString;
  min(minLength: number, message?: errorUtil.ErrMessage): ZodString;
  max(maxLength: number, message?: errorUtil.ErrMessage): ZodString;
  length(len: number, message?: errorUtil.ErrMessage): ZodString;
  /**
   * Equivalent to `.min(1)`
   */
  nonempty(message?: errorUtil.ErrMessage): ZodString;
  trim(): ZodString;
  toLowerCase(): ZodString;
  toUpperCase(): ZodString;
  get isDatetime(): boolean;
  get isDate(): boolean;
  get isTime(): boolean;
  get isDuration(): boolean;
  get isEmail(): boolean;
  get isURL(): boolean;
  get isEmoji(): boolean;
  get isUUID(): boolean;
  get isNANOID(): boolean;
  get isCUID(): boolean;
  get isCUID2(): boolean;
  get isULID(): boolean;
  get isIP(): boolean;
  get isCIDR(): boolean;
  get isBase64(): boolean;
  get isBase64url(): boolean;
  get minLength(): number | null;
  get maxLength(): number | null;
  static create: (params?: RawCreateParams & {
    coerce?: true;
  }) => ZodString;
}
type ZodNumberCheck = {
  kind: "min";
  value: number;
  inclusive: boolean;
  message?: string | undefined;
} | {
  kind: "max";
  value: number;
  inclusive: boolean;
  message?: string | undefined;
} | {
  kind: "int";
  message?: string | undefined;
} | {
  kind: "multipleOf";
  value: number;
  message?: string | undefined;
} | {
  kind: "finite";
  message?: string | undefined;
};
interface ZodNumberDef extends ZodTypeDef {
  checks: ZodNumberCheck[];
  typeName: ZodFirstPartyTypeKind.ZodNumber;
  coerce: boolean;
}
declare class ZodNumber extends ZodType<number, ZodNumberDef, number> {
  _parse(input: ParseInput): ParseReturnType<number>;
  static create: (params?: RawCreateParams & {
    coerce?: boolean;
  }) => ZodNumber;
  gte(value: number, message?: errorUtil.ErrMessage): ZodNumber;
  min: (value: number, message?: errorUtil.ErrMessage) => ZodNumber;
  gt(value: number, message?: errorUtil.ErrMessage): ZodNumber;
  lte(value: number, message?: errorUtil.ErrMessage): ZodNumber;
  max: (value: number, message?: errorUtil.ErrMessage) => ZodNumber;
  lt(value: number, message?: errorUtil.ErrMessage): ZodNumber;
  protected setLimit(kind: "min" | "max", value: number, inclusive: boolean, message?: string): ZodNumber;
  _addCheck(check: ZodNumberCheck): ZodNumber;
  int(message?: errorUtil.ErrMessage): ZodNumber;
  positive(message?: errorUtil.ErrMessage): ZodNumber;
  negative(message?: errorUtil.ErrMessage): ZodNumber;
  nonpositive(message?: errorUtil.ErrMessage): ZodNumber;
  nonnegative(message?: errorUtil.ErrMessage): ZodNumber;
  multipleOf(value: number, message?: errorUtil.ErrMessage): ZodNumber;
  step: (value: number, message?: errorUtil.ErrMessage) => ZodNumber;
  finite(message?: errorUtil.ErrMessage): ZodNumber;
  safe(message?: errorUtil.ErrMessage): ZodNumber;
  get minValue(): number | null;
  get maxValue(): number | null;
  get isInt(): boolean;
  get isFinite(): boolean;
}
type ZodBigIntCheck = {
  kind: "min";
  value: bigint;
  inclusive: boolean;
  message?: string | undefined;
} | {
  kind: "max";
  value: bigint;
  inclusive: boolean;
  message?: string | undefined;
} | {
  kind: "multipleOf";
  value: bigint;
  message?: string | undefined;
};
interface ZodBigIntDef extends ZodTypeDef {
  checks: ZodBigIntCheck[];
  typeName: ZodFirstPartyTypeKind.ZodBigInt;
  coerce: boolean;
}
declare class ZodBigInt extends ZodType<bigint, ZodBigIntDef, bigint> {
  _parse(input: ParseInput): ParseReturnType<bigint>;
  _getInvalidInput(input: ParseInput): INVALID;
  static create: (params?: RawCreateParams & {
    coerce?: boolean;
  }) => ZodBigInt;
  gte(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
  min: (value: bigint, message?: errorUtil.ErrMessage) => ZodBigInt;
  gt(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
  lte(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
  max: (value: bigint, message?: errorUtil.ErrMessage) => ZodBigInt;
  lt(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
  protected setLimit(kind: "min" | "max", value: bigint, inclusive: boolean, message?: string): ZodBigInt;
  _addCheck(check: ZodBigIntCheck): ZodBigInt;
  positive(message?: errorUtil.ErrMessage): ZodBigInt;
  negative(message?: errorUtil.ErrMessage): ZodBigInt;
  nonpositive(message?: errorUtil.ErrMessage): ZodBigInt;
  nonnegative(message?: errorUtil.ErrMessage): ZodBigInt;
  multipleOf(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
  get minValue(): bigint | null;
  get maxValue(): bigint | null;
}
interface ZodBooleanDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodBoolean;
  coerce: boolean;
}
declare class ZodBoolean extends ZodType<boolean, ZodBooleanDef, boolean> {
  _parse(input: ParseInput): ParseReturnType<boolean>;
  static create: (params?: RawCreateParams & {
    coerce?: boolean;
  }) => ZodBoolean;
}
type ZodDateCheck = {
  kind: "min";
  value: number;
  message?: string | undefined;
} | {
  kind: "max";
  value: number;
  message?: string | undefined;
};
interface ZodDateDef extends ZodTypeDef {
  checks: ZodDateCheck[];
  coerce: boolean;
  typeName: ZodFirstPartyTypeKind.ZodDate;
}
declare class ZodDate extends ZodType<Date, ZodDateDef, Date> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  _addCheck(check: ZodDateCheck): ZodDate;
  min(minDate: Date, message?: errorUtil.ErrMessage): ZodDate;
  max(maxDate: Date, message?: errorUtil.ErrMessage): ZodDate;
  get minDate(): Date | null;
  get maxDate(): Date | null;
  static create: (params?: RawCreateParams & {
    coerce?: boolean;
  }) => ZodDate;
}
interface ZodSymbolDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodSymbol;
}
declare class ZodSymbol extends ZodType<symbol, ZodSymbolDef, symbol> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: (params?: RawCreateParams) => ZodSymbol;
}
interface ZodUndefinedDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodUndefined;
}
declare class ZodUndefined extends ZodType<undefined, ZodUndefinedDef, undefined> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  params?: RawCreateParams;
  static create: (params?: RawCreateParams) => ZodUndefined;
}
interface ZodNullDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodNull;
}
declare class ZodNull extends ZodType<null, ZodNullDef, null> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: (params?: RawCreateParams) => ZodNull;
}
interface ZodAnyDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodAny;
}
declare class ZodAny extends ZodType<any, ZodAnyDef, any> {
  _any: true;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: (params?: RawCreateParams) => ZodAny;
}
interface ZodUnknownDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodUnknown;
}
declare class ZodUnknown extends ZodType<unknown, ZodUnknownDef, unknown> {
  _unknown: true;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: (params?: RawCreateParams) => ZodUnknown;
}
interface ZodNeverDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodNever;
}
declare class ZodNever extends ZodType<never, ZodNeverDef, never> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: (params?: RawCreateParams) => ZodNever;
}
interface ZodVoidDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodVoid;
}
declare class ZodVoid extends ZodType<void, ZodVoidDef, void> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: (params?: RawCreateParams) => ZodVoid;
}
interface ZodArrayDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  type: T;
  typeName: ZodFirstPartyTypeKind.ZodArray;
  exactLength: {
    value: number;
    message?: string | undefined;
  } | null;
  minLength: {
    value: number;
    message?: string | undefined;
  } | null;
  maxLength: {
    value: number;
    message?: string | undefined;
  } | null;
}
type ArrayCardinality = "many" | "atleastone";
type arrayOutputType<T extends ZodTypeAny, Cardinality extends ArrayCardinality = "many"> = Cardinality extends "atleastone" ? [T["_output"], ...T["_output"][]] : T["_output"][];
declare class ZodArray<T extends ZodTypeAny, Cardinality extends ArrayCardinality = "many"> extends ZodType<arrayOutputType<T, Cardinality>, ZodArrayDef<T>, Cardinality extends "atleastone" ? [T["_input"], ...T["_input"][]] : T["_input"][]> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get element(): T;
  min(minLength: number, message?: errorUtil.ErrMessage): this;
  max(maxLength: number, message?: errorUtil.ErrMessage): this;
  length(len: number, message?: errorUtil.ErrMessage): this;
  nonempty(message?: errorUtil.ErrMessage): ZodArray<T, "atleastone">;
  static create: <El extends ZodTypeAny>(schema: El, params?: RawCreateParams) => ZodArray<El>;
}
type ZodNonEmptyArray<T extends ZodTypeAny> = ZodArray<T, "atleastone">;
type UnknownKeysParam = "passthrough" | "strict" | "strip";
interface ZodObjectDef<T extends ZodRawShape = ZodRawShape, UnknownKeys extends UnknownKeysParam = UnknownKeysParam, Catchall extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodObject;
  shape: () => T;
  catchall: Catchall;
  unknownKeys: UnknownKeys;
}
type mergeTypes<A, B> = { [k in keyof A | keyof B]: k extends keyof B ? B[k] : k extends keyof A ? A[k] : never; };
type objectOutputType<Shape extends ZodRawShape, Catchall extends ZodTypeAny, UnknownKeys extends UnknownKeysParam = UnknownKeysParam> = objectUtil.flatten<objectUtil.addQuestionMarks<baseObjectOutputType<Shape>>> & CatchallOutput<Catchall> & PassthroughType<UnknownKeys>;
type baseObjectOutputType<Shape extends ZodRawShape> = { [k in keyof Shape]: Shape[k]["_output"]; };
type objectInputType<Shape extends ZodRawShape, Catchall extends ZodTypeAny, UnknownKeys extends UnknownKeysParam = UnknownKeysParam> = objectUtil.flatten<baseObjectInputType<Shape>> & CatchallInput<Catchall> & PassthroughType<UnknownKeys>;
type baseObjectInputType<Shape extends ZodRawShape> = objectUtil.addQuestionMarks<{ [k in keyof Shape]: Shape[k]["_input"]; }>;
type CatchallOutput<T extends ZodType> = ZodType extends T ? unknown : {
  [k: string]: T["_output"];
};
type CatchallInput<T extends ZodType> = ZodType extends T ? unknown : {
  [k: string]: T["_input"];
};
type PassthroughType<T extends UnknownKeysParam> = T extends "passthrough" ? {
  [k: string]: unknown;
} : unknown;
type deoptional<T extends ZodTypeAny> = T extends ZodOptional<infer U> ? deoptional<U> : T extends ZodNullable<infer U> ? ZodNullable<deoptional<U>> : T;
type SomeZodObject = ZodObject<ZodRawShape, UnknownKeysParam, ZodTypeAny>;
type noUnrecognized<Obj extends object, Shape extends object> = { [k in keyof Obj]: k extends keyof Shape ? Obj[k] : never; };
declare class ZodObject<T extends ZodRawShape, UnknownKeys extends UnknownKeysParam = UnknownKeysParam, Catchall extends ZodTypeAny = ZodTypeAny, Output = objectOutputType<T, Catchall, UnknownKeys>, Input = objectInputType<T, Catchall, UnknownKeys>> extends ZodType<Output, ZodObjectDef<T, UnknownKeys, Catchall>, Input> {
  private _cached;
  _getCached(): {
    shape: T;
    keys: string[];
  };
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get shape(): T;
  strict(message?: errorUtil.ErrMessage): ZodObject<T, "strict", Catchall>;
  strip(): ZodObject<T, "strip", Catchall>;
  passthrough(): ZodObject<T, "passthrough", Catchall>;
  /**
   * @deprecated In most cases, this is no longer needed - unknown properties are now silently stripped.
   * If you want to pass through unknown properties, use `.passthrough()` instead.
   */
  nonstrict: () => ZodObject<T, "passthrough", Catchall>;
  extend<Augmentation extends ZodRawShape>(augmentation: Augmentation): ZodObject<objectUtil.extendShape<T, Augmentation>, UnknownKeys, Catchall>;
  /**
   * @deprecated Use `.extend` instead
   *  */
  augment: <Augmentation extends ZodRawShape>(augmentation: Augmentation) => ZodObject<objectUtil.extendShape<T, Augmentation>, UnknownKeys, Catchall>;
  /**
   * Prior to zod@1.0.12 there was a bug in the
   * inferred type of merged objects. Please
   * upgrade if you are experiencing issues.
   */
  merge<Incoming extends AnyZodObject, Augmentation extends Incoming["shape"]>(merging: Incoming): ZodObject<objectUtil.extendShape<T, Augmentation>, Incoming["_def"]["unknownKeys"], Incoming["_def"]["catchall"]>;
  setKey<Key extends string, Schema extends ZodTypeAny>(key: Key, schema: Schema): ZodObject<T & { [k in Key]: Schema; }, UnknownKeys, Catchall>;
  catchall<Index extends ZodTypeAny>(index: Index): ZodObject<T, UnknownKeys, Index>;
  pick<Mask extends util.Exactly<{ [k in keyof T]?: true; }, Mask>>(mask: Mask): ZodObject<Pick<T, Extract<keyof T, keyof Mask>>, UnknownKeys, Catchall>;
  omit<Mask extends util.Exactly<{ [k in keyof T]?: true; }, Mask>>(mask: Mask): ZodObject<Omit<T, keyof Mask>, UnknownKeys, Catchall>;
  /**
   * @deprecated
   */
  deepPartial(): partialUtil.DeepPartial<this>;
  partial(): ZodObject<{ [k in keyof T]: ZodOptional<T[k]>; }, UnknownKeys, Catchall>;
  partial<Mask extends util.Exactly<{ [k in keyof T]?: true; }, Mask>>(mask: Mask): ZodObject<objectUtil.noNever<{ [k in keyof T]: k extends keyof Mask ? ZodOptional<T[k]> : T[k]; }>, UnknownKeys, Catchall>;
  required(): ZodObject<{ [k in keyof T]: deoptional<T[k]>; }, UnknownKeys, Catchall>;
  required<Mask extends util.Exactly<{ [k in keyof T]?: true; }, Mask>>(mask: Mask): ZodObject<objectUtil.noNever<{ [k in keyof T]: k extends keyof Mask ? deoptional<T[k]> : T[k]; }>, UnknownKeys, Catchall>;
  keyof(): ZodEnum<enumUtil.UnionToTupleString<keyof T>>;
  static create: <Shape extends ZodRawShape>(shape: Shape, params?: RawCreateParams) => ZodObject<Shape, "strip", ZodTypeAny, objectOutputType<Shape, ZodTypeAny, "strip">, objectInputType<Shape, ZodTypeAny, "strip">>;
  static strictCreate: <Shape extends ZodRawShape>(shape: Shape, params?: RawCreateParams) => ZodObject<Shape, "strict">;
  static lazycreate: <Shape extends ZodRawShape>(shape: () => Shape, params?: RawCreateParams) => ZodObject<Shape, "strip">;
}
type AnyZodObject = ZodObject<any, any, any>;
type ZodUnionOptions = Readonly<[ZodTypeAny, ...ZodTypeAny[]]>;
interface ZodUnionDef<T extends ZodUnionOptions = Readonly<[ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]>> extends ZodTypeDef {
  options: T;
  typeName: ZodFirstPartyTypeKind.ZodUnion;
}
declare class ZodUnion<T extends ZodUnionOptions> extends ZodType<T[number]["_output"], ZodUnionDef<T>, T[number]["_input"]> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get options(): T;
  static create: <Options extends Readonly<[ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]>>(types: Options, params?: RawCreateParams) => ZodUnion<Options>;
}
type ZodDiscriminatedUnionOption<Discriminator extends string> = ZodObject<{ [key in Discriminator]: ZodTypeAny; } & ZodRawShape, UnknownKeysParam, ZodTypeAny>;
interface ZodDiscriminatedUnionDef<Discriminator extends string, Options extends readonly ZodDiscriminatedUnionOption<string>[] = ZodDiscriminatedUnionOption<string>[]> extends ZodTypeDef {
  discriminator: Discriminator;
  options: Options;
  optionsMap: Map<Primitive$1, ZodDiscriminatedUnionOption<any>>;
  typeName: ZodFirstPartyTypeKind.ZodDiscriminatedUnion;
}
declare class ZodDiscriminatedUnion<Discriminator extends string, Options extends readonly ZodDiscriminatedUnionOption<Discriminator>[]> extends ZodType<output<Options[number]>, ZodDiscriminatedUnionDef<Discriminator, Options>, input<Options[number]>> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get discriminator(): Discriminator;
  get options(): Options;
  get optionsMap(): Map<Primitive$1, ZodDiscriminatedUnionOption<any>>;
  /**
   * The constructor of the discriminated union schema. Its behaviour is very similar to that of the normal z.union() constructor.
   * However, it only allows a union of objects, all of which need to share a discriminator property. This property must
   * have a different value for each object in the union.
   * @param discriminator the name of the discriminator property
   * @param types an array of object schemas
   * @param params
   */
  static create<Discriminator extends string, Types extends readonly [ZodDiscriminatedUnionOption<Discriminator>, ...ZodDiscriminatedUnionOption<Discriminator>[]]>(discriminator: Discriminator, options: Types, params?: RawCreateParams): ZodDiscriminatedUnion<Discriminator, Types>;
}
interface ZodIntersectionDef<T extends ZodTypeAny = ZodTypeAny, U extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  left: T;
  right: U;
  typeName: ZodFirstPartyTypeKind.ZodIntersection;
}
declare class ZodIntersection<T extends ZodTypeAny, U extends ZodTypeAny> extends ZodType<T["_output"] & U["_output"], ZodIntersectionDef<T, U>, T["_input"] & U["_input"]> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: <TSchema extends ZodTypeAny, USchema extends ZodTypeAny>(left: TSchema, right: USchema, params?: RawCreateParams) => ZodIntersection<TSchema, USchema>;
}
type ZodTupleItems = [ZodTypeAny, ...ZodTypeAny[]];
type AssertArray<T> = T extends any[] ? T : never;
type OutputTypeOfTuple<T extends ZodTupleItems | []> = AssertArray<{ [k in keyof T]: T[k] extends ZodType<any, any, any> ? T[k]["_output"] : never; }>;
type OutputTypeOfTupleWithRest<T extends ZodTupleItems | [], Rest extends ZodTypeAny | null = null> = Rest extends ZodTypeAny ? [...OutputTypeOfTuple<T>, ...Rest["_output"][]] : OutputTypeOfTuple<T>;
type InputTypeOfTuple<T extends ZodTupleItems | []> = AssertArray<{ [k in keyof T]: T[k] extends ZodType<any, any, any> ? T[k]["_input"] : never; }>;
type InputTypeOfTupleWithRest<T extends ZodTupleItems | [], Rest extends ZodTypeAny | null = null> = Rest extends ZodTypeAny ? [...InputTypeOfTuple<T>, ...Rest["_input"][]] : InputTypeOfTuple<T>;
interface ZodTupleDef<T extends ZodTupleItems | [] = ZodTupleItems, Rest extends ZodTypeAny | null = null> extends ZodTypeDef {
  items: T;
  rest: Rest;
  typeName: ZodFirstPartyTypeKind.ZodTuple;
}
type AnyZodTuple = ZodTuple<[ZodTypeAny, ...ZodTypeAny[]] | [], ZodTypeAny | null>;
declare class ZodTuple<T extends ZodTupleItems | [] = ZodTupleItems, Rest extends ZodTypeAny | null = null> extends ZodType<OutputTypeOfTupleWithRest<T, Rest>, ZodTupleDef<T, Rest>, InputTypeOfTupleWithRest<T, Rest>> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get items(): T;
  rest<RestSchema extends ZodTypeAny>(rest: RestSchema): ZodTuple<T, RestSchema>;
  static create: <Items extends [ZodTypeAny, ...ZodTypeAny[]] | []>(schemas: Items, params?: RawCreateParams) => ZodTuple<Items, null>;
}
interface ZodRecordDef<Key extends KeySchema = ZodString, Value extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  valueType: Value;
  keyType: Key;
  typeName: ZodFirstPartyTypeKind.ZodRecord;
}
type KeySchema = ZodType<string | number | symbol, any, any>;
type RecordType<K extends string | number | symbol, V> = [string] extends [K] ? Record<K, V> : [number] extends [K] ? Record<K, V> : [symbol] extends [K] ? Record<K, V> : [BRAND<string | number | symbol>] extends [K] ? Record<K, V> : Partial<Record<K, V>>;
declare class ZodRecord<Key extends KeySchema = ZodString, Value extends ZodTypeAny = ZodTypeAny> extends ZodType<RecordType<Key["_output"], Value["_output"]>, ZodRecordDef<Key, Value>, RecordType<Key["_input"], Value["_input"]>> {
  get keySchema(): Key;
  get valueSchema(): Value;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get element(): Value;
  static create<Value extends ZodTypeAny>(valueType: Value, params?: RawCreateParams): ZodRecord<ZodString, Value>;
  static create<Keys extends KeySchema, Value extends ZodTypeAny>(keySchema: Keys, valueType: Value, params?: RawCreateParams): ZodRecord<Keys, Value>;
}
interface ZodMapDef<Key extends ZodTypeAny = ZodTypeAny, Value extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  valueType: Value;
  keyType: Key;
  typeName: ZodFirstPartyTypeKind.ZodMap;
}
declare class ZodMap<Key extends ZodTypeAny = ZodTypeAny, Value extends ZodTypeAny = ZodTypeAny> extends ZodType<Map<Key["_output"], Value["_output"]>, ZodMapDef<Key, Value>, Map<Key["_input"], Value["_input"]>> {
  get keySchema(): Key;
  get valueSchema(): Value;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: <KeySchema extends ZodTypeAny = ZodTypeAny, ValueSchema extends ZodTypeAny = ZodTypeAny>(keyType: KeySchema, valueType: ValueSchema, params?: RawCreateParams) => ZodMap<KeySchema, ValueSchema>;
}
interface ZodSetDef<Value extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  valueType: Value;
  typeName: ZodFirstPartyTypeKind.ZodSet;
  minSize: {
    value: number;
    message?: string | undefined;
  } | null;
  maxSize: {
    value: number;
    message?: string | undefined;
  } | null;
}
declare class ZodSet<Value extends ZodTypeAny = ZodTypeAny> extends ZodType<Set<Value["_output"]>, ZodSetDef<Value>, Set<Value["_input"]>> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  min(minSize: number, message?: errorUtil.ErrMessage): this;
  max(maxSize: number, message?: errorUtil.ErrMessage): this;
  size(size: number, message?: errorUtil.ErrMessage): this;
  nonempty(message?: errorUtil.ErrMessage): ZodSet<Value>;
  static create: <ValueSchema extends ZodTypeAny = ZodTypeAny>(valueType: ValueSchema, params?: RawCreateParams) => ZodSet<ValueSchema>;
}
interface ZodFunctionDef<Args extends ZodTuple<any, any> = ZodTuple<any, any>, Returns extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  args: Args;
  returns: Returns;
  typeName: ZodFirstPartyTypeKind.ZodFunction;
}
type OuterTypeOfFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> = Args["_input"] extends Array<any> ? (...args: Args["_input"]) => Returns["_output"] : never;
type InnerTypeOfFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> = Args["_output"] extends Array<any> ? (...args: Args["_output"]) => Returns["_input"] : never;
declare class ZodFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> extends ZodType<OuterTypeOfFunction<Args, Returns>, ZodFunctionDef<Args, Returns>, InnerTypeOfFunction<Args, Returns>> {
  _parse(input: ParseInput): ParseReturnType<any>;
  parameters(): Args;
  returnType(): Returns;
  args<Items extends Parameters<(typeof ZodTuple)["create"]>[0]>(...items: Items): ZodFunction<ZodTuple<Items, ZodUnknown>, Returns>;
  returns<NewReturnType extends ZodType<any, any, any>>(returnType: NewReturnType): ZodFunction<Args, NewReturnType>;
  implement<F extends InnerTypeOfFunction<Args, Returns>>(func: F): ReturnType<F> extends Returns["_output"] ? (...args: Args["_input"]) => ReturnType<F> : OuterTypeOfFunction<Args, Returns>;
  strictImplement(func: InnerTypeOfFunction<Args, Returns>): InnerTypeOfFunction<Args, Returns>;
  validate: <F extends InnerTypeOfFunction<Args, Returns>>(func: F) => ReturnType<F> extends Returns["_output"] ? (...args: Args["_input"]) => ReturnType<F> : OuterTypeOfFunction<Args, Returns>;
  static create(): ZodFunction<ZodTuple<[], ZodUnknown>, ZodUnknown>;
  static create<T extends AnyZodTuple = ZodTuple<[], ZodUnknown>>(args: T): ZodFunction<T, ZodUnknown>;
  static create<T extends AnyZodTuple, U extends ZodTypeAny>(args: T, returns: U): ZodFunction<T, U>;
  static create<T extends AnyZodTuple = ZodTuple<[], ZodUnknown>, U extends ZodTypeAny = ZodUnknown>(args: T, returns: U, params?: RawCreateParams): ZodFunction<T, U>;
}
interface ZodLazyDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  getter: () => T;
  typeName: ZodFirstPartyTypeKind.ZodLazy;
}
declare class ZodLazy<T extends ZodTypeAny> extends ZodType<output<T>, ZodLazyDef<T>, input<T>> {
  get schema(): T;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: <Inner extends ZodTypeAny>(getter: () => Inner, params?: RawCreateParams) => ZodLazy<Inner>;
}
interface ZodLiteralDef<T = any> extends ZodTypeDef {
  value: T;
  typeName: ZodFirstPartyTypeKind.ZodLiteral;
}
declare class ZodLiteral<T> extends ZodType<T, ZodLiteralDef<T>, T> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get value(): T;
  static create: <Value extends Primitive$1>(value: Value, params?: RawCreateParams) => ZodLiteral<Value>;
}
type ArrayKeys = keyof any[];
type Indices<T> = Exclude<keyof T, ArrayKeys>;
type EnumValues<T extends string = string> = readonly [T, ...T[]];
type Values<T extends EnumValues> = { [k in T[number]]: k; };
interface ZodEnumDef<T extends EnumValues = EnumValues> extends ZodTypeDef {
  values: T;
  typeName: ZodFirstPartyTypeKind.ZodEnum;
}
type Writeable<T> = { -readonly [P in keyof T]: T[P]; };
type FilterEnum<Values, ToExclude> = Values extends [] ? [] : Values extends [infer Head, ...infer Rest] ? Head extends ToExclude ? FilterEnum<Rest, ToExclude> : [Head, ...FilterEnum<Rest, ToExclude>] : never;
type typecast<A, T> = A extends T ? A : never;
declare function createZodEnum<U extends string, T extends Readonly<[U, ...U[]]>>(values: T, params?: RawCreateParams): ZodEnum<Writeable<T>>;
declare function createZodEnum<U extends string, T extends [U, ...U[]]>(values: T, params?: RawCreateParams): ZodEnum<T>;
declare class ZodEnum<T extends [string, ...string[]]> extends ZodType<T[number], ZodEnumDef<T>, T[number]> {
  _cache: Set<T[number]> | undefined;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  get options(): T;
  get enum(): Values<T>;
  get Values(): Values<T>;
  get Enum(): Values<T>;
  extract<ToExtract extends readonly [T[number], ...T[number][]]>(values: ToExtract, newDef?: RawCreateParams): ZodEnum<Writeable<ToExtract>>;
  exclude<ToExclude extends readonly [T[number], ...T[number][]]>(values: ToExclude, newDef?: RawCreateParams): ZodEnum<typecast<Writeable<FilterEnum<T, ToExclude[number]>>, [string, ...string[]]>>;
  static create: typeof createZodEnum;
}
interface ZodNativeEnumDef<T extends EnumLike = EnumLike> extends ZodTypeDef {
  values: T;
  typeName: ZodFirstPartyTypeKind.ZodNativeEnum;
}
type EnumLike = {
  [k: string]: string | number;
  [nu: number]: string;
};
declare class ZodNativeEnum<T extends EnumLike> extends ZodType<T[keyof T], ZodNativeEnumDef<T>, T[keyof T]> {
  _cache: Set<T[keyof T]> | undefined;
  _parse(input: ParseInput): ParseReturnType<T[keyof T]>;
  get enum(): T;
  static create: <Elements extends EnumLike>(values: Elements, params?: RawCreateParams) => ZodNativeEnum<Elements>;
}
interface ZodPromiseDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  type: T;
  typeName: ZodFirstPartyTypeKind.ZodPromise;
}
declare class ZodPromise<T extends ZodTypeAny> extends ZodType<Promise<T["_output"]>, ZodPromiseDef<T>, Promise<T["_input"]>> {
  unwrap(): T;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: <Inner extends ZodTypeAny>(schema: Inner, params?: RawCreateParams) => ZodPromise<Inner>;
}
type Refinement<T> = (arg: T, ctx: RefinementCtx) => any;
type SuperRefinement<T> = (arg: T, ctx: RefinementCtx) => void | Promise<void>;
type RefinementEffect<T> = {
  type: "refinement";
  refinement: (arg: T, ctx: RefinementCtx) => any;
};
type TransformEffect<T> = {
  type: "transform";
  transform: (arg: T, ctx: RefinementCtx) => any;
};
type PreprocessEffect<T> = {
  type: "preprocess";
  transform: (arg: T, ctx: RefinementCtx) => any;
};
type Effect<T> = RefinementEffect<T> | TransformEffect<T> | PreprocessEffect<T>;
interface ZodEffectsDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  schema: T;
  typeName: ZodFirstPartyTypeKind.ZodEffects;
  effect: Effect<any>;
}
declare class ZodEffects<T extends ZodTypeAny, Output = output<T>, Input = input<T>> extends ZodType<Output, ZodEffectsDef<T>, Input> {
  innerType(): T;
  sourceType(): T;
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: <I extends ZodTypeAny>(schema: I, effect: Effect<I["_output"]>, params?: RawCreateParams) => ZodEffects<I, I["_output"]>;
  static createWithPreprocess: <I extends ZodTypeAny>(preprocess: (arg: unknown, ctx: RefinementCtx) => unknown, schema: I, params?: RawCreateParams) => ZodEffects<I, I["_output"], unknown>;
}
interface ZodOptionalDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  innerType: T;
  typeName: ZodFirstPartyTypeKind.ZodOptional;
}
type ZodOptionalType<T extends ZodTypeAny> = ZodOptional<T>;
declare class ZodOptional<T extends ZodTypeAny> extends ZodType<T["_output"] | undefined, ZodOptionalDef<T>, T["_input"] | undefined> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  unwrap(): T;
  static create: <Inner extends ZodTypeAny>(type: Inner, params?: RawCreateParams) => ZodOptional<Inner>;
}
interface ZodNullableDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  innerType: T;
  typeName: ZodFirstPartyTypeKind.ZodNullable;
}
type ZodNullableType<T extends ZodTypeAny> = ZodNullable<T>;
declare class ZodNullable<T extends ZodTypeAny> extends ZodType<T["_output"] | null, ZodNullableDef<T>, T["_input"] | null> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  unwrap(): T;
  static create: <Inner extends ZodTypeAny>(type: Inner, params?: RawCreateParams) => ZodNullable<Inner>;
}
interface ZodDefaultDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  innerType: T;
  defaultValue: () => util.noUndefined<T["_input"]>;
  typeName: ZodFirstPartyTypeKind.ZodDefault;
}
declare class ZodDefault<T extends ZodTypeAny> extends ZodType<util.noUndefined<T["_output"]>, ZodDefaultDef<T>, T["_input"] | undefined> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  removeDefault(): T;
  static create: <Inner extends ZodTypeAny>(type: Inner, params: RawCreateParams & {
    default: Inner["_input"] | (() => util.noUndefined<Inner["_input"]>);
  }) => ZodDefault<Inner>;
}
interface ZodCatchDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  innerType: T;
  catchValue: (ctx: {
    error: ZodError;
    input: unknown;
  }) => T["_input"];
  typeName: ZodFirstPartyTypeKind.ZodCatch;
}
declare class ZodCatch<T extends ZodTypeAny> extends ZodType<T["_output"], ZodCatchDef<T>, unknown> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  removeCatch(): T;
  static create: <Inner extends ZodTypeAny>(type: Inner, params: RawCreateParams & {
    catch: Inner["_output"] | (() => Inner["_output"]);
  }) => ZodCatch<Inner>;
}
interface ZodNaNDef extends ZodTypeDef {
  typeName: ZodFirstPartyTypeKind.ZodNaN;
}
declare class ZodNaN extends ZodType<number, ZodNaNDef, number> {
  _parse(input: ParseInput): ParseReturnType<any>;
  static create: (params?: RawCreateParams) => ZodNaN;
}
interface ZodBrandedDef<T extends ZodTypeAny> extends ZodTypeDef {
  type: T;
  typeName: ZodFirstPartyTypeKind.ZodBranded;
}
declare const BRAND: unique symbol;
type BRAND<T extends string | number | symbol> = {
  [BRAND]: { [k in T]: true; };
};
declare class ZodBranded<T extends ZodTypeAny, B extends string | number | symbol> extends ZodType<T["_output"] & BRAND<B>, ZodBrandedDef<T>, T["_input"]> {
  _parse(input: ParseInput): ParseReturnType<any>;
  unwrap(): T;
}
interface ZodPipelineDef<A extends ZodTypeAny, B extends ZodTypeAny> extends ZodTypeDef {
  in: A;
  out: B;
  typeName: ZodFirstPartyTypeKind.ZodPipeline;
}
declare class ZodPipeline<A extends ZodTypeAny, B extends ZodTypeAny> extends ZodType<B["_output"], ZodPipelineDef<A, B>, A["_input"]> {
  _parse(input: ParseInput): ParseReturnType<any>;
  static create<ASchema extends ZodTypeAny, BSchema extends ZodTypeAny>(a: ASchema, b: BSchema): ZodPipeline<ASchema, BSchema>;
}
type BuiltIn = (((...args: any[]) => any) | (new (...args: any[]) => any)) | {
  readonly [Symbol.toStringTag]: string;
} | Date | Error | Generator | Promise<unknown> | RegExp;
type MakeReadonly<T> = T extends Map<infer K, infer V> ? ReadonlyMap<K, V> : T extends Set<infer V> ? ReadonlySet<V> : T extends [infer Head, ...infer Tail] ? readonly [Head, ...Tail] : T extends Array<infer V> ? ReadonlyArray<V> : T extends BuiltIn ? T : Readonly<T>;
interface ZodReadonlyDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
  innerType: T;
  typeName: ZodFirstPartyTypeKind.ZodReadonly;
}
declare class ZodReadonly<T extends ZodTypeAny> extends ZodType<MakeReadonly<T["_output"]>, ZodReadonlyDef<T>, MakeReadonly<T["_input"]>> {
  _parse(input: ParseInput): ParseReturnType<this["_output"]>;
  static create: <Inner extends ZodTypeAny>(type: Inner, params?: RawCreateParams) => ZodReadonly<Inner>;
  unwrap(): T;
}
type CustomParams = CustomErrorParams & {
  fatal?: boolean;
};
declare function custom<T>(check?: (data: any) => any, _params?: string | CustomParams | ((input: any) => CustomParams),
/**
 * @deprecated
 *
 * Pass `fatal` into the params object instead:
 *
 * ```ts
 * z.string().custom((val) => val.length > 5, { fatal: false })
 * ```
 *
 */
fatal?: boolean): ZodType<T, ZodTypeDef, T>;
declare const late: {
  object: <Shape extends ZodRawShape>(shape: () => Shape, params?: RawCreateParams) => ZodObject<Shape, "strip">;
};
declare enum ZodFirstPartyTypeKind {
  ZodString = "ZodString",
  ZodNumber = "ZodNumber",
  ZodNaN = "ZodNaN",
  ZodBigInt = "ZodBigInt",
  ZodBoolean = "ZodBoolean",
  ZodDate = "ZodDate",
  ZodSymbol = "ZodSymbol",
  ZodUndefined = "ZodUndefined",
  ZodNull = "ZodNull",
  ZodAny = "ZodAny",
  ZodUnknown = "ZodUnknown",
  ZodNever = "ZodNever",
  ZodVoid = "ZodVoid",
  ZodArray = "ZodArray",
  ZodObject = "ZodObject",
  ZodUnion = "ZodUnion",
  ZodDiscriminatedUnion = "ZodDiscriminatedUnion",
  ZodIntersection = "ZodIntersection",
  ZodTuple = "ZodTuple",
  ZodRecord = "ZodRecord",
  ZodMap = "ZodMap",
  ZodSet = "ZodSet",
  ZodFunction = "ZodFunction",
  ZodLazy = "ZodLazy",
  ZodLiteral = "ZodLiteral",
  ZodEnum = "ZodEnum",
  ZodEffects = "ZodEffects",
  ZodNativeEnum = "ZodNativeEnum",
  ZodOptional = "ZodOptional",
  ZodNullable = "ZodNullable",
  ZodDefault = "ZodDefault",
  ZodCatch = "ZodCatch",
  ZodPromise = "ZodPromise",
  ZodBranded = "ZodBranded",
  ZodPipeline = "ZodPipeline",
  ZodReadonly = "ZodReadonly"
}
type ZodFirstPartySchemaTypes = ZodString | ZodNumber | ZodNaN | ZodBigInt | ZodBoolean | ZodDate | ZodUndefined | ZodNull | ZodAny | ZodUnknown | ZodNever | ZodVoid | ZodArray<any, any> | ZodObject<any, any, any> | ZodUnion<any> | ZodDiscriminatedUnion<any, any> | ZodIntersection<any, any> | ZodTuple<any, any> | ZodRecord<any, any> | ZodMap<any> | ZodSet<any> | ZodFunction<any, any> | ZodLazy<any> | ZodLiteral<any> | ZodEnum<any> | ZodEffects<any, any, any> | ZodNativeEnum<any> | ZodOptional<any> | ZodNullable<any> | ZodDefault<any> | ZodCatch<any> | ZodPromise<any> | ZodBranded<any, any> | ZodPipeline<any, any> | ZodReadonly<any> | ZodSymbol;
declare abstract class Class {
  constructor(..._: any[]);
}
declare const instanceOfType: <T extends typeof Class>(cls: T, params?: CustomParams) => ZodType<InstanceType<T>, ZodTypeDef, InstanceType<T>>;
declare const stringType: (params?: RawCreateParams & {
  coerce?: true;
}) => ZodString;
declare const numberType: (params?: RawCreateParams & {
  coerce?: boolean;
}) => ZodNumber;
declare const nanType: (params?: RawCreateParams) => ZodNaN;
declare const bigIntType: (params?: RawCreateParams & {
  coerce?: boolean;
}) => ZodBigInt;
declare const booleanType: (params?: RawCreateParams & {
  coerce?: boolean;
}) => ZodBoolean;
declare const dateType: (params?: RawCreateParams & {
  coerce?: boolean;
}) => ZodDate;
declare const symbolType: (params?: RawCreateParams) => ZodSymbol;
declare const undefinedType: (params?: RawCreateParams) => ZodUndefined;
declare const nullType: (params?: RawCreateParams) => ZodNull;
declare const anyType: (params?: RawCreateParams) => ZodAny;
declare const unknownType: (params?: RawCreateParams) => ZodUnknown;
declare const neverType: (params?: RawCreateParams) => ZodNever;
declare const voidType: (params?: RawCreateParams) => ZodVoid;
declare const arrayType: <El extends ZodTypeAny>(schema: El, params?: RawCreateParams) => ZodArray<El>;
declare const objectType: <Shape extends ZodRawShape>(shape: Shape, params?: RawCreateParams) => ZodObject<Shape, "strip", ZodTypeAny, objectOutputType<Shape, ZodTypeAny, "strip">, objectInputType<Shape, ZodTypeAny, "strip">>;
declare const strictObjectType: <Shape extends ZodRawShape>(shape: Shape, params?: RawCreateParams) => ZodObject<Shape, "strict">;
declare const unionType: <Options extends Readonly<[ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]>>(types: Options, params?: RawCreateParams) => ZodUnion<Options>;
declare const discriminatedUnionType: typeof ZodDiscriminatedUnion.create;
declare const intersectionType: <TSchema extends ZodTypeAny, USchema extends ZodTypeAny>(left: TSchema, right: USchema, params?: RawCreateParams) => ZodIntersection<TSchema, USchema>;
declare const tupleType: <Items extends [ZodTypeAny, ...ZodTypeAny[]] | []>(schemas: Items, params?: RawCreateParams) => ZodTuple<Items, null>;
declare const recordType: typeof ZodRecord.create;
declare const mapType: <KeySchema extends ZodTypeAny = ZodTypeAny, ValueSchema extends ZodTypeAny = ZodTypeAny>(keyType: KeySchema, valueType: ValueSchema, params?: RawCreateParams) => ZodMap<KeySchema, ValueSchema>;
declare const setType: <ValueSchema extends ZodTypeAny = ZodTypeAny>(valueType: ValueSchema, params?: RawCreateParams) => ZodSet<ValueSchema>;
declare const functionType: typeof ZodFunction.create;
declare const lazyType: <Inner extends ZodTypeAny>(getter: () => Inner, params?: RawCreateParams) => ZodLazy<Inner>;
declare const literalType: <Value extends Primitive$1>(value: Value, params?: RawCreateParams) => ZodLiteral<Value>;
declare const enumType: typeof createZodEnum;
declare const nativeEnumType: <Elements extends EnumLike>(values: Elements, params?: RawCreateParams) => ZodNativeEnum<Elements>;
declare const promiseType: <Inner extends ZodTypeAny>(schema: Inner, params?: RawCreateParams) => ZodPromise<Inner>;
declare const effectsType: <I extends ZodTypeAny>(schema: I, effect: Effect<I["_output"]>, params?: RawCreateParams) => ZodEffects<I, I["_output"]>;
declare const optionalType: <Inner extends ZodTypeAny>(type: Inner, params?: RawCreateParams) => ZodOptional<Inner>;
declare const nullableType: <Inner extends ZodTypeAny>(type: Inner, params?: RawCreateParams) => ZodNullable<Inner>;
declare const preprocessType: <I extends ZodTypeAny>(preprocess: (arg: unknown, ctx: RefinementCtx) => unknown, schema: I, params?: RawCreateParams) => ZodEffects<I, I["_output"], unknown>;
declare const pipelineType: typeof ZodPipeline.create;
declare const ostring: () => ZodOptional<ZodString>;
declare const onumber: () => ZodOptional<ZodNumber>;
declare const oboolean: () => ZodOptional<ZodBoolean>;
declare const coerce: {
  string: (typeof ZodString)["create"];
  number: (typeof ZodNumber)["create"];
  boolean: (typeof ZodBoolean)["create"];
  bigint: (typeof ZodBigInt)["create"];
  date: (typeof ZodDate)["create"];
};
declare const NEVER: never;
declare namespace external_d_exports {
  export { AnyZodObject, AnyZodTuple, ArrayCardinality, ArrayKeys, AssertArray, AsyncParseReturnType, BRAND, CatchallInput, CatchallOutput, CustomErrorParams, DIRTY, DenormalizedError, EMPTY_PATH, Effect, EnumLike, EnumValues, ErrorMapCtx, FilterEnum, INVALID, Indices, InnerTypeOfFunction, InputTypeOfTuple, InputTypeOfTupleWithRest, IpVersion, IssueData, KeySchema, NEVER, OK, ObjectPair, OuterTypeOfFunction, OutputTypeOfTuple, OutputTypeOfTupleWithRest, ParseContext, ParseInput, ParseParams, ParsePath, ParsePathComponent, ParseResult, ParseReturnType, ParseStatus, PassthroughType, PreprocessEffect, Primitive$1 as Primitive, ProcessedCreateParams, RawCreateParams, RecordType, Refinement, RefinementCtx, RefinementEffect, SafeParseError, SafeParseReturnType, SafeParseSuccess, Scalars, ZodType as Schema, SomeZodObject, StringValidation, SuperRefinement, SyncParseReturnType, TransformEffect, TypeOf, UnknownKeysParam, Values, Writeable, ZodAny, ZodAnyDef, ZodArray, ZodArrayDef, ZodBigInt, ZodBigIntCheck, ZodBigIntDef, ZodBoolean, ZodBooleanDef, ZodBranded, ZodBrandedDef, ZodCatch, ZodCatchDef, ZodCustomIssue, ZodDate, ZodDateCheck, ZodDateDef, ZodDefault, ZodDefaultDef, ZodDiscriminatedUnion, ZodDiscriminatedUnionDef, ZodDiscriminatedUnionOption, ZodEffects, ZodEffectsDef, ZodEnum, ZodEnumDef, ZodError, ZodErrorMap, ZodFirstPartySchemaTypes, ZodFirstPartyTypeKind, ZodFormattedError, ZodFunction, ZodFunctionDef, ZodIntersection, ZodIntersectionDef, ZodInvalidArgumentsIssue, ZodInvalidDateIssue, ZodInvalidEnumValueIssue, ZodInvalidIntersectionTypesIssue, ZodInvalidLiteralIssue, ZodInvalidReturnTypeIssue, ZodInvalidStringIssue, ZodInvalidTypeIssue, ZodInvalidUnionDiscriminatorIssue, ZodInvalidUnionIssue, ZodIssue, ZodIssueBase, ZodIssueCode, ZodIssueOptionalMessage, ZodLazy, ZodLazyDef, ZodLiteral, ZodLiteralDef, ZodMap, ZodMapDef, ZodNaN, ZodNaNDef, ZodNativeEnum, ZodNativeEnumDef, ZodNever, ZodNeverDef, ZodNonEmptyArray, ZodNotFiniteIssue, ZodNotMultipleOfIssue, ZodNull, ZodNullDef, ZodNullable, ZodNullableDef, ZodNullableType, ZodNumber, ZodNumberCheck, ZodNumberDef, ZodObject, ZodObjectDef, ZodOptional, ZodOptionalDef, ZodOptionalType, ZodParsedType, ZodPipeline, ZodPipelineDef, ZodPromise, ZodPromiseDef, ZodRawShape, ZodReadonly, ZodReadonlyDef, ZodRecord, ZodRecordDef, ZodType as ZodSchema, ZodSet, ZodSetDef, ZodString, ZodStringCheck, ZodStringDef, ZodSymbol, ZodSymbolDef, ZodTooBigIssue, ZodTooSmallIssue, ZodEffects as ZodTransformer, ZodTuple, ZodTupleDef, ZodTupleItems, ZodType, ZodTypeAny, ZodTypeDef, ZodUndefined, ZodUndefinedDef, ZodUnion, ZodUnionDef, ZodUnionOptions, ZodUnknown, ZodUnknownDef, ZodUnrecognizedKeysIssue, ZodVoid, ZodVoidDef, addIssueToContext, anyType as any, arrayType as array, arrayOutputType, baseObjectInputType, baseObjectOutputType, bigIntType as bigint, booleanType as boolean, coerce, custom, dateType as date, datetimeRegex, errorMap as defaultErrorMap, deoptional, discriminatedUnionType as discriminatedUnion, effectsType as effect, enumType as enum, functionType as function, getErrorMap, getParsedType, TypeOf as infer, inferFlattenedErrors, inferFormattedError, input, instanceOfType as instanceof, intersectionType as intersection, isAborted, isAsync, isDirty, isValid, late, lazyType as lazy, literalType as literal, makeIssue, mapType as map, mergeTypes, nanType as nan, nativeEnumType as nativeEnum, neverType as never, noUnrecognized, nullType as null, nullableType as nullable, numberType as number, objectType as object, objectInputType, objectOutputType, objectUtil, oboolean, onumber, optionalType as optional, ostring, output, pipelineType as pipeline, preprocessType as preprocess, promiseType as promise, quotelessJson, recordType as record, setType as set, setErrorMap, strictObjectType as strictObject, stringType as string, symbolType as symbol, effectsType as transformer, tupleType as tuple, typeToFlattenedError, typecast, undefinedType as undefined, unionType as union, unknownType as unknown, util, voidType as void };
}
//#endregion
//#region src/components/CreateSecretModal/index.d.ts
declare const secretFormSchema: ZodObject<{
  name: ZodEffects<ZodString, string, string>;
  description: ZodOptional<ZodString>;
  value: ZodString;
}, "strip", ZodTypeAny, {
  name: string;
  description?: string | undefined;
  value: string;
}, {
  name: string;
  description?: string | undefined;
  value: string;
}>;
type CreateSecretFormData = TypeOf<typeof secretFormSchema>;
interface CreateSecretModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  /** Performs the create. Rejecting leaves the modal open; the caller surfaces the reason via `errorText`. */
  onCreate: (data: CreateSecretFormData) => Promise<void>;
  pending?: boolean;
  errorText?: string;
  /** Where result messages go. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}
export declare const CreateSecretModal: FC<CreateSecretModalProps>;
//#endregion
//#region src/components/DeleteConfirmationModal/index.d.ts
interface DeleteModalProps extends Pick<FormModalProps, 'open' | 'onClose'> {
  onDelete: () => boolean | Promise<boolean>;
  title: string;
  description?: string;
  confirmationText?: string;
  simpleConfirm?: boolean;
  successText?: string;
  errorText?: string;
  suppressResultToasts?: boolean;
  /** Where result messages go. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}
export declare const DeleteConfirmationModal: FC<DeleteModalProps>;
//#endregion
//#region src/components/ExpandableMessage/index.d.ts
interface Props$5 {
  /**
   * The message to display.
   */
  message?: string;
  /**
   * The message to display when there is an error. If this is provided, this takes priority over the message prop.
   */
  errorMessage?: string;
  /**
   * If true, a skeleton will be displayed.
   */
  loading?: boolean;
  /**
   * Character limit before showing "Show more" button. Defaults to 300.
   */
  characterLimit?: number;
  attributes?: {
    Anchor?: ComponentProps<typeof Anchor>;
    Text?: ComponentProps<typeof Text>;
  };
}
export declare const ExpandableMessage: FC<Props$5>;
//#endregion
//#region src/components/FileTag/index.d.ts
type FileTagStatus = 'success' | 'pending' | 'error' | 'idle';
interface FileTagProps extends Omit<TagProps, 'style'> {
  fileName?: string;
  status?: FileTagStatus;
  noFileText?: string;
  required?: boolean;
  onNoFileClick?: () => void;
  slotStart?: ReactNode;
  className?: string;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  disabled?: boolean;
}
/**
 * Tag component for displaying file names with optional status indicators.
 * Used in forms and file upload contexts. Supports required state and custom click handlers.
 */
export declare const FileTag: FC<FileTagProps>;
//#endregion
//#region ../../node_modules/.pnpm/file-selector@2.1.2/node_modules/file-selector/dist/file.d.ts
interface FileWithPath extends File {
  readonly path?: string;
  readonly handle?: FileSystemFileHandle;
  readonly relativePath?: string;
}
//#endregion
//#region ../../node_modules/.pnpm/react-dropzone@14.4.1_react@19.2.7/node_modules/react-dropzone/typings/react-dropzone.d.ts
declare enum ErrorCode {
  FileInvalidType = "file-invalid-type",
  FileTooLarge = "file-too-large",
  FileTooSmall = "file-too-small",
  TooManyFiles = "too-many-files"
}
interface FileError {
  message: string;
  code: ErrorCode | string;
}
interface FileRejection {
  file: FileWithPath;
  errors: readonly FileError[];
}
type DropzoneOptions = Pick<React$2.HTMLProps<HTMLElement>, PropTypes> & {
  accept?: Accept;
  minSize?: number;
  maxSize?: number;
  maxFiles?: number;
  preventDropOnDocument?: boolean;
  noClick?: boolean;
  noKeyboard?: boolean;
  noDrag?: boolean;
  noDragEventsBubbling?: boolean;
  disabled?: boolean;
  onDrop?: <T extends File>(acceptedFiles: T[], fileRejections: FileRejection[], event: DropEvent) => void;
  onDropAccepted?: <T extends File>(files: T[], event: DropEvent) => void;
  onDropRejected?: (fileRejections: FileRejection[], event: DropEvent) => void;
  getFilesFromEvent?: (event: DropEvent) => Promise<Array<File | DataTransferItem>>;
  onFileDialogCancel?: () => void;
  onFileDialogOpen?: () => void;
  onError?: (err: Error) => void;
  validator?: <T extends File>(file: T) => FileError | readonly FileError[] | null;
  useFsAccessApi?: boolean;
  autoFocus?: boolean;
};
type DropEvent = React$2.DragEvent<HTMLElement> | React$2.ChangeEvent<HTMLInputElement> | DragEvent | Event | Array<FileSystemFileHandle>;
type PropTypes = "multiple" | "onDragEnter" | "onDragOver" | "onDragLeave";
interface Accept {
  [key: string]: readonly string[];
}
//#endregion
//#region src/components/FileUpload/index.d.ts
type RenderFileTagFn = (file: File, disabled: boolean, onClick: MouseEventHandler<HTMLButtonElement>) => ReactNode;
interface FileUploadProps extends DropzoneOptions {
  label?: string;
  required?: boolean;
  helperText?: ReactNode;
  errorText?: ReactNode;
  disabled?: boolean;
  files?: File[];
  onRemoveFile: (file: File) => void;
  renderFileTag?: RenderFileTagFn;
  status?: FileTagStatus;
}
/**
 * A generic file upload that allows drag and drop
 */
export declare const FileUpload: FC<FileUploadProps>;
//#endregion
//#region src/components/InputErrorText/index.d.ts
/**
 * Sometimes we have to make our own input component that KUI just doesn't offer
 * but we want it to have the same styling. This component exists because KUI doesn't
 * expose the component it uses for `errorText` in its input components, but we want to
 * standardize the way we do it.
 */
export declare const InputErrorText: FC<PropsWithChildren<ComponentProps<typeof Text>>>;
//#endregion
//#region src/components/ActionsMenu/index.d.ts
interface ActionMenuItem {
  label: string;
  onSelect: () => void;
  icon?: ReactElement<{
    size?: number;
    fill?: string;
    className?: string;
  }>;
  disabled?: boolean;
  danger?: boolean;
  divider?: Omit<DropdownDividerItemEntry, 'kind'>;
}
//#endregion
//#region src/components/QuickActionsMenu/QuickActionsMenuRoot/index.d.ts
type QuickActionItem = ActionMenuItem;
interface QuickActionsMenuProps {
  actions: QuickActionItem[];
}
export declare const QuickActionsMenuRoot: FC<QuickActionsMenuProps>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils.d.ts
type PartialKeys<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
type RequiredKeys<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;
type Overwrite<T, U extends { [TKey in keyof T]?: any; }> = Omit<T, keyof U> & U;
type UnionToIntersection<T> = (T extends any ? (x: T) => any : never) extends ((x: infer R) => any) ? R : never;
type IsAny$1<T, Y, N> = 1 extends 0 & T ? Y : N;
type IsKnown<T, Y, N> = unknown extends T ? N : Y;
type ComputeRange<N extends number, Result extends Array<unknown> = []> = Result['length'] extends N ? Result : ComputeRange<N, [...Result, Result['length']]>;
type Index40 = ComputeRange<40>[number];
type IsTuple$1<T> = T extends readonly any[] & {
  length: infer Length;
} ? Length extends Index40 ? T : never : never;
type AllowedIndexes<Tuple extends ReadonlyArray<any>, Keys extends number = never> = Tuple extends readonly [] ? Keys : Tuple extends readonly [infer _, ...infer Tail] ? AllowedIndexes<Tail, Keys | Tail['length']> : Keys;
type DeepKeys<T, TDepth extends any[] = []> = TDepth['length'] extends 5 ? never : unknown extends T ? string : T extends readonly any[] & IsTuple$1<T> ? AllowedIndexes<T> | DeepKeysPrefix<T, AllowedIndexes<T>, TDepth> : T extends any[] ? DeepKeys<T[number], [...TDepth, any]> : T extends Date ? never : T extends object ? (keyof T & string) | DeepKeysPrefix<T, keyof T, TDepth> : never;
type DeepKeysPrefix<T, TPrefix, TDepth extends any[]> = TPrefix extends keyof T & (number | string) ? `${TPrefix}.${DeepKeys<T[TPrefix], [...TDepth, any]> & string}` : never;
type DeepValue<T, TProp> = T extends Record<string | number, any> ? TProp extends `${infer TBranch}.${infer TDeepProp}` ? DeepValue<T[TBranch], TDeepProp> : T[TProp & string] : never;
type NoInfer<T> = [T][T extends any ? 0 : never];
type Getter<TValue> = <TTValue = TValue>() => NoInfer<TTValue>;
declare function functionalUpdate<T>(updater: Updater<T>, input: T): T;
declare function noop(): void;
declare function makeStateUpdater<K extends keyof TableState>(key: K, instance: unknown): (updater: Updater<TableState[K]>) => void;
type AnyFunction = (...args: any) => any;
declare function isFunction<T extends AnyFunction>(d: any): d is T;
declare function isNumberArray(d: any): d is number[];
declare function flattenBy<TNode>(arr: TNode[], getChildren: (item: TNode) => TNode[]): TNode[];
declare function memo<TDeps extends readonly any[], TDepArgs, TResult>(getDeps: (depArgs?: TDepArgs) => [...TDeps], fn: (...args: NoInfer<[...TDeps]>) => TResult, opts: {
  key: any;
  debug?: () => any;
  onChange?: (result: TResult) => void;
}): (depArgs?: TDepArgs) => TResult;
declare function getMemoOptions(tableOptions: Partial<TableOptionsResolved<any>>, debugLevel: 'debugAll' | 'debugCells' | 'debugTable' | 'debugColumns' | 'debugRows' | 'debugHeaders', key: string, onChange?: (result: any) => void): {
  debug: () => boolean | undefined;
  key: string | false;
  onChange: ((result: any) => void) | undefined;
};
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/core/table.d.ts
interface CoreTableState {}
interface CoreOptions<TData extends RowData> {
  /**
   * An array of extra features that you can add to the table instance.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#_features)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  _features?: TableFeature[];
  /**
   * Set this option to override any of the `autoReset...` feature options.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#autoresetall)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  autoResetAll?: boolean;
  /**
   * The array of column defs to use for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#columns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  columns: ColumnDef<TData, any>[];
  /**
   * The data for the table to display. This array should match the type you provided to `table.setRowType<...>`. Columns can access this data via string/index or a functional accessor. When the `data` option changes reference, the table will reprocess the data.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#data)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  data: TData[];
  /**
   * Set this option to `true` to output all debugging information to the console.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#debugall)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  debugAll?: boolean;
  /**
   * Set this option to `true` to output cell debugging information to the console.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#debugcells]
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  debugCells?: boolean;
  /**
   * Set this option to `true` to output column debugging information to the console.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#debugcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  debugColumns?: boolean;
  /**
   * Set this option to `true` to output header debugging information to the console.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#debugheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  debugHeaders?: boolean;
  /**
   * Set this option to `true` to output row debugging information to the console.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#debugrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  debugRows?: boolean;
  /**
   * Set this option to `true` to output table debugging information to the console.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#debugtable)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  debugTable?: boolean;
  /**
   * Default column options to use for all column defs supplied to the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#defaultcolumn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  defaultColumn?: Partial<ColumnDef<TData, unknown>>;
  /**
   * This required option is a factory for a function that computes and returns the core row model for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getcorerowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getCoreRowModel: (table: Table<any>) => () => RowModel<any>;
  /**
   * This optional function is used to derive a unique ID for any given row. If not provided the rows index is used (nested rows join together with `.` using their grandparents' index eg. `index.index.index`). If you need to identify individual rows that are originating from any server-side operations, it's suggested you use this function to return an ID that makes sense regardless of network IO/ambiguity eg. a userId, taskId, database ID field, etc.
   * @example getRowId: row => row.userId
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getrowid)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getRowId?: (originalRow: TData, index: number, parent?: Row<TData>) => string;
  /**
   * This optional function is used to access the sub rows for any given row. If you are using nested rows, you will need to use this function to return the sub rows object (or undefined) from the row.
   * @example getSubRows: row => row.subRows
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getsubrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getSubRows?: (originalRow: TData, index: number) => undefined | TData[];
  /**
   * Use this option to optionally pass initial state to the table. This state will be used when resetting various table states either automatically by the table (eg. `options.autoResetPageIndex`) or via functions like `table.resetRowSelection()`. Most reset function allow you optionally pass a flag to reset to a blank/default state instead of the initial state.
   *
   * Table state will not be reset when this object changes, which also means that the initial state object does not need to be stable.
   *
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#initialstate)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  initialState?: InitialTableState;
  /**
   * This option is used to optionally implement the merging of table options.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#mergeoptions)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  mergeOptions?: (defaultOptions: TableOptions<TData>, options: Partial<TableOptions<TData>>) => TableOptions<TData>;
  /**
   * You can pass any object to `options.meta` and access it anywhere the `table` is available via `table.options.meta`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#meta)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  meta?: TableMeta<TData>;
  /**
   * The `onStateChange` option can be used to optionally listen to state changes within the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#onstatechange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  onStateChange: (updater: Updater<TableState>) => void;
  /**
   * Value used when the desired value is not found in the data.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#renderfallbackvalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  renderFallbackValue: any;
  /**
   * The `state` option can be used to optionally _control_ part or all of the table state. The state you pass here will merge with and overwrite the internal automatically-managed state to produce the final state for the table. You can also listen to state changes via the `onStateChange` option.
   * > Note: Any state passed in here will override both the internal state and any other `initialState` you provide.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#state)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  state: Partial<TableState>;
}
interface CoreInstance<TData extends RowData> {
  _features: readonly TableFeature[];
  _getAllFlatColumnsById: () => Record<string, Column<TData, unknown>>;
  _getColumnDefs: () => ColumnDef<TData, unknown>[];
  _getCoreRowModel?: () => RowModel<TData>;
  _getDefaultColumnDef: () => Partial<ColumnDef<TData, unknown>>;
  _getRowId: (_: TData, index: number, parent?: Row<TData>) => string;
  _queue: (cb: () => void) => void;
  /**
   * Returns all columns in the table in their normalized and nested hierarchy.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getallcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getAllColumns: () => Column<TData, unknown>[];
  /**
   * Returns all columns in the table flattened to a single level.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getallflatcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getAllFlatColumns: () => Column<TData, unknown>[];
  /**
   * Returns all leaf-node columns in the table flattened to a single level. This does not include parent columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getallleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getAllLeafColumns: () => Column<TData, unknown>[];
  /**
   * Returns a single column by its ID.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getcolumn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getColumn: (columnId: string) => Column<TData, unknown> | undefined;
  /**
   * Returns the core row model before any processing has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getcorerowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getCoreRowModel: () => RowModel<TData>;
  /**
   * Returns the row with the given ID.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getrow)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getRow: (id: string, searchAll?: boolean) => Row<TData>;
  /**
   * Returns the final model after all processing from other used features has been applied. This is the row model that is most commonly used for rendering.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getRowModel: () => RowModel<TData>;
  /**
   * Call this function to get the table's current state. It's recommended to use this function and its state, especially when managing the table state manually. It is the exact same state used internally by the table for every feature and function it provides.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#getstate)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  getState: () => TableState;
  /**
   * This is the resolved initial state of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#initialstate)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  initialState: TableState;
  /**
   * A read-only reference to the table's current options.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#options)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  options: RequiredKeys<TableOptionsResolved<TData>, 'state'>;
  /**
   * Call this function to reset the table state to the initial state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#reset)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  reset: () => void;
  /**
   * This function can be used to update the table options.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#setoptions)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  setOptions: (newOptions: Updater<TableOptionsResolved<TData>>) => void;
  /**
   * Call this function to update the table state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/table#setstate)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/tables)
   */
  setState: (updater: Updater<TableState>) => void;
}
declare function createTable<TData extends RowData>(options: TableOptionsResolved<TData>): Table<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/ColumnVisibility.d.ts
type VisibilityState = Record<string, boolean>;
interface VisibilityTableState {
  columnVisibility: VisibilityState;
}
interface VisibilityOptions {
  /**
   * Whether to enable column hiding. Defaults to `true`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#enablehiding)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  enableHiding?: boolean;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.columnVisibility` changes. This overrides the default internal state management, so you will need to persist the state change either fully or partially outside of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#oncolumnvisibilitychange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  onColumnVisibilityChange?: OnChangeFn<VisibilityState>;
}
type VisibilityDefaultOptions = Pick<VisibilityOptions, 'onColumnVisibilityChange'>;
interface VisibilityInstance<TData extends RowData> {
  /**
   * If column pinning, returns a flat array of leaf-node columns that are visible in the unpinned/center portion of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getcentervisibleleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getCenterVisibleLeafColumns: () => Column<TData, unknown>[];
  /**
   * Returns whether all columns are visible
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getisallcolumnsvisible)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getIsAllColumnsVisible: () => boolean;
  /**
   * Returns whether any columns are visible
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getissomecolumnsvisible)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getIsSomeColumnsVisible: () => boolean;
  /**
   * If column pinning, returns a flat array of leaf-node columns that are visible in the left portion of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getleftvisibleleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getLeftVisibleLeafColumns: () => Column<TData, unknown>[];
  /**
   * If column pinning, returns a flat array of leaf-node columns that are visible in the right portion of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getrightvisibleleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getRightVisibleLeafColumns: () => Column<TData, unknown>[];
  /**
   * Returns a handler for toggling the visibility of all columns, meant to be bound to a `input[type=checkbox]` element.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#gettoggleallcolumnsvisibilityhandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getToggleAllColumnsVisibilityHandler: () => (event: unknown) => void;
  /**
   * Returns a flat array of columns that are visible, including parent columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getvisibleflatcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getVisibleFlatColumns: () => Column<TData, unknown>[];
  /**
   * Returns a flat array of leaf-node columns that are visible.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getvisibleleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getVisibleLeafColumns: () => Column<TData, unknown>[];
  /**
   * Resets the column visibility state to the initial state. If `defaultState` is provided, the state will be reset to `{}`
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#resetcolumnvisibility)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  resetColumnVisibility: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.columnVisibility` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#setcolumnvisibility)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  setColumnVisibility: (updater: Updater<VisibilityState>) => void;
  /**
   * Toggles the visibility of all columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#toggleallcolumnsvisible)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  toggleAllColumnsVisible: (value?: boolean) => void;
}
interface VisibilityColumnDef {
  enableHiding?: boolean;
}
interface VisibilityRow<TData extends RowData> {
  _getAllVisibleCells: () => Cell<TData, unknown>[];
  /**
   * Returns an array of cells that account for column visibility for the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getvisiblecells)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getVisibleCells: () => Cell<TData, unknown>[];
}
interface VisibilityColumn {
  /**
   * Returns whether the column can be hidden
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getcanhide)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getCanHide: () => boolean;
  /**
   * Returns whether the column is visible
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#getisvisible)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getIsVisible: () => boolean;
  /**
   * Returns a function that can be used to toggle the column visibility. This function can be used to bind to an event handler to a checkbox.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#gettogglevisibilityhandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  getToggleVisibilityHandler: () => (event: unknown) => void;
  /**
   * Toggles the visibility of the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-visibility#togglevisibility)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-visibility)
   */
  toggleVisibility: (value?: boolean) => void;
}
declare const ColumnVisibility: TableFeature;
declare function _getVisibleLeafColumns<TData extends RowData>(table: Table<TData>, position?: ColumnPinningPosition | 'center'): Column<TData, unknown>[];
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/ColumnOrdering.d.ts
interface ColumnOrderTableState {
  columnOrder: ColumnOrderState;
}
type ColumnOrderState = string[];
interface ColumnOrderOptions {
  /**
   * If provided, this function will be called with an `updaterFn` when `state.columnOrder` changes. This overrides the default internal state management, so you will need to persist the state change either fully or partially outside of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-ordering#oncolumnorderchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-ordering)
   */
  onColumnOrderChange?: OnChangeFn<ColumnOrderState>;
}
interface ColumnOrderColumn {
  /**
   * Returns the index of the column in the order of the visible columns. Optionally pass a `position` parameter to get the index of the column in a sub-section of the table
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-ordering#getindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-ordering)
   */
  getIndex: (position?: ColumnPinningPosition | 'center') => number;
  /**
   * Returns `true` if the column is the first column in the order of the visible columns. Optionally pass a `position` parameter to check if the column is the first in a sub-section of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-ordering#getisfirstcolumn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-ordering)
   */
  getIsFirstColumn: (position?: ColumnPinningPosition | 'center') => boolean;
  /**
   * Returns `true` if the column is the last column in the order of the visible columns. Optionally pass a `position` parameter to check if the column is the last in a sub-section of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-ordering#getislastcolumn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-ordering)
   */
  getIsLastColumn: (position?: ColumnPinningPosition | 'center') => boolean;
}
interface ColumnOrderDefaultOptions {
  onColumnOrderChange: OnChangeFn<ColumnOrderState>;
}
interface ColumnOrderInstance<TData extends RowData> {
  _getOrderColumnsFn: () => (columns: Column<TData, unknown>[]) => Column<TData, unknown>[];
  /**
   * Resets the **columnOrder** state to `initialState.columnOrder`, or `true` can be passed to force a default blank state reset to `[]`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-ordering#resetcolumnorder)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-ordering)
   */
  resetColumnOrder: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.columnOrder` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-ordering#setcolumnorder)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-ordering)
   */
  setColumnOrder: (updater: Updater<ColumnOrderState>) => void;
}
declare const ColumnOrdering: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/ColumnPinning.d.ts
type ColumnPinningPosition = false | 'left' | 'right';
interface ColumnPinningState {
  left?: string[];
  right?: string[];
}
interface ColumnPinningTableState {
  columnPinning: ColumnPinningState;
}
interface ColumnPinningOptions {
  /**
   * Enables/disables column pinning for the table. Defaults to `true`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#enablecolumnpinning)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  enableColumnPinning?: boolean;
  /**
   * @deprecated Use `enableColumnPinning` or `enableRowPinning` instead.
   * Enables/disables all pinning for the table. Defaults to `true`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#enablepinning)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  enablePinning?: boolean;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.columnPinning` changes. This overrides the default internal state management, so you will also need to supply `state.columnPinning` from your own managed state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#oncolumnpinningchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/oncolumnpinningchange)
   */
  onColumnPinningChange?: OnChangeFn<ColumnPinningState>;
}
interface ColumnPinningDefaultOptions {
  onColumnPinningChange: OnChangeFn<ColumnPinningState>;
}
interface ColumnPinningColumnDef {
  /**
   * Enables/disables column pinning for this column. Defaults to `true`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#enablepinning-1)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  enablePinning?: boolean;
}
interface ColumnPinningColumn {
  /**
   * Returns whether or not the column can be pinned.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getcanpin)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getCanPin: () => boolean;
  /**
   * Returns the pinned position of the column. (`'left'`, `'right'` or `false`)
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getispinned)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getIsPinned: () => ColumnPinningPosition;
  /**
   * Returns the numeric pinned index of the column within a pinned column group.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getpinnedindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getPinnedIndex: () => number;
  /**
   * Pins a column to the `'left'` or `'right'`, or unpins the column to the center if `false` is passed.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#pin)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  pin: (position: ColumnPinningPosition) => void;
}
interface ColumnPinningRow<TData extends RowData> {
  /**
   * Returns all center pinned (unpinned) leaf cells in the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getcentervisiblecells)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getCenterVisibleCells: () => Cell<TData, unknown>[];
  /**
   * Returns all left pinned leaf cells in the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getleftvisiblecells)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getLeftVisibleCells: () => Cell<TData, unknown>[];
  /**
   * Returns all right pinned leaf cells in the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getrightvisiblecells)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getRightVisibleCells: () => Cell<TData, unknown>[];
}
interface ColumnPinningInstance<TData extends RowData> {
  /**
   * Returns all center pinned (unpinned) leaf columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getcenterleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getCenterLeafColumns: () => Column<TData, unknown>[];
  /**
   * Returns whether or not any columns are pinned. Optionally specify to only check for pinned columns in either the `left` or `right` position.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getissomecolumnspinned)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getIsSomeColumnsPinned: (position?: ColumnPinningPosition) => boolean;
  /**
   * Returns all left pinned leaf columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getleftleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getLeftLeafColumns: () => Column<TData, unknown>[];
  /**
   * Returns all right pinned leaf columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#getrightleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  getRightLeafColumns: () => Column<TData, unknown>[];
  /**
   * Resets the **columnPinning** state to `initialState.columnPinning`, or `true` can be passed to force a default blank state reset to `{ left: [], right: [], }`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#resetcolumnpinning)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  resetColumnPinning: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.columnPinning` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-pinning#setcolumnpinning)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-pinning)
   */
  setColumnPinning: (updater: Updater<ColumnPinningState>) => void;
}
declare const ColumnPinning: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/RowPinning.d.ts
type RowPinningPosition = false | 'top' | 'bottom';
interface RowPinningState {
  bottom?: string[];
  top?: string[];
}
interface RowPinningTableState {
  rowPinning: RowPinningState;
}
interface RowPinningOptions<TData extends RowData> {
  /**
   * Enables/disables row pinning for the table. Defaults to `true`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#enablerowpinning)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  enableRowPinning?: boolean | ((row: Row<TData>) => boolean);
  /**
   * When `false`, pinned rows will not be visible if they are filtered or paginated out of the table. When `true`, pinned rows will always be visible regardless of filtering or pagination. Defaults to `true`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#keeppinnedrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  keepPinnedRows?: boolean;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.rowPinning` changes. This overrides the default internal state management, so you will also need to supply `state.rowPinning` from your own managed state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#onrowpinningchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/onrowpinningchange)
   */
  onRowPinningChange?: OnChangeFn<RowPinningState>;
}
interface RowPinningDefaultOptions {
  onRowPinningChange: OnChangeFn<RowPinningState>;
}
interface RowPinningRow {
  /**
   * Returns whether or not the row can be pinned.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#getcanpin-1)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  getCanPin: () => boolean;
  /**
   * Returns the pinned position of the row. (`'top'`, `'bottom'` or `false`)
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#getispinned-1)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  getIsPinned: () => RowPinningPosition;
  /**
   * Returns the numeric pinned index of the row within a pinned row group.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#getpinnedindex-1)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  getPinnedIndex: () => number;
  /**
   * Pins a row to the `'top'` or `'bottom'`, or unpins the row to the center if `false` is passed.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#pin-1)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  pin: (position: RowPinningPosition, includeLeafRows?: boolean, includeParentRows?: boolean) => void;
}
interface RowPinningInstance<TData extends RowData> {
  _getPinnedRows: (visiblePinnedRows: Array<Row<TData>>, pinnedRowIds: Array<string> | undefined, position: 'top' | 'bottom') => Row<TData>[];
  /**
   * Returns all bottom pinned rows.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#getbottomrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  getBottomRows: () => Row<TData>[];
  /**
   * Returns all rows that are not pinned to the top or bottom.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#getcenterrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  getCenterRows: () => Row<TData>[];
  /**
   * Returns whether or not any rows are pinned. Optionally specify to only check for pinned rows in either the `top` or `bottom` position.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#getissomerowspinned)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  getIsSomeRowsPinned: (position?: RowPinningPosition) => boolean;
  /**
   * Returns all top pinned rows.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#gettoprows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  getTopRows: () => Row<TData>[];
  /**
   * Resets the **rowPinning** state to `initialState.rowPinning`, or `true` can be passed to force a default blank state reset to `{ top: [], bottom: [], }`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#resetrowpinning)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  resetRowPinning: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.rowPinning` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-pinning#setrowpinning)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-pinning)
   */
  setRowPinning: (updater: Updater<RowPinningState>) => void;
}
declare const RowPinning: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/core/headers.d.ts
interface CoreHeaderGroup<TData extends RowData> {
  depth: number;
  headers: Header<TData, unknown>[];
  id: string;
}
interface HeaderContext<TData, TValue> {
  /**
   * An instance of a column.
   */
  column: Column<TData, TValue>;
  /**
   * An instance of a header.
   */
  header: Header<TData, TValue>;
  /**
   * The table instance.
   */
  table: Table<TData>;
}
interface CoreHeader<TData extends RowData, TValue> {
  /**
   * The col-span for the header.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#colspan)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  colSpan: number;
  /**
   * The header's associated column object.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#column)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  column: Column<TData, TValue>;
  /**
   * The depth of the header, zero-indexed based.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#depth)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  depth: number;
  /**
   * Returns the rendering context (or props) for column-based components like headers, footers and filters.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#getcontext)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getContext: () => HeaderContext<TData, TValue>;
  /**
   * Returns the leaf headers hierarchically nested under this header.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#getleafheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getLeafHeaders: () => Header<TData, unknown>[];
  /**
   * The header's associated header group object.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#headergroup)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  headerGroup: HeaderGroup<TData>;
  /**
   * The unique identifier for the header.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#id)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  id: string;
  /**
   * The index for the header within the header group.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#index)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  index: number;
  /**
   * A boolean denoting if the header is a placeholder header.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#isplaceholder)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  isPlaceholder: boolean;
  /**
   * If the header is a placeholder header, this will be a unique header ID that does not conflict with any other headers across the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#placeholderid)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  placeholderId?: string;
  /**
   * The row-span for the header.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#rowspan)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  rowSpan: number;
  /**
   * The header's hierarchical sub/child headers. Will be empty if the header's associated column is a leaf-column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/header#subheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  subHeaders: Header<TData, TValue>[];
}
interface HeadersInstance<TData extends RowData> {
  /**
   * Returns all header groups for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getheadergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getHeaderGroups: () => HeaderGroup<TData>[];
  /**
   * If pinning, returns the header groups for the left pinned columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getleftheadergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getLeftHeaderGroups: () => HeaderGroup<TData>[];
  /**
   * If pinning, returns the header groups for columns that are not pinned.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getcenterheadergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getCenterHeaderGroups: () => HeaderGroup<TData>[];
  /**
   * If pinning, returns the header groups for the right pinned columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getrightheadergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getRightHeaderGroups: () => HeaderGroup<TData>[];
  /**
   * Returns the footer groups for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getfootergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getFooterGroups: () => HeaderGroup<TData>[];
  /**
   * If pinning, returns the footer groups for the left pinned columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getleftfootergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getLeftFooterGroups: () => HeaderGroup<TData>[];
  /**
   * If pinning, returns the footer groups for columns that are not pinned.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getcenterfootergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getCenterFooterGroups: () => HeaderGroup<TData>[];
  /**
   * If pinning, returns the footer groups for the right pinned columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getrightfootergroups)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getRightFooterGroups: () => HeaderGroup<TData>[];
  /**
   * Returns headers for all columns in the table, including parent headers.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getflatheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getFlatHeaders: () => Header<TData, unknown>[];
  /**
   * If pinning, returns headers for all left pinned columns in the table, including parent headers.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getleftflatheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getLeftFlatHeaders: () => Header<TData, unknown>[];
  /**
   * If pinning, returns headers for all columns that are not pinned, including parent headers.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getcenterflatheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getCenterFlatHeaders: () => Header<TData, unknown>[];
  /**
   * If pinning, returns headers for all right pinned columns in the table, including parent headers.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getrightflatheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getRightFlatHeaders: () => Header<TData, unknown>[];
  /**
   * Returns headers for all leaf columns in the table, (not including parent headers).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getleafheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getLeafHeaders: () => Header<TData, unknown>[];
  /**
   * If pinning, returns headers for all left pinned leaf columns in the table, (not including parent headers).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getleftleafheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getLeftLeafHeaders: () => Header<TData, unknown>[];
  /**
   * If pinning, returns headers for all columns that are not pinned, (not including parent headers).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getcenterleafheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getCenterLeafHeaders: () => Header<TData, unknown>[];
  /**
   * If pinning, returns headers for all right pinned leaf columns in the table, (not including parent headers).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/headers#getrightleafheaders)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/headers)
   */
  getRightLeafHeaders: () => Header<TData, unknown>[];
}
declare const Headers: TableFeature;
declare function buildHeaderGroups<TData extends RowData>(allColumns: Column<TData, unknown>[], columnsToGroup: Column<TData, unknown>[], table: Table<TData>, headerFamily?: 'center' | 'left' | 'right'): HeaderGroup<TData>[];
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/ColumnFaceting.d.ts
interface FacetedColumn<TData extends RowData> {
  _getFacetedMinMaxValues?: () => undefined | [number, number];
  _getFacetedRowModel?: () => RowModel<TData>;
  _getFacetedUniqueValues?: () => Map<any, number>;
  /**
   * A function that **computes and returns** a min/max tuple derived from `column.getFacetedRowModel`. Useful for displaying faceted result values.
   * > ⚠️ Requires that you pass a valid `getFacetedMinMaxValues` function to `options.getFacetedMinMaxValues`. A default implementation is provided via the exported `getFacetedMinMaxValues` function.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-faceting#getfacetedminmaxvalues)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-faceting)
   */
  getFacetedMinMaxValues: () => undefined | [number, number];
  /**
   * Returns the row model with all other column filters applied, excluding its own filter. Useful for displaying faceted result counts.
   * > ⚠️ Requires that you pass a valid `getFacetedRowModel` function to `options.facetedRowModel`. A default implementation is provided via the exported `getFacetedRowModel` function.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-faceting#getfacetedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-faceting)
   */
  getFacetedRowModel: () => RowModel<TData>;
  /**
   * A function that **computes and returns** a `Map` of unique values and their occurrences derived from `column.getFacetedRowModel`. Useful for displaying faceted result values.
   * > ⚠️ Requires that you pass a valid `getFacetedUniqueValues` function to `options.getFacetedUniqueValues`. A default implementation is provided via the exported `getFacetedUniqueValues` function.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-faceting#getfaceteduniquevalues)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-faceting)
   */
  getFacetedUniqueValues: () => Map<any, number>;
}
interface FacetedOptions<TData extends RowData> {
  getFacetedMinMaxValues?: (table: Table<TData>, columnId: string) => () => undefined | [number, number];
  getFacetedRowModel?: (table: Table<TData>, columnId: string) => () => RowModel<TData>;
  getFacetedUniqueValues?: (table: Table<TData>, columnId: string) => () => Map<any, number>;
}
declare const ColumnFaceting: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/GlobalFaceting.d.ts
interface GlobalFacetingInstance<TData extends RowData> {
  _getGlobalFacetedMinMaxValues?: () => undefined | [number, number];
  _getGlobalFacetedRowModel?: () => RowModel<TData>;
  _getGlobalFacetedUniqueValues?: () => Map<any, number>;
  /**
   * Currently, this function returns the built-in `includesString` filter function. In future releases, it may return more dynamic filter functions based on the nature of the data provided.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-faceting#getglobalautofilterfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-faceting)
   */
  getGlobalFacetedMinMaxValues: () => undefined | [number, number];
  /**
   * Returns the row model for the table after **global** filtering has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-faceting#getglobalfacetedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-faceting)
   */
  getGlobalFacetedRowModel: () => RowModel<TData>;
  /**
   * Returns the faceted unique values for the global filter.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-faceting#getglobalfaceteduniquevalues)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-faceting)
   */
  getGlobalFacetedUniqueValues: () => Map<any, number>;
}
declare const GlobalFaceting: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/filterFns.d.ts
declare const filterFns: {
  includesString: FilterFn<any>;
  includesStringSensitive: FilterFn<any>;
  equalsString: FilterFn<any>;
  arrIncludes: FilterFn<any>;
  arrIncludesAll: FilterFn<any>;
  arrIncludesSome: FilterFn<any>;
  equals: FilterFn<any>;
  weakEquals: FilterFn<any>;
  inNumberRange: FilterFn<any>;
};
type BuiltInFilterFn = keyof typeof filterFns;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/ColumnFiltering.d.ts
interface ColumnFiltersTableState {
  columnFilters: ColumnFiltersState;
}
type ColumnFiltersState = ColumnFilter[];
interface ColumnFilter {
  id: string;
  value: unknown;
}
interface ResolvedColumnFilter<TData extends RowData> {
  filterFn: FilterFn<TData>;
  id: string;
  resolvedValue: unknown;
}
interface FilterFn<TData extends RowData> {
  (row: Row<TData>, columnId: string, filterValue: any, addMeta: (meta: FilterMeta) => void): boolean;
  autoRemove?: ColumnFilterAutoRemoveTestFn<TData>;
  resolveFilterValue?: TransformFilterValueFn<TData>;
}
type TransformFilterValueFn<TData extends RowData> = (value: any, column?: Column<TData, unknown>) => unknown;
type ColumnFilterAutoRemoveTestFn<TData extends RowData> = (value: any, column?: Column<TData, unknown>) => boolean;
type CustomFilterFns<TData extends RowData> = Record<string, FilterFn<TData>>;
type FilterFnOption<TData extends RowData> = 'auto' | BuiltInFilterFn | keyof FilterFns | FilterFn<TData>;
interface ColumnFiltersColumnDef<TData extends RowData> {
  /**
   * Enables/disables the **column** filter for this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#enablecolumnfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  enableColumnFilter?: boolean;
  /**
   * The filter function to use with this column. Can be the name of a built-in filter function or a custom filter function.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#filterfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  filterFn?: FilterFnOption<TData>;
}
interface ColumnFiltersColumn<TData extends RowData> {
  /**
   * Returns an automatically calculated filter function for the column based off of the columns first known value.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getautofilterfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getAutoFilterFn: () => FilterFn<TData> | undefined;
  /**
   * Returns whether or not the column can be **column** filtered.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getcanfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getCanFilter: () => boolean;
  /**
   * Returns the filter function (either user-defined or automatic, depending on configuration) for the columnId specified.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getfilterfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getFilterFn: () => FilterFn<TData> | undefined;
  /**
   * Returns the index (including `-1`) of the column filter in the table's `state.columnFilters` array.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getfilterindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getFilterIndex: () => number;
  /**
   * Returns the current filter value for the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getfiltervalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getFilterValue: () => unknown;
  /**
   * Returns whether or not the column is currently filtered.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getisfiltered)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getIsFiltered: () => boolean;
  /**
   * A function that sets the current filter value for the column. You can pass it a value or an updater function for immutability-safe operations on existing values.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#setfiltervalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  setFilterValue: (updater: Updater<any>) => void;
}
interface ColumnFiltersRow<TData extends RowData> {
  /**
   * The column filters map for the row. This object tracks whether a row is passing/failing specific filters by their column ID.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#columnfilters)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  columnFilters: Record<string, boolean>;
  /**
   * The column filters meta map for the row. This object tracks any filter meta for a row as optionally provided during the filtering process.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#columnfiltersmeta)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  columnFiltersMeta: Record<string, FilterMeta>;
}
interface ColumnFiltersOptionsBase<TData extends RowData> {
  /**
   * Enables/disables **column** filtering for all columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#enablecolumnfilters)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  enableColumnFilters?: boolean;
  /**
   * Enables/disables all filtering for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#enablefilters)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  enableFilters?: boolean;
  /**
   * By default, filtering is done from parent rows down (so if a parent row is filtered out, all of its children will be filtered out as well). Setting this option to `true` will cause filtering to be done from leaf rows up (which means parent rows will be included so long as one of their child or grand-child rows is also included).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#filterfromleafrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  filterFromLeafRows?: boolean;
  /**
   * If provided, this function is called **once** per table and should return a **new function** which will calculate and return the row model for the table when it's filtered.
   * - For server-side filtering, this function is unnecessary and can be ignored since the server should already return the filtered row model.
   * - For client-side filtering, this function is required. A default implementation is provided via any table adapter's `{ getFilteredRowModel }` export.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getfilteredrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getFilteredRowModel?: (table: Table<any>) => () => RowModel<any>;
  /**
   * Disables the `getFilteredRowModel` from being used to filter data. This may be useful if your table needs to dynamically support both client-side and server-side filtering.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#manualfiltering)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  manualFiltering?: boolean;
  /**
   * By default, filtering is done for all rows (max depth of 100), no matter if they are root level parent rows or the child leaf rows of a parent row. Setting this option to `0` will cause filtering to only be applied to the root level parent rows, with all sub-rows remaining unfiltered. Similarly, setting this option to `1` will cause filtering to only be applied to child leaf rows 1 level deep, and so on.
   * This is useful for situations where you want a row's entire child hierarchy to be visible regardless of the applied filter.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#maxleafrowfilterdepth)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  maxLeafRowFilterDepth?: number;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.columnFilters` changes. This overrides the default internal state management, so you will need to persist the state change either fully or partially outside of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#oncolumnfilterschange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  onColumnFiltersChange?: OnChangeFn<ColumnFiltersState>;
}
type ResolvedFilterFns = keyof FilterFns extends never ? {
  filterFns?: Record<string, FilterFn<any>>;
} : {
  filterFns: Record<keyof FilterFns, FilterFn<any>>;
};
interface ColumnFiltersOptions<TData extends RowData> extends ColumnFiltersOptionsBase<TData>, ResolvedFilterFns {}
interface ColumnFiltersInstance<TData extends RowData> {
  _getFilteredRowModel?: () => RowModel<TData>;
  /**
   * Returns the row model for the table after **column** filtering has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getfilteredrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getFilteredRowModel: () => RowModel<TData>;
  /**
   * Returns the row model for the table before any **column** filtering has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#getprefilteredrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  getPreFilteredRowModel: () => RowModel<TData>;
  /**
   * Resets the **columnFilters** state to `initialState.columnFilters`, or `true` can be passed to force a default blank state reset to `[]`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#resetcolumnfilters)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  resetColumnFilters: (defaultState?: boolean) => void;
  /**
   * Resets the **globalFilter** state to `initialState.globalFilter`, or `true` can be passed to force a default blank state reset to `undefined`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#resetglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  resetGlobalFilter: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.columnFilters` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#setcolumnfilters)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  setColumnFilters: (updater: Updater<ColumnFiltersState>) => void;
  /**
   * Sets or updates the `state.globalFilter` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-filtering#setglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-filtering)
   */
  setGlobalFilter: (updater: Updater<any>) => void;
}
declare const ColumnFiltering: TableFeature;
declare function shouldAutoRemoveFilter<TData extends RowData>(filterFn?: FilterFn<TData>, value?: any, column?: Column<TData, unknown>): boolean;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/GlobalFiltering.d.ts
interface GlobalFilterTableState {
  globalFilter: any;
}
interface GlobalFilterColumnDef {
  /**
   * Enables/disables the **global** filter for this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#enableglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  enableGlobalFilter?: boolean;
}
interface GlobalFilterColumn {
  /**
   * Returns whether or not the column can be **globally** filtered. Set to `false` to disable a column from being scanned during global filtering.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#getcanglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  getCanGlobalFilter: () => boolean;
}
interface GlobalFilterOptions<TData extends RowData> {
  /**
   * Enables/disables **global** filtering for all columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#enableglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  enableGlobalFilter?: boolean;
  /**
   * If provided, this function will be called with the column and should return `true` or `false` to indicate whether this column should be used for global filtering.
   *
   * This is useful if the column can contain data that is not `string` or `number` (i.e. `undefined`).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#getcolumncanglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  getColumnCanGlobalFilter?: (column: Column<TData, unknown>) => boolean;
  /**
   * The filter function to use for global filtering.
   * - A `string` referencing a built-in filter function
   * - A `string` that references a custom filter functions provided via the `tableOptions.filterFns` option
   * - A custom filter function
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#globalfilterfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  globalFilterFn?: FilterFnOption<TData>;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.globalFilter` changes. This overrides the default internal state management, so you will need to persist the state change either fully or partially outside of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#onglobalfilterchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  onGlobalFilterChange?: OnChangeFn<any>;
}
interface GlobalFilterInstance<TData extends RowData> {
  /**
   * Currently, this function returns the built-in `includesString` filter function. In future releases, it may return more dynamic filter functions based on the nature of the data provided.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#getglobalautofilterfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  getGlobalAutoFilterFn: () => FilterFn<TData> | undefined;
  /**
   * Returns the filter function (either user-defined or automatic, depending on configuration) for the global filter.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#getglobalfilterfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  getGlobalFilterFn: () => FilterFn<TData> | undefined;
  /**
   * Resets the **globalFilter** state to `initialState.globalFilter`, or `true` can be passed to force a default blank state reset to `undefined`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#resetglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  resetGlobalFilter: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.globalFilter` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/global-filtering#setglobalfilter)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/global-filtering)
   */
  setGlobalFilter: (updater: Updater<any>) => void;
}
declare const GlobalFiltering: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/sortingFns.d.ts
declare const reSplitAlphaNumeric: RegExp;
declare const sortingFns: {
  alphanumeric: SortingFn<any>;
  alphanumericCaseSensitive: SortingFn<any>;
  text: SortingFn<any>;
  textCaseSensitive: SortingFn<any>;
  datetime: SortingFn<any>;
  basic: SortingFn<any>;
};
type BuiltInSortingFn = keyof typeof sortingFns;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/RowSorting.d.ts
type SortDirection = 'asc' | 'desc';
interface ColumnSort {
  desc: boolean;
  id: string;
}
type SortingState = ColumnSort[];
interface SortingTableState {
  sorting: SortingState;
}
interface SortingFn<TData extends RowData> {
  (rowA: Row<TData>, rowB: Row<TData>, columnId: string): number;
}
type CustomSortingFns<TData extends RowData> = Record<string, SortingFn<TData>>;
type SortingFnOption<TData extends RowData> = 'auto' | keyof SortingFns | BuiltInSortingFn | SortingFn<TData>;
interface SortingColumnDef<TData extends RowData> {
  /**
   * Enables/Disables multi-sorting for this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#enablemultisort)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  enableMultiSort?: boolean;
  /**
   * Enables/Disables sorting for this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#enablesorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  enableSorting?: boolean;
  /**
   * Inverts the order of the sorting for this column. This is useful for values that have an inverted best/worst scale where lower numbers are better, eg. a ranking (1st, 2nd, 3rd) or golf-like scoring
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#invertsorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  invertSorting?: boolean;
  /**
   * Set to `true` for sorting toggles on this column to start in the descending direction.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#sortdescfirst)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  sortDescFirst?: boolean;
  /**
   * The sorting function to use with this column.
   * - A `string` referencing a built-in sorting function
   * - A custom sorting function
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#sortingfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  sortingFn?: SortingFnOption<TData>;
  /**
   * The priority of undefined values when sorting this column.
   * - `false`
   *   - Undefined values will be considered tied and need to be sorted by the next column filter or original index (whichever applies)
   * - `-1`
   *   - Undefined values will be sorted with higher priority (ascending) (if ascending, undefined will appear on the beginning of the list)
   * - `1`
   *   - Undefined values will be sorted with lower priority (descending) (if ascending, undefined will appear on the end of the list)
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#sortundefined)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  sortUndefined?: false | -1 | 1 | 'first' | 'last';
}
interface SortingColumn<TData extends RowData> {
  /**
   * Removes this column from the table's sorting state
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#clearsorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  clearSorting: () => void;
  /**
   * Returns a sort direction automatically inferred based on the columns values.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getautosortdir)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getAutoSortDir: () => SortDirection;
  /**
   * Returns a sorting function automatically inferred based on the columns values.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getautosortingfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getAutoSortingFn: () => SortingFn<TData>;
  /**
   * Returns whether this column can be multi-sorted.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getcanmultisort)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getCanMultiSort: () => boolean;
  /**
   * Returns whether this column can be sorted.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getcansort)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getCanSort: () => boolean;
  /**
   * Returns the first direction that should be used when sorting this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getfirstsortdir)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getFirstSortDir: () => SortDirection;
  /**
   * Returns the current sort direction of this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getissorted)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getIsSorted: () => false | SortDirection;
  /**
   * Returns the next sorting order.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getnextsortingorder)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getNextSortingOrder: () => SortDirection | false;
  /**
   * Returns the index position of this column's sorting within the sorting state
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getsortindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getSortIndex: () => number;
  /**
   * Returns the resolved sorting function to be used for this column
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getsortingfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getSortingFn: () => SortingFn<TData>;
  /**
   * Returns a function that can be used to toggle this column's sorting state. This is useful for attaching a click handler to the column header.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#gettogglesortinghandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getToggleSortingHandler: () => undefined | ((event: unknown) => void);
  /**
   * Toggles this columns sorting state. If `desc` is provided, it will force the sort direction to that value. If `isMulti` is provided, it will additivity multi-sort the column (or toggle it if it is already sorted).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#togglesorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  toggleSorting: (desc?: boolean, isMulti?: boolean) => void;
}
interface SortingOptionsBase {
  /**
   * Enables/disables the ability to remove multi-sorts
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#enablemultiremove)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  enableMultiRemove?: boolean;
  /**
   * Enables/Disables multi-sorting for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#enablemultisort)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  enableMultiSort?: boolean;
  /**
   * Enables/Disables sorting for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#enablesorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  enableSorting?: boolean;
  /**
   * Enables/Disables the ability to remove sorting for the table.
   * - If `true` then changing sort order will circle like: 'none' -> 'desc' -> 'asc' -> 'none' -> ...
   * - If `false` then changing sort order will circle like: 'none' -> 'desc' -> 'asc' -> 'desc' -> 'asc' -> ...
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#enablesortingremoval)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  enableSortingRemoval?: boolean;
  /**
   * This function is used to retrieve the sorted row model. If using server-side sorting, this function is not required. To use client-side sorting, pass the exported `getSortedRowModel()` from your adapter to your table or implement your own.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getsortedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getSortedRowModel?: (table: Table<any>) => () => RowModel<any>;
  /**
   * Pass a custom function that will be used to determine if a multi-sort event should be triggered. It is passed the event from the sort toggle handler and should return `true` if the event should trigger a multi-sort.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#ismultisortevent)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  isMultiSortEvent?: (e: unknown) => boolean;
  /**
   * Enables manual sorting for the table. If this is `true`, you will be expected to sort your data before it is passed to the table. This is useful if you are doing server-side sorting.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#manualsorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  manualSorting?: boolean;
  /**
   * Set a maximum number of columns that can be multi-sorted.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#maxmultisortcolcount)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  maxMultiSortColCount?: number;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.sorting` changes. This overrides the default internal state management, so you will need to persist the state change either fully or partially outside of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#onsortingchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  onSortingChange?: OnChangeFn<SortingState>;
  /**
   * If `true`, all sorts will default to descending as their first toggle state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#sortdescfirst)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  sortDescFirst?: boolean;
}
type ResolvedSortingFns = keyof SortingFns extends never ? {
  sortingFns?: Record<string, SortingFn<any>>;
} : {
  sortingFns: Record<keyof SortingFns, SortingFn<any>>;
};
interface SortingOptions<TData extends RowData> extends SortingOptionsBase, ResolvedSortingFns {}
interface SortingInstance<TData extends RowData> {
  _getSortedRowModel?: () => RowModel<TData>;
  /**
   * Returns the row model for the table before any sorting has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getpresortedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getPreSortedRowModel: () => RowModel<TData>;
  /**
   * Returns the row model for the table after sorting has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#getsortedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  getSortedRowModel: () => RowModel<TData>;
  /**
   * Resets the **sorting** state to `initialState.sorting`, or `true` can be passed to force a default blank state reset to `[]`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#resetsorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  resetSorting: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.sorting` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/sorting#setsorting)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/sorting)
   */
  setSorting: (updater: Updater<SortingState>) => void;
}
declare const RowSorting: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/aggregationFns.d.ts
declare const aggregationFns: {
  sum: AggregationFn<any>;
  min: AggregationFn<any>;
  max: AggregationFn<any>;
  extent: AggregationFn<any>;
  mean: AggregationFn<any>;
  median: AggregationFn<any>;
  unique: AggregationFn<any>;
  uniqueCount: AggregationFn<any>;
  count: AggregationFn<any>;
};
type BuiltInAggregationFn = keyof typeof aggregationFns;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/ColumnGrouping.d.ts
type GroupingState = string[];
interface GroupingTableState {
  grouping: GroupingState;
}
type AggregationFn<TData extends RowData> = (columnId: string, leafRows: Row<TData>[], childRows: Row<TData>[]) => any;
type CustomAggregationFns = Record<string, AggregationFn<any>>;
type AggregationFnOption<TData extends RowData> = 'auto' | keyof AggregationFns | BuiltInAggregationFn | AggregationFn<TData>;
interface GroupingColumnDef<TData extends RowData, TValue> {
  /**
   * The cell to display each row for the column if the cell is an aggregate. If a function is passed, it will be passed a props object with the context of the cell and should return the property type for your adapter (the exact type depends on the adapter being used).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#aggregatedcell)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  aggregatedCell?: ColumnDefTemplate<ReturnType<Cell<TData, TValue>['getContext']>>;
  /**
   * The resolved aggregation function for the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#aggregationfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  aggregationFn?: AggregationFnOption<TData>;
  /**
   * Enables/disables grouping for this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#enablegrouping)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  enableGrouping?: boolean;
  /**
   * Specify a value to be used for grouping rows on this column. If this option is not specified, the value derived from `accessorKey` / `accessorFn` will be used instead.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getgroupingvalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getGroupingValue?: (row: TData) => any;
}
interface GroupingColumn<TData extends RowData> {
  /**
   * Returns the aggregation function for the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getaggregationfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getAggregationFn: () => AggregationFn<TData> | undefined;
  /**
   * Returns the automatically inferred aggregation function for the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getautoaggregationfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getAutoAggregationFn: () => AggregationFn<TData> | undefined;
  /**
   * Returns whether or not the column can be grouped.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getcangroup)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getCanGroup: () => boolean;
  /**
   * Returns the index of the column in the grouping state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getgroupedindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getGroupedIndex: () => number;
  /**
   * Returns whether or not the column is currently grouped.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getisgrouped)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getIsGrouped: () => boolean;
  /**
   * Returns a function that toggles the grouping state of the column. This is useful for passing to the `onClick` prop of a button.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#gettogglegroupinghandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getToggleGroupingHandler: () => () => void;
  /**
   * Toggles the grouping state of the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#togglegrouping)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  toggleGrouping: () => void;
}
interface GroupingRow {
  _groupingValuesCache: Record<string, any>;
  /**
   * Returns the grouping value for any row and column (including leaf rows).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getgroupingvalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getGroupingValue: (columnId: string) => unknown;
  /**
   * Returns whether or not the row is currently grouped.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getisgrouped)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getIsGrouped: () => boolean;
  /**
   * If this row is grouped, this is the id of the column that this row is grouped by.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#groupingcolumnid)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  groupingColumnId?: string;
  /**
   * If this row is grouped, this is the unique/shared value for the `groupingColumnId` for all of the rows in this group.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#groupingvalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  groupingValue?: unknown;
}
interface GroupingCell {
  /**
   * Returns whether or not the cell is currently aggregated.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getisaggregated)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getIsAggregated: () => boolean;
  /**
   * Returns whether or not the cell is currently grouped.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getisgrouped)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getIsGrouped: () => boolean;
  /**
   * Returns whether or not the cell is currently a placeholder cell.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getisplaceholder)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getIsPlaceholder: () => boolean;
}
interface ColumnDefaultOptions {
  enableGrouping: boolean;
  onGroupingChange: OnChangeFn<GroupingState>;
}
interface GroupingOptionsBase {
  /**
   * Enables/disables grouping for the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#enablegrouping)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  enableGrouping?: boolean;
  /**
   * Returns the row model after grouping has taken place, but no further.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getgroupedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getGroupedRowModel?: (table: Table<any>) => () => RowModel<any>;
  /**
   * Grouping columns are automatically reordered by default to the start of the columns list. If you would rather remove them or leave them as-is, set the appropriate mode here.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#groupedcolumnmode)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  groupedColumnMode?: false | 'reorder' | 'remove';
  /**
   * Enables manual grouping. If this option is set to `true`, the table will not automatically group rows using `getGroupedRowModel()` and instead will expect you to manually group the rows before passing them to the table. This is useful if you are doing server-side grouping and aggregation.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#manualgrouping)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  manualGrouping?: boolean;
  /**
   * If this function is provided, it will be called when the grouping state changes and you will be expected to manage the state yourself. You can pass the managed state back to the table via the `tableOptions.state.grouping` option.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#ongroupingchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  onGroupingChange?: OnChangeFn<GroupingState>;
}
type ResolvedAggregationFns = keyof AggregationFns extends never ? {
  aggregationFns?: Record<string, AggregationFn<any>>;
} : {
  aggregationFns: Record<keyof AggregationFns, AggregationFn<any>>;
};
interface GroupingOptions extends GroupingOptionsBase, ResolvedAggregationFns {}
type GroupingColumnMode = false | 'reorder' | 'remove';
interface GroupingInstance<TData extends RowData> {
  _getGroupedRowModel?: () => RowModel<TData>;
  /**
   * Returns the row model for the table after grouping has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getgroupedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getGroupedRowModel: () => RowModel<TData>;
  /**
   * Returns the row model for the table before any grouping has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#getpregroupedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  getPreGroupedRowModel: () => RowModel<TData>;
  /**
   * Resets the **grouping** state to `initialState.grouping`, or `true` can be passed to force a default blank state reset to `[]`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#resetgrouping)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  resetGrouping: (defaultState?: boolean) => void;
  /**
   * Updates the grouping state of the table via an update function or value.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/grouping#setgrouping)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/grouping)
   */
  setGrouping: (updater: Updater<GroupingState>) => void;
}
declare const ColumnGrouping: TableFeature;
declare function orderColumns<TData extends RowData>(leafColumns: Column<TData, unknown>[], grouping: string[], groupedColumnMode?: GroupingColumnMode): Column<TData, unknown>[];
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/RowExpanding.d.ts
type ExpandedStateList = Record<string, boolean>;
type ExpandedState = true | Record<string, boolean>;
interface ExpandedTableState {
  expanded: ExpandedState;
}
interface ExpandedRow {
  /**
   * Returns whether the row can be expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getcanexpand)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getCanExpand: () => boolean;
  /**
   * Returns whether all parent rows of the row are expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getisallparentsexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getIsAllParentsExpanded: () => boolean;
  /**
   * Returns whether the row is expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getisexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getIsExpanded: () => boolean;
  /**
   * Returns a function that can be used to toggle the expanded state of the row. This function can be used to bind to an event handler to a button.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#gettoggleexpandedhandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getToggleExpandedHandler: () => () => void;
  /**
   * Toggles the expanded state (or sets it if `expanded` is provided) for the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#toggleexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  toggleExpanded: (expanded?: boolean) => void;
}
interface ExpandedOptions<TData extends RowData> {
  /**
   * Enable this setting to automatically reset the expanded state of the table when expanding state changes.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#autoresetexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  autoResetExpanded?: boolean;
  /**
   * Enable/disable expanding for all rows.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#enableexpanding)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  enableExpanding?: boolean;
  /**
   * This function is responsible for returning the expanded row model. If this function is not provided, the table will not expand rows. You can use the default exported `getExpandedRowModel` function to get the expanded row model or implement your own.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getexpandedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getExpandedRowModel?: (table: Table<any>) => () => RowModel<any>;
  /**
   * If provided, allows you to override the default behavior of determining whether a row is currently expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getisrowexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getIsRowExpanded?: (row: Row<TData>) => boolean;
  /**
   * If provided, allows you to override the default behavior of determining whether a row can be expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getrowcanexpand)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getRowCanExpand?: (row: Row<TData>) => boolean;
  /**
   * Enables manual row expansion. If this is set to `true`, `getExpandedRowModel` will not be used to expand rows and you would be expected to perform the expansion in your own data model. This is useful if you are doing server-side expansion.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#manualexpanding)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  manualExpanding?: boolean;
  /**
   * This function is called when the `expanded` table state changes. If a function is provided, you will be responsible for managing this state on your own. To pass the managed state back to the table, use the `tableOptions.state.expanded` option.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#onexpandedchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  onExpandedChange?: OnChangeFn<ExpandedState>;
  /**
   * If `true` expanded rows will be paginated along with the rest of the table (which means expanded rows may span multiple pages). If `false` expanded rows will not be considered for pagination (which means expanded rows will always render on their parents page. This also means more rows will be rendered than the set page size)
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#paginateexpandedrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  paginateExpandedRows?: boolean;
}
interface ExpandedInstance<TData extends RowData> {
  _autoResetExpanded: () => void;
  _getExpandedRowModel?: () => RowModel<TData>;
  /**
   * Returns whether there are any rows that can be expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getcansomerowsexpand)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getCanSomeRowsExpand: () => boolean;
  /**
   * Returns the maximum depth of the expanded rows.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getexpandeddepth)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getExpandedDepth: () => number;
  /**
   * Returns the row model after expansion has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getexpandedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getExpandedRowModel: () => RowModel<TData>;
  /**
   * Returns whether all rows are currently expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getisallrowsexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getIsAllRowsExpanded: () => boolean;
  /**
   * Returns whether there are any rows that are currently expanded.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getissomerowsexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getIsSomeRowsExpanded: () => boolean;
  /**
   * Returns the row model before expansion has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#getpreexpandedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getPreExpandedRowModel: () => RowModel<TData>;
  /**
   * Returns a handler that can be used to toggle the expanded state of all rows. This handler is meant to be used with an `input[type=checkbox]` element.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#gettoggleallrowsexpandedhandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  getToggleAllRowsExpandedHandler: () => (event: unknown) => void;
  /**
   * Resets the expanded state of the table to the initial state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#resetexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  resetExpanded: (defaultState?: boolean) => void;
  /**
   * Updates the expanded state of the table via an update function or value.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#setexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  setExpanded: (updater: Updater<ExpandedState>) => void;
  /**
   * Toggles the expanded state for all rows.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/expanding#toggleallrowsexpanded)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/expanding)
   */
  toggleAllRowsExpanded: (expanded?: boolean) => void;
}
declare const RowExpanding: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/ColumnSizing.d.ts
interface ColumnSizingTableState {
  columnSizing: ColumnSizingState;
  columnSizingInfo: ColumnSizingInfoState;
}
type ColumnSizingState = Record<string, number>;
interface ColumnSizingInfoState {
  columnSizingStart: [string, number][];
  deltaOffset: null | number;
  deltaPercentage: null | number;
  isResizingColumn: false | string;
  startOffset: null | number;
  startSize: null | number;
}
type ColumnResizeMode = 'onChange' | 'onEnd';
type ColumnResizeDirection = 'ltr' | 'rtl';
interface ColumnSizingOptions {
  /**
   * Determines when the columnSizing state is updated. `onChange` updates the state when the user is dragging the resize handle. `onEnd` updates the state when the user releases the resize handle.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#columnresizemode)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  columnResizeMode?: ColumnResizeMode;
  /**
   * Enables or disables column resizing for the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#enablecolumnresizing)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  enableColumnResizing?: boolean;
  /**
   * Enables or disables right-to-left support for resizing the column. defaults to 'ltr'.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#columnResizeDirection)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  columnResizeDirection?: ColumnResizeDirection;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.columnSizing` changes. This overrides the default internal state management, so you will also need to supply `state.columnSizing` from your own managed state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#oncolumnsizingchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  onColumnSizingChange?: OnChangeFn<ColumnSizingState>;
  /**
   * If provided, this function will be called with an `updaterFn` when `state.columnSizingInfo` changes. This overrides the default internal state management, so you will also need to supply `state.columnSizingInfo` from your own managed state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#oncolumnsizinginfochange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  onColumnSizingInfoChange?: OnChangeFn<ColumnSizingInfoState>;
}
type ColumnSizingDefaultOptions = Pick<ColumnSizingOptions, 'columnResizeMode' | 'onColumnSizingChange' | 'onColumnSizingInfoChange' | 'columnResizeDirection'>;
interface ColumnSizingInstance {
  /**
   * If pinning, returns the total size of the center portion of the table by calculating the sum of the sizes of all unpinned/center leaf-columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getcentertotalsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getCenterTotalSize: () => number;
  /**
   * Returns the total size of the left portion of the table by calculating the sum of the sizes of all left leaf-columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getlefttotalsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getLeftTotalSize: () => number;
  /**
   * Returns the total size of the right portion of the table by calculating the sum of the sizes of all right leaf-columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getrighttotalsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getRightTotalSize: () => number;
  /**
   * Returns the total size of the table by calculating the sum of the sizes of all leaf-columns.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#gettotalsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getTotalSize: () => number;
  /**
   * Resets column sizing to its initial state. If `defaultState` is `true`, the default state for the table will be used instead of the initialValue provided to the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#resetcolumnsizing)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  resetColumnSizing: (defaultState?: boolean) => void;
  /**
   * Resets column sizing info to its initial state. If `defaultState` is `true`, the default state for the table will be used instead of the initialValue provided to the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#resetheadersizeinfo)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  resetHeaderSizeInfo: (defaultState?: boolean) => void;
  /**
   * Sets the column sizing state using an updater function or a value. This will trigger the underlying `onColumnSizingChange` function if one is passed to the table options, otherwise the state will be managed automatically by the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#setcolumnsizing)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  setColumnSizing: (updater: Updater<ColumnSizingState>) => void;
  /**
   * Sets the column sizing info state using an updater function or a value. This will trigger the underlying `onColumnSizingInfoChange` function if one is passed to the table options, otherwise the state will be managed automatically by the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#setcolumnsizinginfo)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  setColumnSizingInfo: (updater: Updater<ColumnSizingInfoState>) => void;
}
interface ColumnSizingColumnDef {
  /**
   * Enables or disables column resizing for the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#enableresizing)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  enableResizing?: boolean;
  /**
   * The maximum allowed size for the column
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#maxsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  maxSize?: number;
  /**
   * The minimum allowed size for the column
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#minsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  minSize?: number;
  /**
   * The desired size for the column
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#size)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  size?: number;
}
interface ColumnSizingColumn {
  /**
   * Returns `true` if the column can be resized.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getcanresize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getCanResize: () => boolean;
  /**
   * Returns `true` if the column is currently being resized.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getisresizing)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getIsResizing: () => boolean;
  /**
   * Returns the current size of the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getSize: () => number;
  /**
   * Returns the offset measurement along the row-axis (usually the x-axis for standard tables) for the header. This is effectively a sum of the offset measurements of all preceding (left) headers in relation to the current column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getstart)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getStart: (position?: ColumnPinningPosition | 'center') => number;
  /**
   * Returns the offset measurement along the row-axis (usually the x-axis for standard tables) for the header. This is effectively a sum of the offset measurements of all succeeding (right) headers in relation to the current column.
   */
  getAfter: (position?: ColumnPinningPosition | 'center') => number;
  /**
   * Resets the column to its initial size.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#resetsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  resetSize: () => void;
}
interface ColumnSizingHeader {
  /**
   * Returns an event handler function that can be used to resize the header. It can be used as an:
   * - `onMouseDown` handler
   * - `onTouchStart` handler
   *
   * The dragging and release events are automatically handled for you.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getresizehandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getResizeHandler: (context?: Document) => (event: unknown) => void;
  /**
   * Returns the current size of the header.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getsize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getSize: () => number;
  /**
   * Returns the offset measurement along the row-axis (usually the x-axis for standard tables) for the header. This is effectively a sum of the offset measurements of all preceding headers.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/column-sizing#getstart)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-sizing)
   */
  getStart: (position?: ColumnPinningPosition) => number;
}
declare const defaultColumnSizing: {
  size: number;
  minSize: number;
  maxSize: number;
};
declare const ColumnSizing: TableFeature;
declare function passiveEventSupported(): boolean;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/RowPagination.d.ts
interface PaginationState {
  pageIndex: number;
  pageSize: number;
}
interface PaginationTableState {
  pagination: PaginationState;
}
interface PaginationInitialTableState {
  pagination?: Partial<PaginationState>;
}
interface PaginationOptions {
  /**
   * If set to `true`, pagination will be reset to the first page when page-altering state changes eg. `data` is updated, filters change, grouping changes, etc.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#autoresetpageindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  autoResetPageIndex?: boolean;
  /**
   * Returns the row model after pagination has taken place, but no further.
   *
   * Pagination columns are automatically reordered by default to the start of the columns list. If you would rather remove them or leave them as-is, set the appropriate mode here.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getpaginationrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getPaginationRowModel?: (table: Table<any>) => () => RowModel<any>;
  /**
   * Enables manual pagination. If this option is set to `true`, the table will not automatically paginate rows using `getPaginationRowModel()` and instead will expect you to manually paginate the rows before passing them to the table. This is useful if you are doing server-side pagination and aggregation.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#manualpagination)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  manualPagination?: boolean;
  /**
   * If this function is provided, it will be called when the pagination state changes and you will be expected to manage the state yourself. You can pass the managed state back to the table via the `tableOptions.state.pagination` option.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#onpaginationchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  onPaginationChange?: OnChangeFn<PaginationState>;
  /**
   * When manually controlling pagination, you can supply a total `pageCount` value to the table if you know it (Or supply a `rowCount` and `pageCount` will be calculated). If you do not know how many pages there are, you can set this to `-1`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#pagecount)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  pageCount?: number;
  /**
   * When manually controlling pagination, you can supply a total `rowCount` value to the table if you know it. The `pageCount` can be calculated from this value and the `pageSize`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#rowcount)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  rowCount?: number;
}
interface PaginationDefaultOptions {
  onPaginationChange: OnChangeFn<PaginationState>;
}
interface PaginationInstance<TData extends RowData> {
  _autoResetPageIndex: () => void;
  _getPaginationRowModel?: () => RowModel<TData>;
  /**
   * Returns whether the table can go to the next page.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getcannextpage)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getCanNextPage: () => boolean;
  /**
   * Returns whether the table can go to the previous page.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getcanpreviouspage)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getCanPreviousPage: () => boolean;
  /**
   * Returns the page count. If manually paginating or controlling the pagination state, this will come directly from the `options.pageCount` table option, otherwise it will be calculated from the table data using the total row count and current page size.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getpagecount)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getPageCount: () => number;
  /**
   * Returns the row count. If manually paginating or controlling the pagination state, this will come directly from the `options.rowCount` table option, otherwise it will be calculated from the table data.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getrowcount)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getRowCount: () => number;
  /**
   * Returns an array of page options (zero-index-based) for the current page size.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getpageoptions)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getPageOptions: () => number[];
  /**
   * Returns the row model for the table after pagination has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getpaginationrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getPaginationRowModel: () => RowModel<TData>;
  /**
   * Returns the row model for the table before any pagination has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#getprepaginationrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  getPrePaginationRowModel: () => RowModel<TData>;
  /**
   * Increments the page index by one, if possible.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#nextpage)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  nextPage: () => void;
  /**
   * Decrements the page index by one, if possible.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#previouspage)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  previousPage: () => void;
  /**
   * Sets the page index to `0`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#firstpage)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  firstPage: () => void;
  /**
   * Sets the page index to the last page.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#lastpage)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  lastPage: () => void;
  /**
   * Resets the page index to its initial state. If `defaultState` is `true`, the page index will be reset to `0` regardless of initial state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#resetpageindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  resetPageIndex: (defaultState?: boolean) => void;
  /**
   * Resets the page size to its initial state. If `defaultState` is `true`, the page size will be reset to `10` regardless of initial state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#resetpagesize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  resetPageSize: (defaultState?: boolean) => void;
  /**
   * Resets the **pagination** state to `initialState.pagination`, or `true` can be passed to force a default blank state reset to `[]`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#resetpagination)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  resetPagination: (defaultState?: boolean) => void;
  /**
   * @deprecated The page count no longer exists in the pagination state. Just pass as a table option instead.
   */
  setPageCount: (updater: Updater<number>) => void;
  /**
   * Updates the page index using the provided function or value in the `state.pagination.pageIndex` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#setpageindex)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  setPageIndex: (updater: Updater<number>) => void;
  /**
   * Updates the page size using the provided function or value in the `state.pagination.pageSize` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#setpagesize)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  setPageSize: (updater: Updater<number>) => void;
  /**
   * Sets or updates the `state.pagination` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/pagination#setpagination)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/pagination)
   */
  setPagination: (updater: Updater<PaginationState>) => void;
}
declare const RowPagination: TableFeature;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/features/RowSelection.d.ts
type RowSelectionState = Record<string, boolean>;
interface RowSelectionTableState {
  rowSelection: RowSelectionState;
}
interface RowSelectionOptions<TData extends RowData> {
  /**
   * - Enables/disables multiple row selection for all rows in the table OR
   * - A function that given a row, returns whether to enable/disable multiple row selection for that row's children/grandchildren
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#enablemultirowselection)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  enableMultiRowSelection?: boolean | ((row: Row<TData>) => boolean);
  /**
   * - Enables/disables row selection for all rows in the table OR
   * - A function that given a row, returns whether to enable/disable row selection for that row
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#enablerowselection)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  enableRowSelection?: boolean | ((row: Row<TData>) => boolean);
  /**
   * Enables/disables automatic sub-row selection when a parent row is selected, or a function that enables/disables automatic sub-row selection for each row.
   * (Use in combination with expanding or grouping features)
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#enablesubrowselection)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  enableSubRowSelection?: boolean | ((row: Row<TData>) => boolean);
  /**
   * If provided, this function will be called with an `updaterFn` when `state.rowSelection` changes. This overrides the default internal state management, so you will need to persist the state change either fully or partially outside of the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#onrowselectionchange)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  onRowSelectionChange?: OnChangeFn<RowSelectionState>;
}
interface RowSelectionRow {
  /**
   * Returns whether or not the row can multi-select.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getcanmultiselect)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getCanMultiSelect: () => boolean;
  /**
   * Returns whether or not the row can be selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getcanselect)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getCanSelect: () => boolean;
  /**
   * Returns whether or not the row can select sub rows automatically when the parent row is selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getcanselectsubrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getCanSelectSubRows: () => boolean;
  /**
   * Returns whether or not all of the row's sub rows are selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getisallsubrowsselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getIsAllSubRowsSelected: () => boolean;
  /**
   * Returns whether or not the row is selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getisselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getIsSelected: () => boolean;
  /**
   * Returns whether or not some of the row's sub rows are selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getissomeselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getIsSomeSelected: () => boolean;
  /**
   * Returns a handler that can be used to toggle the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#gettoggleselectedhandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getToggleSelectedHandler: () => (event: unknown) => void;
  /**
   * Selects/deselects the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#toggleselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  toggleSelected: (value?: boolean, opts?: {
    selectChildren?: boolean;
  }) => void;
}
interface RowSelectionInstance<TData extends RowData> {
  /**
   * Returns the row model of all rows that are selected after filtering has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getfilteredselectedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getFilteredSelectedRowModel: () => RowModel<TData>;
  /**
   * Returns the row model of all rows that are selected after grouping has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getgroupedselectedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getGroupedSelectedRowModel: () => RowModel<TData>;
  /**
   * Returns whether or not all rows on the current page are selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getisallpagerowsselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getIsAllPageRowsSelected: () => boolean;
  /**
   * Returns whether or not all rows in the table are selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getisallrowsselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getIsAllRowsSelected: () => boolean;
  /**
   * Returns whether or not any rows on the current page are selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getissomepagerowsselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getIsSomePageRowsSelected: () => boolean;
  /**
   * Returns whether or not any rows in the table are selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getissomerowsselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getIsSomeRowsSelected: () => boolean;
  /**
   * Returns the core row model of all rows before row selection has been applied.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getpreselectedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getPreSelectedRowModel: () => RowModel<TData>;
  /**
   * Returns the row model of all rows that are selected.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#getselectedrowmodel)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getSelectedRowModel: () => RowModel<TData>;
  /**
   * Returns a handler that can be used to toggle all rows on the current page.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#gettoggleallpagerowsselectedhandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getToggleAllPageRowsSelectedHandler: () => (event: unknown) => void;
  /**
   * Returns a handler that can be used to toggle all rows in the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#gettoggleallrowsselectedhandler)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  getToggleAllRowsSelectedHandler: () => (event: unknown) => void;
  /**
   * Resets the **rowSelection** state to the `initialState.rowSelection`, or `true` can be passed to force a default blank state reset to `{}`.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#resetrowselection)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  resetRowSelection: (defaultState?: boolean) => void;
  /**
   * Sets or updates the `state.rowSelection` state.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#setrowselection)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  setRowSelection: (updater: Updater<RowSelectionState>) => void;
  /**
   * Selects/deselects all rows on the current page.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#toggleallpagerowsselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  toggleAllPageRowsSelected: (value?: boolean) => void;
  /**
   * Selects/deselects all rows in the table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/features/row-selection#toggleallrowsselected)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/row-selection)
   */
  toggleAllRowsSelected: (value?: boolean) => void;
}
declare const RowSelection: TableFeature;
declare function selectRowsFn<TData extends RowData>(table: Table<TData>, rowModel: RowModel<TData>): RowModel<TData>;
declare function isRowSelected<TData extends RowData>(row: Row<TData>, selection: Record<string, boolean>): boolean;
declare function isSubRowSelected<TData extends RowData>(row: Row<TData>, selection: Record<string, boolean>, table: Table<TData>): boolean | 'some' | 'all';
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/core/row.d.ts
interface CoreRow<TData extends RowData> {
  _getAllCellsByColumnId: () => Record<string, Cell<TData, unknown>>;
  _uniqueValuesCache: Record<string, unknown>;
  _valuesCache: Record<string, unknown>;
  /**
   * The depth of the row (if nested or grouped) relative to the root row array.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#depth)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  depth: number;
  /**
   * Returns all of the cells for the row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#getallcells)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  getAllCells: () => Cell<TData, unknown>[];
  /**
   * Returns the leaf rows for the row, not including any parent rows.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#getleafrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  getLeafRows: () => Row<TData>[];
  /**
   * Returns the parent row for the row, if it exists.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#getparentrow)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  getParentRow: () => Row<TData> | undefined;
  /**
   * Returns the parent rows for the row, all the way up to a root row.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#getparentrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  getParentRows: () => Row<TData>[];
  /**
   * Returns a unique array of values from the row for a given columnId.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#getuniquevalues)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  getUniqueValues: <TValue>(columnId: string) => TValue[];
  /**
   * Returns the value from the row for a given columnId.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#getvalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  getValue: <TValue>(columnId: string) => TValue;
  /**
   * The resolved unique identifier for the row resolved via the `options.getRowId` option. Defaults to the row's index (or relative index if it is a subRow).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#id)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  id: string;
  /**
   * The index of the row within its parent array (or the root data array).
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#index)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  index: number;
  /**
   * The original row object provided to the table. If the row is a grouped row, the original row object will be the first original in the group.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#original)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  original: TData;
  /**
   * An array of the original subRows as returned by the `options.getSubRows` option.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#originalsubrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  originalSubRows?: TData[];
  /**
   * If nested, this row's parent row id.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#parentid)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  parentId?: string;
  /**
   * Renders the value for the row in a given columnId the same as `getValue`, but will return the `renderFallbackValue` if no value is found.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#rendervalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  renderValue: <TValue>(columnId: string) => TValue;
  /**
   * An array of subRows for the row as returned and created by the `options.getSubRows` option.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/row#subrows)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/rows)
   */
  subRows: Row<TData>[];
}
declare const createRow: <TData extends unknown>(table: Table<TData>, id: string, original: TData, rowIndex: number, depth: number, subRows?: Row<TData>[], parentId?: string) => Row<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/core/cell.d.ts
interface CellContext<TData extends RowData, TValue> {
  cell: Cell<TData, TValue>;
  column: Column<TData, TValue>;
  getValue: Getter<TValue>;
  renderValue: Getter<TValue | null>;
  row: Row<TData>;
  table: Table<TData>;
}
interface CoreCell<TData extends RowData, TValue> {
  /**
   * The associated Column object for the cell.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/cell#column)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/cells)
   */
  column: Column<TData, TValue>;
  /**
   * Returns the rendering context (or props) for cell-based components like cells and aggregated cells. Use these props with your framework's `flexRender` utility to render these using the template of your choice:
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/cell#getcontext)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/cells)
   */
  getContext: () => CellContext<TData, TValue>;
  /**
   * Returns the value for the cell, accessed via the associated column's accessor key or accessor function.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/cell#getvalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/cells)
   */
  getValue: CellContext<TData, TValue>['getValue'];
  /**
   * The unique ID for the cell across the entire table.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/cell#id)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/cells)
   */
  id: string;
  /**
   * Renders the value for a cell the same as `getValue`, but will return the `renderFallbackValue` if no value is found.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/cell#rendervalue)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/cells)
   */
  renderValue: CellContext<TData, TValue>['renderValue'];
  /**
   * The associated Row object for the cell.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/cell#row)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/cells)
   */
  row: Row<TData>;
}
declare function createCell<TData extends RowData, TValue>(table: Table<TData>, row: Row<TData>, column: Column<TData, TValue>, columnId: string): Cell<TData, TValue>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/core/column.d.ts
interface CoreColumn<TData extends RowData, TValue> {
  /**
   * The resolved accessor function to use when extracting the value for the column from each row. Will only be defined if the column def has a valid accessor key or function defined.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#accessorfn)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
   */
  accessorFn?: AccessorFn<TData, TValue>;
  /**
   * The original column def used to create the column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#columndef)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
   */
  columnDef: ColumnDef<TData, TValue>;
  /**
   * The child column (if the column is a group column). Will be an empty array if the column is not a group column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#columns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
   */
  columns: Column<TData, TValue>[];
  /**
   * The depth of the column (if grouped) relative to the root column def array.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#depth)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
   */
  depth: number;
  /**
   * Returns the flattened array of this column and all child/grand-child columns for this column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#getflatcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
   */
  getFlatColumns: () => Column<TData, TValue>[];
  /**
   * Returns an array of all leaf-node columns for this column. If a column has no children, it is considered the only leaf-node column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#getleafcolumns)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
   */
  getLeafColumns: () => Column<TData, TValue>[];
  /**
     * The resolved unique identifier for the column resolved in this priority:
        - A manual `id` property from the column def
        - The accessor key from the column def
        - The header string from the column def
     * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#id)
     * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
     */
  id: string;
  /**
   * The parent column for this column. Will be undefined if this is a root column.
   * @link [API Docs](https://tanstack.com/table/v8/docs/api/core/column#parent)
   * @link [Guide](https://tanstack.com/table/v8/docs/guide/column-defs)
   */
  parent?: Column<TData, TValue>;
}
declare function createColumn<TData extends RowData, TValue>(table: Table<TData>, columnDef: ColumnDef<TData, TValue>, depth: number, parent?: Column<TData, TValue>): Column<TData, TValue>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/types.d.ts
interface TableFeature<TData extends RowData = any> {
  createCell?: (cell: Cell<TData, unknown>, column: Column<TData>, row: Row<TData>, table: Table<TData>) => void;
  createColumn?: (column: Column<TData, unknown>, table: Table<TData>) => void;
  createHeader?: (header: Header<TData, unknown>, table: Table<TData>) => void;
  createRow?: (row: Row<TData>, table: Table<TData>) => void;
  createTable?: (table: Table<TData>) => void;
  getDefaultColumnDef?: () => Partial<ColumnDef<TData, unknown>>;
  getDefaultOptions?: (table: Table<TData>) => Partial<TableOptionsResolved<TData>>;
  getInitialState?: (initialState?: InitialTableState) => Partial<TableState>;
}
interface TableMeta<TData extends RowData> {}
interface ColumnMeta<TData extends RowData, TValue> {}
interface FilterMeta {}
interface FilterFns {}
interface SortingFns {}
interface AggregationFns {}
type Updater<T> = T | ((old: T) => T);
type OnChangeFn<T> = (updaterOrValue: Updater<T>) => void;
type RowData = unknown | object | any[];
type AnyRender = (Comp: any, props: any) => any;
interface Table<TData extends RowData> extends CoreInstance<TData>, HeadersInstance<TData>, VisibilityInstance<TData>, ColumnOrderInstance<TData>, ColumnPinningInstance<TData>, RowPinningInstance<TData>, ColumnFiltersInstance<TData>, GlobalFilterInstance<TData>, GlobalFacetingInstance<TData>, SortingInstance<TData>, GroupingInstance<TData>, ColumnSizingInstance, ExpandedInstance<TData>, PaginationInstance<TData>, RowSelectionInstance<TData> {}
interface FeatureOptions<TData extends RowData> extends VisibilityOptions, ColumnOrderOptions, ColumnPinningOptions, RowPinningOptions<TData>, FacetedOptions<TData>, ColumnFiltersOptions<TData>, GlobalFilterOptions<TData>, SortingOptions<TData>, GroupingOptions, ExpandedOptions<TData>, ColumnSizingOptions, PaginationOptions, RowSelectionOptions<TData> {}
interface TableOptionsResolved<TData extends RowData> extends CoreOptions<TData>, FeatureOptions<TData> {}
interface TableOptions<TData extends RowData> extends PartialKeys<TableOptionsResolved<TData>, 'state' | 'onStateChange' | 'renderFallbackValue'> {}
interface TableState extends CoreTableState, VisibilityTableState, ColumnOrderTableState, ColumnPinningTableState, RowPinningTableState, ColumnFiltersTableState, GlobalFilterTableState, SortingTableState, ExpandedTableState, GroupingTableState, ColumnSizingTableState, PaginationTableState, RowSelectionTableState {}
interface CompleteInitialTableState extends CoreTableState, VisibilityTableState, ColumnOrderTableState, ColumnPinningTableState, RowPinningTableState, ColumnFiltersTableState, GlobalFilterTableState, SortingTableState, ExpandedTableState, GroupingTableState, ColumnSizingTableState, PaginationInitialTableState, RowSelectionTableState {}
interface InitialTableState extends Partial<CompleteInitialTableState> {}
interface Row<TData extends RowData> extends CoreRow<TData>, VisibilityRow<TData>, ColumnPinningRow<TData>, RowPinningRow, ColumnFiltersRow<TData>, GroupingRow, RowSelectionRow, ExpandedRow {}
interface RowModel<TData extends RowData> {
  rows: Row<TData>[];
  flatRows: Row<TData>[];
  rowsById: Record<string, Row<TData>>;
}
type AccessorFn<TData extends RowData, TValue = unknown> = (originalRow: TData, index: number) => TValue;
type ColumnDefTemplate<TProps extends object> = string | ((props: TProps) => any);
type StringOrTemplateHeader<TData, TValue> = string | ColumnDefTemplate<HeaderContext<TData, TValue>>;
interface StringHeaderIdentifier {
  header: string;
  id?: string;
}
interface IdIdentifier<TData extends RowData, TValue> {
  id: string;
  header?: StringOrTemplateHeader<TData, TValue>;
}
type ColumnIdentifiers<TData extends RowData, TValue> = IdIdentifier<TData, TValue> | StringHeaderIdentifier;
interface ColumnDefExtensions<TData extends RowData, TValue = unknown> extends VisibilityColumnDef, ColumnPinningColumnDef, ColumnFiltersColumnDef<TData>, GlobalFilterColumnDef, SortingColumnDef<TData>, GroupingColumnDef<TData, TValue>, ColumnSizingColumnDef {}
interface ColumnDefBase<TData extends RowData, TValue = unknown> extends ColumnDefExtensions<TData, TValue> {
  getUniqueValues?: AccessorFn<TData, unknown[]>;
  footer?: ColumnDefTemplate<HeaderContext<TData, TValue>>;
  cell?: ColumnDefTemplate<CellContext<TData, TValue>>;
  meta?: ColumnMeta<TData, TValue>;
}
interface IdentifiedColumnDef<TData extends RowData, TValue = unknown> extends ColumnDefBase<TData, TValue> {
  id?: string;
  header?: StringOrTemplateHeader<TData, TValue>;
}
type DisplayColumnDef<TData extends RowData, TValue = unknown> = ColumnDefBase<TData, TValue> & ColumnIdentifiers<TData, TValue>;
interface GroupColumnDefBase<TData extends RowData, TValue = unknown> extends ColumnDefBase<TData, TValue> {
  columns?: ColumnDef<TData, any>[];
}
type GroupColumnDef<TData extends RowData, TValue = unknown> = GroupColumnDefBase<TData, TValue> & ColumnIdentifiers<TData, TValue>;
interface AccessorFnColumnDefBase<TData extends RowData, TValue = unknown> extends ColumnDefBase<TData, TValue> {
  accessorFn: AccessorFn<TData, TValue>;
}
type AccessorFnColumnDef<TData extends RowData, TValue = unknown> = AccessorFnColumnDefBase<TData, TValue> & ColumnIdentifiers<TData, TValue>;
interface AccessorKeyColumnDefBase<TData extends RowData, TValue = unknown> extends ColumnDefBase<TData, TValue> {
  id?: string;
  accessorKey: (string & {}) | keyof TData;
}
type AccessorKeyColumnDef<TData extends RowData, TValue = unknown> = AccessorKeyColumnDefBase<TData, TValue> & Partial<ColumnIdentifiers<TData, TValue>>;
type AccessorColumnDef<TData extends RowData, TValue = unknown> = AccessorKeyColumnDef<TData, TValue> | AccessorFnColumnDef<TData, TValue>;
type ColumnDef<TData extends RowData, TValue = unknown> = DisplayColumnDef<TData, TValue> | GroupColumnDef<TData, TValue> | AccessorColumnDef<TData, TValue>;
type ColumnDefResolved<TData extends RowData, TValue = unknown> = Partial<UnionToIntersection<ColumnDef<TData, TValue>>> & {
  accessorKey?: string;
};
interface Column<TData extends RowData, TValue = unknown> extends CoreColumn<TData, TValue>, VisibilityColumn, ColumnPinningColumn, FacetedColumn<TData>, ColumnFiltersColumn<TData>, GlobalFilterColumn, SortingColumn<TData>, GroupingColumn<TData>, ColumnSizingColumn, ColumnOrderColumn {}
interface Cell<TData extends RowData, TValue> extends CoreCell<TData, TValue>, GroupingCell {}
interface Header<TData extends RowData, TValue> extends CoreHeader<TData, TValue>, ColumnSizingHeader {}
interface HeaderGroup<TData extends RowData> extends CoreHeaderGroup<TData> {}
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/columnHelper.d.ts
type ColumnHelper<TData extends RowData> = {
  accessor: <TAccessor extends AccessorFn<TData> | DeepKeys<TData>, TValue extends TAccessor extends AccessorFn<TData, infer TReturn> ? TReturn : TAccessor extends DeepKeys<TData> ? DeepValue<TData, TAccessor> : never>(accessor: TAccessor, column: TAccessor extends AccessorFn<TData> ? DisplayColumnDef<TData, TValue> : IdentifiedColumnDef<TData, TValue>) => TAccessor extends AccessorFn<TData> ? AccessorFnColumnDef<TData, TValue> : AccessorKeyColumnDef<TData, TValue>;
  display: (column: DisplayColumnDef<TData>) => DisplayColumnDef<TData, unknown>;
  group: (column: GroupColumnDef<TData>) => GroupColumnDef<TData, unknown>;
};
declare function createColumnHelper<TData extends RowData>(): ColumnHelper<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getCoreRowModel.d.ts
declare function getCoreRowModel<TData extends RowData>(): (table: Table<TData>) => () => RowModel<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getExpandedRowModel.d.ts
declare function getExpandedRowModel<TData extends RowData>(): (table: Table<TData>) => () => RowModel<TData>;
declare function expandRows<TData extends RowData>(rowModel: RowModel<TData>): {
  rows: Row<TData>[];
  flatRows: Row<TData>[];
  rowsById: Record<string, Row<TData>>;
};
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getFacetedMinMaxValues.d.ts
declare function getFacetedMinMaxValues<TData extends RowData>(): (table: Table<TData>, columnId: string) => () => undefined | [number, number];
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getFacetedRowModel.d.ts
declare function getFacetedRowModel<TData extends RowData>(): (table: Table<TData>, columnId: string) => () => RowModel<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getFacetedUniqueValues.d.ts
declare function getFacetedUniqueValues<TData extends RowData>(): (table: Table<TData>, columnId: string) => () => Map<any, number>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getFilteredRowModel.d.ts
declare function getFilteredRowModel<TData extends RowData>(): (table: Table<TData>) => () => RowModel<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getGroupedRowModel.d.ts
declare function getGroupedRowModel<TData extends RowData>(): (table: Table<TData>) => () => RowModel<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getPaginationRowModel.d.ts
declare function getPaginationRowModel<TData extends RowData>(opts?: {
  initialSync: boolean;
}): (table: Table<TData>) => () => RowModel<TData>;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+table-core@8.20.5/node_modules/@tanstack/table-core/build/lib/utils/getSortedRowModel.d.ts
declare function getSortedRowModel<TData extends RowData>(): (table: Table<TData>) => () => RowModel<TData>;
declare namespace index_d_exports$1 {
  export { AccessorColumnDef, AccessorFn, AccessorFnColumnDef, AccessorFnColumnDefBase, AccessorKeyColumnDef, AccessorKeyColumnDefBase, AggregationFn, AggregationFnOption, AggregationFns, AnyRender, BuiltInAggregationFn, BuiltInFilterFn, BuiltInSortingFn, Cell, CellContext, Column, ColumnDef, ColumnDefBase, ColumnDefResolved, ColumnDefTemplate, ColumnDefaultOptions, ColumnFaceting, ColumnFilter, ColumnFilterAutoRemoveTestFn, ColumnFiltering, ColumnFiltersColumn, ColumnFiltersColumnDef, ColumnFiltersInstance, ColumnFiltersOptions, ColumnFiltersRow, ColumnFiltersState, ColumnFiltersTableState, ColumnGrouping, ColumnHelper, ColumnMeta, ColumnOrderColumn, ColumnOrderDefaultOptions, ColumnOrderInstance, ColumnOrderOptions, ColumnOrderState, ColumnOrderTableState, ColumnOrdering, ColumnPinning, ColumnPinningColumn, ColumnPinningColumnDef, ColumnPinningDefaultOptions, ColumnPinningInstance, ColumnPinningOptions, ColumnPinningPosition, ColumnPinningRow, ColumnPinningState, ColumnPinningTableState, ColumnResizeDirection, ColumnResizeMode, ColumnSizing, ColumnSizingColumn, ColumnSizingColumnDef, ColumnSizingDefaultOptions, ColumnSizingHeader, ColumnSizingInfoState, ColumnSizingInstance, ColumnSizingOptions, ColumnSizingState, ColumnSizingTableState, ColumnSort, ColumnVisibility, CoreCell, CoreColumn, CoreHeader, CoreHeaderGroup, CoreInstance, CoreOptions, CoreRow, CoreTableState, CustomAggregationFns, CustomFilterFns, CustomSortingFns, DeepKeys, DeepValue, DisplayColumnDef, ExpandedInstance, ExpandedOptions, ExpandedRow, ExpandedState, ExpandedStateList, ExpandedTableState, FacetedColumn, FacetedOptions, FilterFn, FilterFnOption, FilterFns, FilterMeta, Getter, GlobalFaceting, GlobalFacetingInstance, GlobalFilterColumn, GlobalFilterColumnDef, GlobalFilterInstance, GlobalFilterOptions, GlobalFilterTableState, GlobalFiltering, GroupColumnDef, GroupingCell, GroupingColumn, GroupingColumnDef, GroupingColumnMode, GroupingInstance, GroupingOptions, GroupingRow, GroupingState, GroupingTableState, Header, HeaderContext, HeaderGroup, Headers, HeadersInstance, IdIdentifier, IdentifiedColumnDef, InitialTableState, IsAny$1 as IsAny, IsKnown, NoInfer, OnChangeFn, Overwrite, PaginationDefaultOptions, PaginationInitialTableState, PaginationInstance, PaginationOptions, PaginationState, PaginationTableState, PartialKeys, Renderable, RequiredKeys, ResolvedColumnFilter, Row, RowData, RowExpanding, RowModel, RowPagination, RowPinning, RowPinningDefaultOptions, RowPinningInstance, RowPinningOptions, RowPinningPosition, RowPinningRow, RowPinningState, RowPinningTableState, RowSelection, RowSelectionInstance, RowSelectionOptions, RowSelectionRow, RowSelectionState, RowSelectionTableState, RowSorting, SortDirection, SortingColumn, SortingColumnDef, SortingFn, SortingFnOption, SortingFns, SortingInstance, SortingOptions, SortingState, SortingTableState, StringHeaderIdentifier, StringOrTemplateHeader, Table, TableFeature, TableMeta, TableOptions, TableOptionsResolved, TableState, TransformFilterValueFn, UnionToIntersection, Updater, VisibilityColumn, VisibilityColumnDef, VisibilityDefaultOptions, VisibilityInstance, VisibilityOptions, VisibilityRow, VisibilityState, VisibilityTableState, _getVisibleLeafColumns, aggregationFns, buildHeaderGroups, createCell, createColumn, createColumnHelper, createRow, createTable, defaultColumnSizing, expandRows, filterFns, flattenBy, flexRender, functionalUpdate, getCoreRowModel, getExpandedRowModel, getFacetedMinMaxValues, getFacetedRowModel, getFacetedUniqueValues, getFilteredRowModel, getGroupedRowModel, getMemoOptions, getPaginationRowModel, getSortedRowModel, isFunction, isNumberArray, isRowSelected, isSubRowSelected, makeStateUpdater, memo, noop, orderColumns, passiveEventSupported, reSplitAlphaNumeric, selectRowsFn, shouldAutoRemoveFilter, sortingFns, useReactTable };
}
type Renderable<TProps> = React$2.ReactNode | React$2.ComponentType<TProps>;
/**
 * If rendering headers, cells, or footers with custom markup, use flexRender instead of `cell.getValue()` or `cell.renderValue()`.
 */
declare function flexRender<TProps extends object>(Comp: Renderable<TProps>, props: TProps): React$2.ReactNode | JSX.Element;
declare function useReactTable<TData extends RowData>(options: TableOptions<TData>): Table<TData>;
//#endregion
//#region src/components/DataView/internal/types.d.ts
/**
 * Use this type to indicate that a value is a placeholder and should be replaced with a proper type.
 * Not meant to just replace `any` - we should use `any` in appropriate places instead of TSFixMe.
 */
type TSFixMe = any;
/**
 * Use this type to indicate that a value is intentionally `any` and should not be changed.
 */
type IntentionalAny = any;
/** The status a request might be in that should be reflected in the UI. */
type QueryStatus = 'loading' | 'error' | 'success';
/**
 * When managing the state for a multi-select type filter we need to track multiple values - we do
 * so with a Record where the key is the value of the filter item and the value is a boolean
 * indicating whether the filter item is selected (always true in this case).
 */
type MultiState = Record<FilterItem['value'], true>;
/**
 * When managing the state for a filter it can be a string (the value of a text filter or
 * single-select), or a MultiState for multi-select filters. If a value hasn't been applied it can
 * be `undefined`.
 */
type FilterValue = string | MultiState | undefined;
/**
 * When defining single or multi select filters provide your options in this format. If a label is
 * not provided, the `value` will be used as the label.
 */
interface FilterItem {
  /** Whether the filter item is disabled. @defaultValue false */
  disabled?: boolean;
  /** The label to display for the filter item. If not provided, the id will be used. */
  label?: string;
  /** The value to use when filtering. */
  value: string;
}
type DataViewColumnFilterDef<TData> = {
  /** The label for the applied filter, will default to the Header if not provided. */
  label?: string;
  /** Whether the filter is loading. If true the filter will be put into a loading state. */
  loading?: boolean;
} & ({
  /** Renders a text filter. */
  type: 'text';
  /** Placeholder for the text filter. @defaultValue "Filter" */
  placeholder?: string;
} | {
  /** Renders a boolean filter. */
  type: 'boolean';
} | {
  /**
   * `single-select` renders radio buttons; `multi-select` renders checkboxes.
   */
  type: 'single-select' | 'multi-select';
  /**
   * The options to display. If not provided, the table will auto-generate options from the
   * first 500 unique values from currently filtered data.
   */
  options?: FilterItem[];
  /** Customize options based on the column. */
  optionsBuilder?: (column: Column<TData>) => FilterItem[];
} | {
  /** Provide a fully custom filter component. */
  type: 'custom';
  renderFilter: (props: {
    column: Column<TData>;
    setValue: (value: FilterValue) => void;
    value: FilterValue;
  }) => JSX$1.Element;
});
/**
 * Describes how data is managed.
 *
 * - `auto` - DataView manages sorting, filtering, and pagination.
 * - `manual` - DataView does not manage sorting, filtering, or pagination.
 * - `sort-filter-only` - DataView manages sorting and filtering, but not pagination.
 */
type WithDataViewDataMode = {
  dataMode?: 'auto';
  totalCount?: never;
} | {
  dataMode: 'manual' | 'sort-filter-only';
  totalCount: number | undefined;
};
type DataMode = NonNullable<WithDataViewDataMode['dataMode']>;
//#endregion
//#region src/components/DataView/internal/utils/filterFunctions.d.ts
/**
 * Supplemental filter functions for the DataView component. These are custom filter functions
 * that can be used in the `filterFn` or `globalFilterFn` props.
 */
declare const filterFunctions: {
  /**
   * Fuzzy filter — approximately matches the text entered to the data in the column.
   * @see https://tanstack.com/table/latest/docs/guide/fuzzy-filtering#defining-a-custom-fuzzy-filter-function
   */
  fuzzy: (row: Row<any>, columnId: string, value: any, addMeta: (meta: FilterMeta) => void) => boolean;
  /** @deprecated Use `includesString` instead. */
  singleSelect: (row: Row<any>, columnId: string, value: string | undefined) => boolean;
  /** Case insensitive multi-select filter. */
  multiSelect: (row: Row<any>, columnId: string, value: MultiState | undefined) => boolean;
  /** Case sensitive multi-select filter. */
  multiSelectSensitive: (row: Row<any>, columnId: string, value: MultiState | undefined) => boolean;
  /**
   * Numeric range filter for `{ $gte, $lte }` values. Keeps rows whose numeric value falls
   * within the inclusive bounds; either bound may be omitted for an open-ended range.
   *
   * An explicit `autoRemove` is essential: without a `filterFn`, TanStack resolves a numeric
   * column to its built-in `inNumberRange`, whose `autoRemove` reads the value as a `[min, max]`
   * tuple and treats our `{ $gte, $lte }` object as empty — silently dropping the filter on every
   * `setFilterValue`. This keeps a filter with either bound set and only clears it when both are
   * absent (matching the control, which emits `undefined` to clear).
   */
  numberRange: FilterFn<any>;
};
//#endregion
//#region src/components/DataView/internal/module-augmentation.d.ts
type DataViewFilterFns = typeof filterFunctions;
/**
 * Extends LucideProps so icons accept the `variant` prop used by the KUI design system.
 * The KUI runtime maps this to the appropriate icon styling.
 */
declare module 'lucide-react' {
  interface LucideProps {
    variant?: 'fill' | 'line';
  }
}
/**
 * Extends the ColumnMeta interface from react-table to include the filter definition and other
 * data-view-specific column meta fields.
 * https://tanstack.com/table/latest/docs/api/core/column-def#meta
 */
declare module '@tanstack/react-table' {
  interface ColumnMeta<TData extends RowData, TValue> {
    /** For internal use. Indicates if the column is a prebuilt column. */
    _isPrebuiltColumn?: boolean;
    /**
     * For internal use. Indicates if the initial column definition provides a size. We need this
     * because Tanstack Table will set a default size of 150px if no size is provided. If a size is
     * not provided we want to auto size the column - not use the default 150px size.
     */
    _isSizeInitialized?: boolean;
    /**
     * Controls the column cell alignment for both the header and the cell.
     *
     * To control the header separately, use `headerAlignment`.
     */
    alignment?: 'left' | 'center' | 'right';
    /**
     * Controls the header cell alignment.
     * @defaultValue "left"
     */
    headerAlignment?: 'left' | 'center' | 'right';
    /** The filter definition for the column. If provided, column filtering will be enabled. */
    filter?: DataViewColumnFilterDef<TData>;
    /**
     * By default table cells will render an OS tooltip of its children. Use this to disable the
     * tooltips, or customize the tooltip that gets generated.
     * @deprecated Replace with `title`.
     */
    tooltip?: false | ((cell: Cell<TData, TValue>) => string | undefined);
    /**
     * By default table cells will render an OS tooltip of its children. Use this to disable the
     * tooltips, or customize the tooltip that gets generated.
     */
    title?: false | ((cell: Cell<TData, TValue>) => string | undefined);
  }
}
//#endregion
//#region src/components/DataView/internal/AppliedFilters.d.ts
/**
 * Displays the list of applied filters in the DataView. Should be rendered below the toolbar.
 */
declare function AppliedFilters(props: Partial<FlexProps>): JSX$1.Element | null;
/** A tag that represents a filter applied to a table. */
declare function ColumnFilterTag({ column, value }: {
  column: Column<IntentionalAny> | undefined;
  value: FilterValue | string[];
}): JSX$1.Element | null;
//#endregion
//#region src/components/DataView/internal/BulkActions.d.ts
interface DataViewBulkActionsProps<TData> {
  /**
   * Function that returns a React node to render. Called with the selected rows and the table.
   *
   * @example
   * ```tsx
   * <DataView.BulkActions>
   *   {({ selectedRows, table }) => (...render your buttons here...)}
   * </DataView.BulkActions>
   * ```
   */
  children: (props: {
    selectedRows: Row<TData>[];
    table: Table<TData>;
  }) => ReactNode;
  onCancel?: () => void;
}
/**
 * Renders bulk actions. Must be rendered inside `DataView.Toolbar` so it can position itself
 * over the toolbar correctly.
 */
declare function BulkActions<TData>({ children, onCancel }: DataViewBulkActionsProps<TData>): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/cells/CopyCell.d.ts
type CellComponent<TData, TValue> = (ctx: CellContext<TData, TValue>) => TSFixMe;
/**
 * Plugin cell that renders a copy button. Can wrap other cells, or be used directly as a cell.
 *
 * @example
 * ```tsx
 * columnHelper.accessor('id', { cell: CopyCell, header: 'ID' });
 * columnHelper.accessor('lastModified', { cell: CopyCell(DateCell), header: 'Last Modified' });
 * ```
 */
declare function CopyCell<TData, TValue>(cellOrContext: CellComponent<TData, TValue> | CellContext<TData, TValue>): TSFixMe;
//#endregion
//#region src/components/DataView/internal/useDataViewState.d.ts
/**
 * A hook to be used with the DataView component to manage state and access table state.
 *
 * @example
 * ```tsx
 * const tableState = DataView.useDataViewState();
 * return <DataView.Root state={tableState} ... />;
 * ```
 */
declare function useDataViewState(defaultState?: {
  columnFilters?: ColumnFiltersState;
  columnOrder?: ColumnOrderState;
  columnPinning?: ColumnPinningState;
  columnVisibility?: Record<string, boolean>;
  displayMode?: 'card' | 'table';
  pagination?: Partial<PaginationState> & {
    paginationOptions?: number[];
  };
  expansion?: ExpandedState;
  rowHighlight?: string;
  searchBar?: string;
  sorting?: SortingState;
  tab?: string;
}): {
  columnFiltering: {
    state: ColumnFiltersState;
    set: import("react").Dispatch<import("react").SetStateAction<ColumnFiltersState>>;
  };
  columnOrder: {
    state: ColumnOrderState;
    set: import("react").Dispatch<import("react").SetStateAction<ColumnOrderState>>;
  };
  columnPinning: {
    state: ColumnPinningState;
    set: import("react").Dispatch<import("react").SetStateAction<ColumnPinningState>>;
  };
  columnVisibility: {
    state: Record<string, boolean>;
    set: import("react").Dispatch<import("react").SetStateAction<Record<string, boolean>>>;
  };
  displayMode: {
    state: string;
    set: import("react").Dispatch<import("react").SetStateAction<string>>;
  };
  expansion: {
    state: ExpandedState;
    set: import("react").Dispatch<import("react").SetStateAction<ExpandedState>>;
  };
  pagination: {
    isPageIndexDirty: boolean;
    isPageSizeDirty: boolean;
    /** The page being currently rendered, 0 indexed. */
    state: PaginationState & {
      paginationOptions?: number[];
    };
    /** State setter for pagination state. */
    set: import("react").Dispatch<import("react").SetStateAction<PaginationState & {
      paginationOptions?: number[];
    }>>;
    /** Reset pagination to the first page. */
    goToFirstPage: () => void;
  };
  rowHighlight: {
    state: string | number | undefined;
    set: import("react").Dispatch<import("react").SetStateAction<string | number | undefined>>;
  };
  rowSelection: {
    state: RowSelectionState;
    set: import("react").Dispatch<import("react").SetStateAction<RowSelectionState>>;
  };
  searchBar: {
    state: string;
    set: import("react").Dispatch<import("react").SetStateAction<string>>;
  };
  sorting: {
    state: SortingState;
    set: import("react").Dispatch<import("react").SetStateAction<SortingState>>;
  };
  tab: {
    state: string | undefined;
    set: import("react").Dispatch<import("react").SetStateAction<string | undefined>>;
  };
};
type DataViewState = ReturnType<typeof useDataViewState>;
//#endregion
//#region src/components/DataView/internal/context.d.ts
interface DataViewContextStore {
  autoCellTooltips: boolean;
  data: unknown[];
  dataMode: DataMode;
  isDataViewEmptyState: boolean;
  isDataViewErrorState: boolean;
  isDataViewLoadingState: boolean;
  renderCustomRowExpansion: ((data: {
    row: Row<IntentionalAny>;
  }) => JSX$1.Element) | undefined;
  requestStatus: QueryStatus | undefined;
  totalCount: number | undefined;
  state: ReturnType<typeof useDataViewState>;
  table: Table<IntentionalAny>;
}
declare const DataViewContext: import("react").Provider<DataViewContextStore>;
/**
 * A context to store data view state that is useful in data view sub-components.
 */
declare function useInnerDataViewContext(): DataViewContextStore;
//#endregion
//#region src/components/DataView/internal/StatusResult.d.ts
type TableStatusState = Pick<ReturnType<typeof useInnerDataViewContext>, 'isDataViewEmptyState' | 'isDataViewErrorState' | 'table' | 'state'>;
interface StatusResultProps extends Partial<StatusMessageProps> {
  /** Render a custom error state. */
  renderErrorState?: (tableState: TableStatusState) => JSX$1.Element;
  /** Render a custom empty state. */
  renderEmptyState?: (tableState: TableStatusState & {
    hasFiltersApplied: boolean;
    hasSearchApplied: boolean;
  }) => JSX$1.Element | null;
}
declare function StatusResult({ renderErrorState, renderEmptyState, ...props }: StatusResultProps): JSX$1.Element | null;
//#endregion
//#region src/components/DataView/internal/CustomContent.d.ts
interface CustomContentProps<TData> extends Pick<StatusResultProps, 'renderErrorState' | 'renderEmptyState'> {
  children: (args: {
    rows: Row<TData>[];
  }) => JSX$1.Element;
  /** Custom loading state. By default a spinner is rendered. */
  renderLoadingState?: () => JSX$1.Element;
  /** Associated displayMode value. If provided, only render when the displayMode matches. */
  value?: string;
}
/**
 * Manages loading, empty, and error states and provides rows for rendering custom content
 * such as cards.
 */
declare function CustomContent<TData>({ children, renderEmptyState, renderErrorState, renderLoadingState, value }: CustomContentProps<TData>): JSX$1.Element | null;
//#endregion
//#region src/components/DataView/internal/cells/DateCell.d.ts
interface DateCellProps {
  format?: string;
}
/**
 * Plugin cell that renders a date. Can pass a format to customize the date format.
 *
 * @example
 * ```tsx
 * columnHelper.accessor('lastModified', { cell: DateCell, header: 'Last Modified' });
 * columnHelper.accessor('created', { cell: DateCell({ format: 'yyyy-MM-dd' }), header: 'Created' });
 * ```
 */
declare function DateCell<TData, TValue>(cellOrContext: DateCellProps | CellContext<TData, TValue>): TSFixMe;
//#endregion
//#region src/constants/index.d.ts
declare const DEFAULT_DEBOUNCE_MS = 500;
export declare const JOB_POLLING_INTERVAL_MS = 5000;
//#endregion
//#region src/components/DataView/internal/DebouncedTextInput.d.ts
interface DebouncedTextInputProps extends TextInputProps {
  /**
   * The number of milliseconds to debounce the onValueChange handler.
   * @defaultValue DEFAULT_DEBOUNCE_MS
   */
  debounce?: typeof DEFAULT_DEBOUNCE_MS | number;
}
/** A TextInput component with a debounced onValueChange handler. */
declare function DebouncedTextInput({ debounce, onValueChange, value: initialValue, ...props }: DebouncedTextInputProps): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/cells/DefaultCell.d.ts
/**
 * The default cell rendered for columns. Renders the value or a dash if the value is undefined.
 */
declare function DefaultCell<TData, TValue>(cellContext: CellContext<TData, TValue>): (NoInfer<TValue> & {}) | '-';
//#endregion
//#region src/components/DataView/internal/DownloadButton.d.ts
interface DownloadButtonFileContent {
  content: string;
  mimeType: string;
}
interface PrepareDownloadContext<TData> {
  table: Table<TData>;
  rows: Row<TData>[];
  columns: Column<TData>[];
}
interface DownloadButtonProps extends Omit<ButtonProps, 'onClick'> {
  /** The filename to use when downloading the file. @defaultValue "data.csv" */
  filename?: string;
  /**
   * Function to prepare the file content for download. Receives the table, rows, and columns
   * and should return the content string and MIME type.
   * @defaultValue Returns CSV content with "text/csv;charset=utf-8;" MIME type.
   */
  prepareDownload?: (context: PrepareDownloadContext<IntentionalAny>) => DownloadButtonFileContent;
  /** Optional callback invoked after preparing download with the rows and generated content. */
  onClick?: (data: {
    rows: Row<IntentionalAny>[];
    content: string;
  }) => void;
}
/**
 * A button that downloads table data as a file. By default exports all visible rows as CSV.
 * Use `prepareDownload` to customize output format or row selection.
 */
declare function DownloadButton({ filename, prepareDownload, onClick, children, ...props }: DownloadButtonProps): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/EditColumnsMenu.d.ts
interface EditColumnsMenuProps extends Pick<DropdownRootProps, 'size'>, Omit<DropdownTriggerProps, 'children' | 'size'> {
  /** Content to render inside the dropdown trigger, before the columns. */
  children?: ReactNode;
  /** Additional content rendered inside the menu, after the columns. */
  slotContent?: ReactNode;
}
/**
 * When clicked this button renders a menu to control column settings.
 */
declare function EditColumnsMenu({ children, size, slotContent, ...props }: EditColumnsMenuProps): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/FilterMenu.d.ts
/**
 * FilterMenu component used to display the filter menu in the data view. Should be rendered
 * inside a `DataView.Toolbar` component.
 */
declare function FilterMenu({ children, closeOnFilterChange, disabled, size, ...props }: {
  children?: ReactNode;
  /** Whether the menu should be closed when a filter is applied. @defaultValue false */
  closeOnFilterChange?: boolean;
  size?: 'small' | 'medium' | 'large';
} & Omit<DropdownTriggerProps, 'children'>): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/cells/LoadingCell.d.ts
/** A plugin cell that renders a loading state. Used for the loading state of a table. */
declare function LoadingCell(): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/Pagination.d.ts
interface DataViewPaginationProps extends Omit<PaginationRootProps, 'totalItems'> {
  /**
   * Treat `DataView.Pagination` as only a state provider and pass your own pagination
   * components as `children` for maximum control.
   */
  children?: ReactNode;
  /** Whether to show the first and last page buttons. @defaultValue true */
  showFirstAndLastButtons?: boolean;
  /** Whether to show the go to page input. @defaultValue true */
  showGoToPage?: boolean;
  /** Whether to show the items per page select. @defaultValue true */
  showItemsPerPage?: boolean;
  /** Whether to show pagination while in an empty state. @defaultValue false */
  showWhileEmpty?: boolean;
  /** Whether to show pagination while in an error state. @defaultValue false */
  showWhileError?: boolean;
  /** Whether to show pagination while in a loading state. @defaultValue false */
  showWhileLoading?: boolean;
  /** Whether to show pagination while items < page size. @defaultValue false */
  showWhileLessThanPageSize?: boolean;
}
/**
 * A pagination component for the DataView. Should be rendered below the table content.
 */
declare function Pagination({ className, children, showFirstAndLastButtons, showGoToPage, showItemsPerPage, showWhileError, showWhileEmpty, showWhileLoading, showWhileLessThanPageSize, ...props }: DataViewPaginationProps): JSX$1.Element | null;
/** Displays the total count of items in the DataView. */
declare function PaginationStatus({ className, text, ...props }: {
  text?: {
    singular: string;
    plural: string;
  };
} & ComponentPropsWithoutRef<'span'>): JSX$1.Element | null;
//#endregion
//#region src/components/DataView/internal/RefreshButton.d.ts
/**
 * A refresh button for the table. If the table is fetching data, a spinner is displayed
 * instead. Intended for use within `DataView.Toolbar`.
 */
declare function RefreshButton({ disabled, isFetching, ...props }: {
  /** Whether the table is fetching data. */
  isFetching: boolean;
  /** Function to call when the button is clicked. Should trigger a refetch of the table data. */
  onClick: () => void;
} & Omit<ButtonProps, 'onClick'>): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/cells/RowActionsCell.d.ts
interface RowActionsCellProps<TData> extends Omit<DropdownProps, 'items'> {
  /** The cell context from tanstack/react-table. */
  ctx: CellContext<TData, unknown>;
  /**
   * A function that returns the items to render in the dropdown. For conditional row actions,
   * return a falsey value to render nothing.
   */
  rowActions?: (row: TData, ctx: CellContext<TData, unknown>) => DropdownProps['items'] | false | null | undefined;
}
/**
 * A cell component for the row actions column. Renders the row actions dropdown.
 * To customize the trigger, pass a custom `children` prop.
 */
declare function RowActionsCell<TData>({ children, ctx, rowActions, ...props }: RowActionsCellProps<TData>): JSX$1.Element | null;
//#endregion
//#region src/components/DataView/internal/hooks/useMakeColumns.d.ts
declare const PREBUILT_COLUMN_IDS: string[];
declare function rowActionsColumn<TData>(options?: Partial<DisplayColumnDef<TData>> & {
  cellProps?: Partial<DropdownProps>;
  rowActions?: RowActionsCellProps<TData>['rowActions'];
}): DisplayColumnDef<TData>;
declare function rowExpansionColumn<TData>(options?: Partial<DisplayColumnDef<TData>> & {
  headerProps?: Partial<ButtonProps>;
  props?: Partial<ButtonProps>;
}): DisplayColumnDef<TData>;
declare function rowSelectionColumn<TData>(options?: Partial<DisplayColumnDef<TData>> & {
  headerProps?: Partial<CheckboxProps>;
  props?: Partial<CheckboxProps>;
}): DisplayColumnDef<TData>;
/**
 * A set of helper pre-built columns that can be used to quickly create a table.
 * Includes row actions, row expansion, and row selection columns.
 */
declare const PREBUILT_COLUMNS: {
  rowActionsColumn: typeof rowActionsColumn;
  rowExpansionColumn: typeof rowExpansionColumn;
  rowSelectionColumn: typeof rowSelectionColumn;
};
type PrebuiltColumns = typeof PREBUILT_COLUMNS;
type PrebuiltColumnIds = typeof PREBUILT_COLUMN_IDS;
type MakeColumns<TData> = (columnHelper: ColumnHelper<TData>, prebuiltColumns: typeof PREBUILT_COLUMNS) => ColumnDef<TData, IntentionalAny>[];
//#endregion
//#region src/components/DataView/internal/Root.d.ts
type DataViewCommonProps<TData> = {
  /**
   * If true, tooltips via the "title" attribute will be automatically added to cells. This will
   * help when cells are truncated.
   * @defaultValue true
   */
  autoCellTooltips?: boolean;
  /** The data to display in the table. */
  data: TData[] | undefined;
  /** Builds the column definitions. */
  makeColumns: MakeColumns<TData>;
  /** Pass options to the underlying tanstack/react-table hook. */
  reactTableOptions?: Partial<TableOptions<TData>>;
  /**
   * Custom row expansion component, as an alternative to subRows. If both are provided, both
   * will be rendered.
   */
  renderCustomRowExpansion?: (data: {
    row: Row<TData>;
  }) => JSX$1.Element;
  /** If provided, the table will display a loading or error state. */
  requestStatus?: QueryStatus;
  /** The returned object from `useDataViewState`. Manages the state of the DataView. */
  state: DataViewState;
  /** Number of rows to display while loading. */
  loadingRows?: number;
};
type DataViewProps<TData> = DataViewCommonProps<TData> & WithDataViewDataMode;
/**
 * The root component for the DataView. All DataView components should be rendered within this
 * component, which provides the context.
 */
declare function Root<TData>({ autoCellTooltips, className, children, data: _data, dataMode, makeColumns, reactTableOptions, renderCustomRowExpansion, requestStatus, state, totalCount, loadingRows, ...props }: DataViewProps<TData> & FlexProps): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/cells/RowExpansionCell.d.ts
interface RowExpansionCellProps<TData> extends ButtonProps {
  /** The cell context from tanstack/react-table. */
  ctx: CellContext<TData, unknown>;
}
/**
 * A cell component for the row expansion column. Renders the row expansion button.
 * Only renders if the row is expandable.
 */
declare function RowExpansionCell<TData>({ children, ctx, ...props }: {
  ctx: CellContext<TData, unknown>;
} & ButtonProps): false | JSX$1.Element;
interface RowExpansionHeaderCellProps<TData> extends ButtonProps {
  /** The header context from tanstack/react-table. */
  ctx: HeaderContext<TData, unknown>;
}
/**
 * A header cell component for the row expansion column.
 * Only renders if the table has any expandable rows.
 */
declare function RowExpansionHeaderCell<TData>({ children, ctx, ...props }: RowExpansionHeaderCellProps<TData>): false | JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/cells/RowSelectionCell.d.ts
/**
 * A cell component for the row selection column. Renders the row selection checkbox.
 * Disabled if the row is not selectable.
 */
declare function RowSelectionCell<TData>({ ctx, ...props }: {
  ctx: CellContext<TData, unknown>;
} & CheckboxProps): JSX$1.Element;
/** A header cell component for the row selection column. Renders the global selection checkbox. */
declare function RowSelectionHeaderCell<TData>({ ctx, ...props }: {
  ctx: HeaderContext<TData, unknown>;
} & CheckboxProps): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/SearchBar.d.ts
interface DataViewSearchBarProps extends Omit<Partial<DebouncedTextInputProps>, 'placeholder'> {
  /**
   * Provide a placeholder that informs the user what they're able to search for.
   * @defaultValue "Search table"
   */
  placeholder?: string;
}
/**
 * A search bar for the data view. Should be rendered in `DataView.Toolbar`. Controls the
 * DataView's "global filter" — i.e. cross-column search.
 */
declare function SearchBar(props: DataViewSearchBarProps): JSX$1.Element;
//#endregion
//#region ../../node_modules/.pnpm/@tanstack+virtual-core@3.13.6/node_modules/@tanstack/virtual-core/dist/esm/index.d.ts
type ScrollDirection = 'forward' | 'backward';
type ScrollAlignment = 'start' | 'center' | 'end' | 'auto';
type ScrollBehavior = 'auto' | 'smooth';
interface ScrollToOptions {
  align?: ScrollAlignment;
  behavior?: ScrollBehavior;
}
type ScrollToOffsetOptions = ScrollToOptions;
type ScrollToIndexOptions = ScrollToOptions;
interface Range {
  startIndex: number;
  endIndex: number;
  overscan: number;
  count: number;
}
type Key$1 = number | string | bigint;
interface VirtualItem {
  key: Key$1;
  index: number;
  start: number;
  end: number;
  size: number;
  lane: number;
}
interface Rect {
  width: number;
  height: number;
}
type ObserveOffsetCallBack = (offset: number, isScrolling: boolean) => void;
interface VirtualizerOptions<TScrollElement extends Element | Window, TItemElement extends Element> {
  count: number;
  getScrollElement: () => TScrollElement | null;
  estimateSize: (index: number) => number;
  scrollToFn: (offset: number, options: {
    adjustments?: number;
    behavior?: ScrollBehavior;
  }, instance: Virtualizer<TScrollElement, TItemElement>) => void;
  observeElementRect: (instance: Virtualizer<TScrollElement, TItemElement>, cb: (rect: Rect) => void) => void | (() => void);
  observeElementOffset: (instance: Virtualizer<TScrollElement, TItemElement>, cb: ObserveOffsetCallBack) => void | (() => void);
  debug?: boolean;
  initialRect?: Rect;
  onChange?: (instance: Virtualizer<TScrollElement, TItemElement>, sync: boolean) => void;
  measureElement?: (element: TItemElement, entry: ResizeObserverEntry | undefined, instance: Virtualizer<TScrollElement, TItemElement>) => number;
  overscan?: number;
  horizontal?: boolean;
  paddingStart?: number;
  paddingEnd?: number;
  scrollPaddingStart?: number;
  scrollPaddingEnd?: number;
  initialOffset?: number | (() => number);
  getItemKey?: (index: number) => Key$1;
  rangeExtractor?: (range: Range) => Array<number>;
  scrollMargin?: number;
  gap?: number;
  indexAttribute?: string;
  initialMeasurementsCache?: Array<VirtualItem>;
  lanes?: number;
  isScrollingResetDelay?: number;
  useScrollendEvent?: boolean;
  enabled?: boolean;
  isRtl?: boolean;
  useAnimationFrameWithResizeObserver?: boolean;
}
declare class Virtualizer<TScrollElement extends Element | Window, TItemElement extends Element> {
  private unsubs;
  options: Required<VirtualizerOptions<TScrollElement, TItemElement>>;
  scrollElement: TScrollElement | null;
  targetWindow: (Window & typeof globalThis) | null;
  isScrolling: boolean;
  private scrollToIndexTimeoutId;
  measurementsCache: Array<VirtualItem>;
  private itemSizeCache;
  private pendingMeasuredCacheIndexes;
  scrollRect: Rect | null;
  scrollOffset: number | null;
  scrollDirection: ScrollDirection | null;
  private scrollAdjustments;
  shouldAdjustScrollPositionOnItemSizeChange: undefined | ((item: VirtualItem, delta: number, instance: Virtualizer<TScrollElement, TItemElement>) => boolean);
  elementsCache: Map<Key$1, TItemElement>;
  private observer;
  range: {
    startIndex: number;
    endIndex: number;
  } | null;
  constructor(opts: VirtualizerOptions<TScrollElement, TItemElement>);
  setOptions: (opts: VirtualizerOptions<TScrollElement, TItemElement>) => void;
  private notify;
  private maybeNotify;
  private cleanup;
  _didMount: () => () => void;
  _willUpdate: () => void;
  private getSize;
  private getScrollOffset;
  private getFurthestMeasurement;
  private getMeasurementOptions;
  private getMeasurements;
  calculateRange: {
    (): {
      startIndex: number;
      endIndex: number;
    } | null;
    updateDeps(newDeps: [VirtualItem[], number, number, number]): void;
  };
  getVirtualIndexes: {
    (): number[];
    updateDeps(newDeps: [(range: Range) => number[], number, number, number | null, number | null]): void;
  };
  indexFromElement: (node: TItemElement) => number;
  private _measureElement;
  resizeItem: (index: number, size: number) => void;
  measureElement: (node: TItemElement | null | undefined) => void;
  getVirtualItems: {
    (): VirtualItem[];
    updateDeps(newDeps: [number[], VirtualItem[]]): void;
  };
  getVirtualItemForOffset: (offset: number) => VirtualItem | undefined;
  getOffsetForAlignment: (toOffset: number, align: ScrollAlignment, itemSize?: number) => number;
  getOffsetForIndex: (index: number, align?: ScrollAlignment) => readonly [number, "auto"] | readonly [number, "start" | "center" | "end"] | undefined;
  private isDynamicMode;
  private cancelScrollToIndex;
  scrollToOffset: (toOffset: number, { align, behavior }?: ScrollToOffsetOptions) => void;
  scrollToIndex: (index: number, { align: initialAlign, behavior }?: ScrollToIndexOptions) => void;
  scrollBy: (delta: number, { behavior }?: ScrollToOffsetOptions) => void;
  getTotalSize: () => number;
  private _scrollToOffset;
  measure: () => void;
}
//#endregion
//#region src/components/DataView/internal/TableContent.d.ts
interface TableContentProps extends TableRootProps, Pick<StatusResultProps, 'renderEmptyState' | 'renderErrorState'> {
  /** If provided, limits rendering to the number of rows passed. */
  rowLimit?: number;
  /** Slot for the status result component (empty/error states). */
  slotStatusResult?: ReactNode;
  /** If true, the table header will be sticky. @defaultValue false */
  stickyTableHeader?: boolean;
  virtualizer?: Virtualizer<HTMLTableElement, HTMLElement>;
  /** When true, column headers become draggable for reordering. Pinned columns are excluded. */
  enableColumnReordering?: boolean;
}
/**
 * The DataView Table content component. For virtualized tables, use `VirtualizedTableContent`.
 */
declare const TableContent: import("react").ForwardRefExoticComponent<Omit<TableContentProps, "ref"> & import("react").RefAttributes<HTMLTableElement>>;
//#endregion
//#region src/components/DataView/internal/Tabs.d.ts
interface DeprecatedDataViewTab {
  children?: never;
  /**
   * An optional count value to append to the label in parenthesis.
   * @example { label: 'All', count: 10, value: 'all' } => 'All (10)'
   */
  count?: number;
  /**
   * The label to display in the tab. Must be unique.
   * @deprecated Use `children` instead. Will be removed in the next major version.
   */
  label: string;
  /** The value to return when the tab is selected. */
  value: string;
}
/** A DataView Tab. */
type DataViewTab = DeprecatedDataViewTab | (Pick<TabItem, 'children' | 'disabled' | 'attributes' | 'value'> & {
  /** Optional count to append to the label in parenthesis. */
  count?: number;
  /** @deprecated Use `children` instead. */
  label?: never;
});
/**
 * Renders tabs for the DataView component, used to switch between different views of the data.
 * Should be rendered directly above the table/content.
 */
declare function Tabs({ tabs, ...props }: {
  tabs: DataViewTab[];
} & Partial<TabsProps>): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/Toolbar.d.ts
/**
 * Renders the main toolbar in the DataView. Contains the search bar, view toggle, refresh
 * button, and other persistent controls.
 */
declare function Toolbar({ className, children, slotBulkActions, ...props }: TableToolbarProps): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/ViewToggleButton.d.ts
interface ViewToggleItem {
  /** The label to display for the view. Rendered when there is enough space. */
  children: string;
  /** The icon to display for the view. Always rendered. */
  slotLeft: ReactElement;
  /** The value to set when the view is toggled to. */
  value: string;
}
declare const DEFAULT_VIEW_ITEMS: ViewToggleItem[];
/**
 * Toggles between display modes for the DataView. Cycles through items in the order they're
 * defined. Defaults to "table" and "card" views.
 */
declare function ViewToggleButton({ items, ...props }: Omit<ButtonProps, 'children'> & {
  items?: ViewToggleItem[];
}): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/VirtualizedTableContent.d.ts
type VirtualizedTableContentProps = Omit<TableContentProps, 'width' | 'virtualizer'> & ({
  /** @deprecated Replace with `maxHeight` */
  height: CSSProperties['height'];
  maxHeight?: never;
} | {
  height?: never;
  /** Virtualized content requires a height to limit it. */
  maxHeight: CSSProperties['maxHeight'];
}) & {
  /**
   * Number of additional hidden rows to add to the virtualized list. The current implementation
   * does not account for subrows; if you use subrows, set this to the max subrows expected.
   * @defaultValue 0
   */
  countOffset?: number;
  /**
   * Max number of rows to render to measure approximate column widths.
   * @defaultValue 5
   */
  measurementModeRows?: number;
  /**
   * Number of items to render outside of the visible window.
   * @defaultValue 5
   */
  overscan?: number;
  /** Height of each row in pixels. @defaultValue 56 */
  rowHeight?: number;
  /** Options to pass to the virtualizer. */
  virtualizeOptions?: VirtualizerOptions<HTMLTableElement, HTMLElement>;
  /** @deprecated Use `style` and CSS instead. */
  width?: CSSProperties['width'];
};
/**
 * The DataView Virtualized Table content component. For non-virtualized tables, use `TableContent`.
 */
declare function VirtualizedTableContent({ className, countOffset, height, maxHeight, measurementModeRows, overscan, rowHeight, stickyTableHeader, style, virtualizeOptions, width, ...props }: VirtualizedTableContentProps): JSX$1.Element;
//#endregion
//#region src/components/DataView/internal/utils/formatters.d.ts
/**
 * Returns a function that formats a date string using the given date-fns format pattern.
 * @see https://date-fns.org/docs/format
 */
declare function makeDateFormatter(format: string): (date: string) => string;
declare const formatSimplifiedDateTime: (date: string) => string;
declare function formatMultiCapitalize(str: string): string;
//#endregion
//#region src/components/DataView/internal/utils/cell-utils.d.ts
/**
 * Renders the cell for the given row & columnId. Useful when reusing cell rendering logic
 * manually — for example, when rendering a cell in a Card format instead of a table.
 */
declare function renderCell(row: Row<IntentionalAny>, columnId: string): ReactNode;
/**
 * Convenience function to create a cell renderer function for use in the `cell` prop of a
 * column definition. Handles typing of the cell context.
 */
declare function makeCell(cellRenderFunction: <TData, TValue>(cellContext: CellContext<TData, TValue>) => ReactNode): <TData, TValue>(cellContext: CellContext<TData, TValue>) => ReactNode;
/** Resolves whether the given input is a CellContext. */
declare function isCellContext<TData, TValue, TOther>(cellOrContext: TOther | CellContext<TData, TValue>): cellOrContext is CellContext<TData, TValue>;
/** Returns the title to be used for a cell. */
declare function getCellTitle(cell: Cell<unknown, unknown>): string | undefined;
//#endregion
//#region src/components/DataView/internal/cells/TriggerCell.d.ts
type MakeTriggerCellProps<TData> = {
  onSelect: (data: TData, ctx: CellContext<TData, unknown>) => void;
  link?: never;
} | {
  onSelect?: never;
  link: (props: {
    children: ReactNode;
    data: TData;
  }) => ReactNode;
};
/**
 * Generates a cell for triggering an action or navigating to a page. Accepts either an
 * `onSelect` function or a `link` component.
 */
declare function makeTriggerCell<TData, TValue = unknown>({ link: LinkComponent, onSelect }: MakeTriggerCellProps<TData>): (cellContext: CellContext<TData, TValue>) => JSX$1.Element;
declare namespace index_d_exports {
  export { AppliedFilters, BulkActions, ColumnFilterTag, CopyCell, CustomContent, CustomContentProps, DEFAULT_VIEW_ITEMS, DataMode, DataViewBulkActionsProps, DataViewCommonProps, DataViewContext, DataViewFilterFns, DataViewPaginationProps, DataViewProps, DataViewSearchBarProps, DataViewState, DataViewTab, DateCell, DebouncedTextInput, DebouncedTextInputProps, DefaultCell, DownloadButton, DownloadButtonFileContent, DownloadButtonProps, EditColumnsMenu, FilterItem, FilterMenu, FilterValue, IntentionalAny, LoadingCell, MakeColumns, Pagination, PaginationStatus, PrebuiltColumnIds, PrebuiltColumns, PrepareDownloadContext, QueryStatus, RefreshButton, Root, RowActionsCell, RowActionsCellProps, RowExpansionCell, RowExpansionCellProps, RowExpansionHeaderCell, RowExpansionHeaderCellProps, RowSelectionCell, RowSelectionHeaderCell, SearchBar, StatusResult, StatusResultProps, TSFixMe, TableContent, TableContentProps, Tabs, index_d_exports$1 as TanstackTable, Toolbar, ViewToggleButton, VirtualizedTableContent, VirtualizedTableContentProps, WithDataViewDataMode, filterFunctions, formatMultiCapitalize, formatSimplifiedDateTime, getCellTitle, isCellContext, makeCell, makeDateFormatter, makeTriggerCell, renderCell, useDataViewState, useInnerDataViewContext };
}
//#endregion
//#region src/components/DataView/StudioDataViewToolbar.d.ts
interface StudioDataViewToolbarProps<DataType = unknown> {
  searchField?: string;
  showFilters: boolean;
  onToggleFilters: () => void;
  renderBulkActions?: (props: {
    selectedRows: DataType[];
    table: Table<DataType>;
  }) => ReactNode;
  searchBarProps?: ComponentProps<typeof SearchBar>;
  /** Rendered before the search bar, at the toolbar's leading edge. */
  slotStart?: ReactNode;
  /**
   * Additional content rendered inside the toolbar row, after the filter toggle button.
   * Use this to inject view-specific controls such as a sort dropdown.
   */
  slotEnd?: ReactNode;
}
/**
 * Toolbar for StudioDataView-style views. Renders the search bar, filter toggle button,
 * and applied filter tags. Must be used inside a `DataView.Root` context.
 *
 * Exported so it can be used in non-table views (e.g. card grids) that wrap their content
 * in a headless `DataView.Root` for filter state management.
 */
export declare function StudioDataViewToolbar<DataType = unknown>({ searchField, showFilters, onToggleFilters, renderBulkActions, searchBarProps, slotStart, slotEnd }: StudioDataViewToolbarProps<DataType>): import("react/jsx-runtime").JSX.Element | null;
//#endregion
//#region src/components/DataView/StudioDataView.d.ts
export declare const ROW_SELECTION_COLUMN_SIZE = 50;
export declare const ROW_ACTIONS_COLUMN_SIZE = 50;
/**
 * Implements a common DataView component for Studio.
 * Opinionated prop choices for DataView components for consistency across Studio.
 */
interface Props$4<DataType> {
  dataViewState: DataViewState;
  makeColumns: ComponentProps<typeof Root<DataType>>['makeColumns'];
  /**
   * Called when a row is clicked or activated via keyboard (Enter/Space).
   * When provided, rows become clickable with cursor-pointer styling and keyboard-navigable.
   * Clicks on interactive child elements (buttons, links, inputs, etc.) are excluded automatically.
   * Add `data-no-row-click` to any element to opt it out of row-click delegation.
   */
  onRowClick?: (row: DataType, index: number) => void;
  /**
   * Maximum number of text lines to show in each data cell before truncating
   * with an ellipsis. Prebuilt columns (row-selection, row-actions) are not affected.
   */
  maxTwoLines?: boolean;
  /**
   * The data field to search against. When provided, the search bar is rendered
   * in the toolbar. When omitted, no search bar is shown.
   */
  searchField?: string;
  /**
   * Render function for bulk actions shown in the toolbar when rows are selected.
   * Receives the selected rows' original data (unwrapped from TanStack Row objects).
   * When omitted, no bulk actions toolbar is rendered.
   */
  renderBulkActions?: (props: {
    selectedRows: DataType[];
    table: Table<DataType>;
  }) => ReactNode;
  /**
   * Custom content rendered in place of DataView.TableContent + DataView.Pagination.
   * Use this to render a card grid or other non-table layout inside DataView.Root.
   * When omitted, the default table with pagination is rendered.
   */
  children?: ReactNode;
  /** Rendered at the leading end of the toolbar row, before the search bar. */
  toolbarSlotStart?: ReactNode;
  /**
   * Rendered at the trailing end of the toolbar row (after the search bar and filter toggle).
   * Useful for controls like a sort dropdown that belong visually in the toolbar.
   */
  toolbarSlotEnd?: ReactNode;
  /**
   * Ref attached to the scrollable container that wraps custom `children`.
   * Pass this to a virtualizer so it can observe the scroll position.
   */
  scrollContainerRef?: RefObject<HTMLDivElement | null>;
  attributes?: {
    DataViewRoot?: Omit<ComponentProps<typeof Root<DataType>>, 'dataMode' | 'state' | 'makeColumns'>;
    DataViewTableContent?: ComponentProps<typeof TableContent>;
    DataViewPagination?: ComponentProps<typeof Pagination>;
    DataViewSearchBar?: ComponentProps<typeof SearchBar>;
  };
}
export declare const StudioDataView: <DataType>({ attributes, children, makeColumns, dataViewState, onRowClick, maxTwoLines, renderBulkActions, scrollContainerRef, searchField, toolbarSlotStart, toolbarSlotEnd }: Props$4<DataType>) => import("react/jsx-runtime").JSX.Element;
//#endregion
//#region src/utils/logs.d.ts
/**
 * Incremental state of a multi-page log fetch. `total` is the server-side line count
 * for the whole job, known once the first page resolves.
 */
interface LogLoadProgress {
  /** Lines fetched so far across every page walked. Not reduced by retention trimming. */
  loaded: number;
  total: number;
}
//#endregion
//#region src/components/LogViewer/index.d.ts
interface LogViewerProps {
  logs: PlatformJobLog[];
  isLoading?: boolean;
  downloadFilename?: string;
  rows?: number;
  fillHeight?: boolean;
  emptyMessage?: string;
  /** Progress of the initial multi-page fetch, shown beneath the loading spinner.
   *  Ignored once logs are on screen. Callers that fetch in one shot can omit it. */
  loadProgress?: LogLoadProgress | null;
  /** Where the copy confirmation goes. Defaults to the surrounding ToastProvider; plugins pass `host.notifications.notify`. */
  onNotify?: NotifyFn;
}
export declare const LogViewer: FC<LogViewerProps>;
//#endregion
//#region src/components/RadioCard/index.d.ts
interface RadioCardProps extends Omit<ComponentProps<typeof RadioGroupItem>, 'children'> {
  /** The radio value (forwarded to RadioGroupInput) */
  value: string;
  /** Primary label (e.g. "Radio Label") */
  label: ReactNode;
  /** Optional secondary description text */
  description?: ReactNode;
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
    Card?: Partial<ComponentProps<typeof Card$1>>;
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
export declare const RadioCard: FC<RadioCardProps>;
//#endregion
//#region src/components/RelativeTime/util.d.ts
type DateStringISO = string;
//#endregion
//#region src/components/RelativeTime/index.d.ts
type RelativeTimeProps = {
  datetime: DateStringISO;
  align?: 'center' | 'left' | 'right';
  underline?: boolean;
  abbreviated?: boolean;
  /**
   * When true (default), the timestamp is focusable (`tabIndex={0}`) so keyboard users can open
   * the absolute-time tooltip on focus. Set to false in modal/side-panel content where focus
   * traps would move focus here on open and incorrectly show the tooltip.
   */
  focusableForTooltip?: boolean;
};
export declare const useRelativeTimeSince: (datetime: DateStringISO, abbreviated?: boolean) => string;
export declare const RelativeTime: ({ datetime, align, underline, abbreviated, focusableForTooltip }: RelativeTimeProps) => import("react/jsx-runtime").JSX.Element;
//#endregion
//#region src/components/StatusBadge/badgeStatus.d.ts
interface StatusConfigEntry {
  label: string;
  color: Exclude<BadgeProps['color'], null>;
  icon?: ComponentType<SVGProps<SVGSVGElement>>;
}
type BadgeStatus<T = PlatformJobStatus> = Exclude<T, undefined> | 'error' | 'active' | 'in_progress' | 'unavailable' | 'ready' | 'unknown' | 'default' | 'starting' | 'running' | 'failed' | 'deleting' | 'deleted' | 'lost';
//#endregion
//#region src/components/StatusBadge/index.d.ts
interface StatusBadgeProps<T = string> {
  status: BadgeStatus<T> | string | undefined;
  statusConfig?: Record<string, StatusConfigEntry>;
  fallback?: StatusConfigEntry;
  label?: string;
}
export declare const StatusBadge: <T extends string = string>({ status, statusConfig, fallback, label: labelOverride }: StatusBadgeProps<T>) => import("react/jsx-runtime").JSX.Element;
//#endregion
//#region src/components/TableEmptyState/index.d.ts
interface Props$3 {
  className?: string;
  header: string;
  emptyMessage?: React.ReactNode;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
}
/**
 * A generic component that renders a standard empty state in-place of a table when there is no data.
 */
export declare const TableEmptyState: FC<Props$3>;
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/constants.d.ts
declare const VALIDATION_MODE: {
  readonly onBlur: "onBlur";
  readonly onChange: "onChange";
  readonly onSubmit: "onSubmit";
  readonly onTouched: "onTouched";
  readonly all: "all";
};
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/utils/createSubject.d.ts
type Observer<T> = {
  next: (value: T) => void;
};
type Subscription = {
  unsubscribe: Noop;
};
type Subject<T> = {
  readonly observers: Observer<T>[];
  subscribe: (value: Observer<T>) => Subscription;
  unsubscribe: Noop;
} & Observer<T>;
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/events.d.ts
type EventType = 'focus' | 'blur' | 'change' | 'changeText' | 'valueChange' | 'contentSizeChange' | 'endEditing' | 'keyPress' | 'submitEditing' | 'layout' | 'selectionChange' | 'longPress' | 'press' | 'pressIn' | 'pressOut' | 'momentumScrollBegin' | 'momentumScrollEnd' | 'scroll' | 'scrollBeginDrag' | 'scrollEndDrag' | 'load' | 'error' | 'progress' | 'custom';
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/path/common.d.ts
/**
 * Type to query whether an array type T is a tuple type.
 * @typeParam T - type which may be an array or tuple
 * @example
 * ```
 * IsTuple<[number]> = true
 * IsTuple<number[]> = false
 * ```
 */
type IsTuple<T extends ReadonlyArray<any>> = number extends T['length'] ? false : true;
/**
 * Type which can be used to index an array or tuple type.
 */
type ArrayKey = number;
/**
 * Type which given a tuple type returns its own keys, i.e. only its indices.
 * @typeParam T - tuple type
 * @example
 * ```
 * TupleKeys<[number, string]> = '0' | '1'
 * ```
 */
type TupleKeys<T extends ReadonlyArray<any>> = Exclude<keyof T, keyof any[]>;
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/path/eager.d.ts
/**
 * Helper function to break apart T1 and check if any are equal to T2
 *
 * See {@link IsEqual}
 */
type AnyIsEqual<T1, T2> = T1 extends T2 ? IsEqual<T1, T2> extends true ? true : never : never;
/**
 * Helper type for recursively constructing paths through a type.
 * This actually constructs the strings and recurses into nested
 * object types.
 *
 * See {@link Path}
 */
type PathImpl<K extends string | number, V, TraversedTypes> = V extends Primitive | BrowserNativeObject ? `${K}` : true extends AnyIsEqual<TraversedTypes, V> ? `${K}` : `${K}` | `${K}.${PathInternal<V, TraversedTypes | V>}`;
/**
 * Helper type for recursively constructing paths through a type.
 * This obscures the internal type param TraversedTypes from exported contract.
 *
 * See {@link Path}
 */
type PathInternal<T, TraversedTypes = T> = T extends ReadonlyArray<infer V> ? IsTuple<T> extends true ? { [K in TupleKeys<T>]-?: PathImpl<K & string, T[K], TraversedTypes>; }[TupleKeys<T>] : PathImpl<ArrayKey, V, TraversedTypes> : { [K in keyof T]-?: PathImpl<K & string, T[K], TraversedTypes>; }[keyof T];
/**
 * Type which eagerly collects all paths through a type
 * @typeParam T - type which should be introspected
 * @example
 * ```
 * Path<{foo: {bar: string}}> = 'foo' | 'foo.bar'
 * ```
 */
type Path<T> = T extends any ? PathInternal<T> : never;
/**
 * See {@link Path}
 */
type FieldPath<TFieldValues extends FieldValues> = Path<TFieldValues>;
/**
 * Helper type for recursively constructing paths through a type.
 * This actually constructs the strings and recurses into nested
 * object types.
 *
 * See {@link ArrayPath}
 */
type ArrayPathImpl<K extends string | number, V, TraversedTypes> = V extends Primitive | BrowserNativeObject ? IsAny<V> extends true ? string : never : V extends ReadonlyArray<infer U> ? U extends Primitive | BrowserNativeObject ? IsAny<V> extends true ? string : never : true extends AnyIsEqual<TraversedTypes, V> ? never : `${K}` | `${K}.${ArrayPathInternal<V, TraversedTypes | V>}` : true extends AnyIsEqual<TraversedTypes, V> ? never : `${K}.${ArrayPathInternal<V, TraversedTypes | V>}`;
/**
 * Helper type for recursively constructing paths through a type.
 * This obscures the internal type param TraversedTypes from exported contract.
 *
 * See {@link ArrayPath}
 */
type ArrayPathInternal<T, TraversedTypes = T> = T extends ReadonlyArray<infer V> ? IsTuple<T> extends true ? { [K in TupleKeys<T>]-?: ArrayPathImpl<K & string, T[K], TraversedTypes>; }[TupleKeys<T>] : ArrayPathImpl<ArrayKey, V, TraversedTypes> : { [K in keyof T]-?: ArrayPathImpl<K & string, T[K], TraversedTypes>; }[keyof T];
/**
 * Type which eagerly collects all paths through a type which point to an array
 * type.
 * @typeParam T - type which should be introspected.
 * @example
 * ```
 * Path<{foo: {bar: string[], baz: number[]}}> = 'foo.bar' | 'foo.baz'
 * ```
 */
type ArrayPath<T> = T extends any ? ArrayPathInternal<T> : never;
/**
 * See {@link ArrayPath}
 */
type FieldArrayPath<TFieldValues extends FieldValues> = ArrayPath<TFieldValues>;
/**
 * Type to evaluate the type which the given path points to.
 * @typeParam T - deeply nested type which is indexed by the path
 * @typeParam P - path into the deeply nested type
 * @example
 * ```
 * PathValue<{foo: {bar: string}}, 'foo.bar'> = string
 * PathValue<[number, string], '1'> = string
 * ```
 */
type PathValue<T, P extends Path<T> | ArrayPath<T>> = PathValueImpl<T, P>;
type PathValueImpl<T, P extends string> = T extends any ? P extends `${infer K}.${infer R}` ? K extends keyof T ? undefined extends T[K] ? PathValueImpl<T[K], R> | undefined : PathValueImpl<T[K], R> : K extends `${ArrayKey}` ? T extends ReadonlyArray<infer V> ? PathValueImpl<V, R> : never : never : P extends keyof T ? T[P] : P extends `${ArrayKey}` ? T extends ReadonlyArray<infer V> ? V : T extends undefined ? undefined : never : never : never;
/**
 * See {@link PathValue}
 */
type FieldPathValue<TFieldValues extends FieldValues, TFieldPath extends FieldPath<TFieldValues>> = PathValue<TFieldValues, TFieldPath>;
/**
 * See {@link PathValue}
 */
type FieldArrayPathValue<TFieldValues extends FieldValues, TFieldArrayPath extends FieldArrayPath<TFieldValues>> = PathValue<TFieldValues, TFieldArrayPath>;
/**
 * Type to evaluate the type which the given paths point to.
 * @typeParam TFieldValues - field values which are indexed by the paths
 * @typeParam TPath        - paths into the deeply nested field values
 * @example
 * ```
 * FieldPathValues<{foo: {bar: string}}, ['foo', 'foo.bar']>
 *   = [{bar: string}, string]
 * ```
 */
type FieldPathValues<TFieldValues extends FieldValues, TPath extends FieldPath<TFieldValues>[] | readonly FieldPath<TFieldValues>[]> = {} & { [K in keyof TPath]: FieldPathValue<TFieldValues, TPath[K] & FieldPath<TFieldValues>>; };
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/fieldArray.d.ts
type FieldArray<TFieldValues extends FieldValues = FieldValues, TFieldArrayName extends FieldArrayPath<TFieldValues> = FieldArrayPath<TFieldValues>> = FieldArrayPathValue<TFieldValues, TFieldArrayName> extends ReadonlyArray<infer U> | null | undefined ? U : never;
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/resolvers.d.ts
type ResolverSuccess<TTransformedValues> = {
  values: TTransformedValues;
  errors: Record<string, never>;
};
type ResolverError<TFieldValues extends FieldValues = FieldValues> = {
  values: Record<string, never>;
  errors: FieldErrors<TFieldValues>;
};
type ResolverResult<TFieldValues extends FieldValues = FieldValues, TTransformedValues = TFieldValues> = ResolverSuccess<TTransformedValues> | ResolverError<TFieldValues>;
interface ResolverOptions<TFieldValues extends FieldValues> {
  criteriaMode?: CriteriaMode;
  fields: Record<InternalFieldName, Field['_f']>;
  names?: FieldName<TFieldValues>[];
  shouldUseNativeValidation: boolean | undefined;
}
type Resolver<TFieldValues extends FieldValues = FieldValues, TContext = any, TTransformedValues = TFieldValues> = (values: TFieldValues, context: TContext | undefined, options: ResolverOptions<TFieldValues>) => Promise<ResolverResult<TFieldValues, TTransformedValues>> | ResolverResult<TFieldValues, TTransformedValues>;
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/form.d.ts
declare const $NestedValue: unique symbol;
/**
 * @deprecated to be removed in the next major version
 */
type NestedValue<TValue extends object = object> = {
  [$NestedValue]: never;
} & TValue;
type DefaultValues<TFieldValues> = TFieldValues extends AsyncDefaultValues<TFieldValues> ? DeepPartial$1<Awaited<TFieldValues>> : DeepPartial$1<TFieldValues>;
type InternalNameSet = Set<InternalFieldName>;
type ValidationMode = typeof VALIDATION_MODE;
type Mode = keyof ValidationMode;
type CriteriaMode = 'firstError' | 'all';
type SubmitHandler<T> = (data: T, event?: React$1.BaseSyntheticEvent) => unknown | Promise<unknown>;
type SubmitErrorHandler<TFieldValues extends FieldValues> = (errors: FieldErrors<TFieldValues>, event?: React$1.BaseSyntheticEvent) => unknown | Promise<unknown>;
type SetValueConfig = Partial<{
  shouldValidate: boolean;
  shouldDirty: boolean;
  shouldTouch: boolean;
}>;
type TriggerConfig = Partial<{
  shouldFocus: boolean;
}>;
type ResetFieldConfig<TFieldValues extends FieldValues, TFieldName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> = Partial<{
  keepDirty: boolean;
  keepTouched: boolean;
  keepError: boolean;
  defaultValue: FieldPathValue<TFieldValues, TFieldName>;
}>;
type ChangeHandler = (event: {
  target: any;
  type?: any;
}) => Promise<void | boolean>;
type AsyncDefaultValues<TFieldValues> = (payload?: unknown) => Promise<TFieldValues>;
type UseFormProps<TFieldValues extends FieldValues = FieldValues, TContext = any, TTransformedValues = TFieldValues> = Partial<{
  mode: Mode;
  disabled: boolean;
  reValidateMode: Exclude<Mode, 'onTouched' | 'all'>;
  defaultValues: DefaultValues<TFieldValues> | AsyncDefaultValues<TFieldValues>;
  values: TFieldValues;
  errors: FieldErrors<TFieldValues>;
  resetOptions: Parameters<UseFormReset<TFieldValues>>[1];
  resolver: Resolver<TFieldValues, TContext, TTransformedValues>;
  context: TContext;
  shouldFocusError: boolean;
  shouldUnregister: boolean;
  shouldUseNativeValidation: boolean;
  progressive: boolean;
  criteriaMode: CriteriaMode;
  delayError: number;
  formControl?: Omit<UseFormReturn<TFieldValues, TContext, TTransformedValues>, 'formState'>;
}>;
type FieldNamesMarkedBoolean<TFieldValues extends FieldValues> = DeepMap<DeepPartial$1<TFieldValues>, boolean>;
type FormStateProxy<TFieldValues extends FieldValues = FieldValues> = {
  isDirty: boolean;
  isValidating: boolean;
  dirtyFields: FieldNamesMarkedBoolean<TFieldValues>;
  touchedFields: FieldNamesMarkedBoolean<TFieldValues>;
  validatingFields: FieldNamesMarkedBoolean<TFieldValues>;
  errors: boolean;
  isValid: boolean;
};
type ReadFormState = { [K in keyof FormStateProxy]: boolean | 'all'; } & {
  values?: boolean;
};
type FormState<TFieldValues extends FieldValues> = {
  isDirty: boolean;
  isLoading: boolean;
  isSubmitted: boolean;
  isSubmitSuccessful: boolean;
  isSubmitting: boolean;
  isValidating: boolean;
  isValid: boolean;
  disabled: boolean;
  submitCount: number;
  defaultValues?: undefined | Readonly<DeepPartial$1<TFieldValues>>;
  dirtyFields: Partial<Readonly<FieldNamesMarkedBoolean<TFieldValues>>>;
  touchedFields: Partial<Readonly<FieldNamesMarkedBoolean<TFieldValues>>>;
  validatingFields: Partial<Readonly<FieldNamesMarkedBoolean<TFieldValues>>>;
  errors: FieldErrors<TFieldValues>;
  isReady: boolean;
};
type KeepStateOptions = Partial<{
  keepDirtyValues: boolean;
  keepErrors: boolean;
  keepDirty: boolean;
  keepValues: boolean;
  keepDefaultValues: boolean;
  keepIsSubmitted: boolean;
  keepIsSubmitSuccessful: boolean;
  keepTouched: boolean;
  keepIsValidating: boolean;
  keepIsValid: boolean;
  keepSubmitCount: boolean;
  keepFieldsRef: boolean;
}>;
type RefCallBack = (instance: any) => void;
type UseFormRegisterReturn<TFieldName extends InternalFieldName = InternalFieldName> = {
  onChange: ChangeHandler;
  onBlur: ChangeHandler;
  ref: RefCallBack;
  name: TFieldName;
  min?: string | number;
  max?: string | number;
  maxLength?: number;
  minLength?: number;
  pattern?: string;
  required?: boolean;
  disabled?: boolean;
};
/**
 * Register field into hook form with or without the actual DOM ref. You can invoke register anywhere in the component including at `useEffect`.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/register) • [Demo](https://codesandbox.io/s/react-hook-form-register-ts-ip2j3) • [Video](https://www.youtube.com/watch?v=JFIpCoajYkA)
 *
 * @param name - the path name to the form field value, name is required and unique
 * @param options - register options include validation, disabled, unregister, value as and dependent validation
 *
 * @returns onChange, onBlur, name, ref, and native contribute attribute if browser validation is enabled.
 *
 * @example
 * ```tsx
 * // Register HTML native input
 * <input {...register("input")} />
 * <select {...register("select")} />
 *
 * // Register options
 * <textarea {...register("textarea", { required: "This is required.", maxLength: 20 })} />
 * <input type="number" {...register("name2", { valueAsNumber: true })} />
 * <input {...register("name3", { deps: ["name2"] })} />
 *
 * // Register custom field at useEffect
 * useEffect(() => {
 *   register("name4");
 *   register("name5", { value: "hiddenValue" });
 * }, [register])
 *
 * // Register without ref
 * const { onChange, onBlur, name } = register("name6")
 * <input onChange={onChange} onBlur={onBlur} name={name} />
 * ```
 */
type UseFormRegister<TFieldValues extends FieldValues> = <TFieldName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>(name: TFieldName, options?: RegisterOptions<TFieldValues, TFieldName>) => UseFormRegisterReturn<TFieldName>;
type SetFocusOptions = Partial<{
  shouldSelect: boolean;
}>;
/**
 * Set focus on a registered field. You can start to invoke this method after all fields are mounted to the DOM.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/setfocus) • [Demo](https://codesandbox.io/s/setfocus-rolus)
 *
 * @param name - the path name to the form field value.
 * @param options - input focus behavior options
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   setFocus("name");
 * }, [setFocus])
 * // shouldSelect allows to select input's content on focus
 * <button onClick={() => setFocus("name", { shouldSelect: true })}>Focus</button>
 * ```
 */
type UseFormSetFocus<TFieldValues extends FieldValues> = <TFieldName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>(name: TFieldName, options?: SetFocusOptions) => void;
type EitherOption<T> = { [K in keyof T]: { [P in K]: T[P]; } & Partial<Record<Exclude<keyof T, K>, never>>; }[keyof T];
type GetValuesConfig = EitherOption<{
  dirtyFields: boolean;
  touchedFields: boolean;
}>;
type UseFormGetValues<TFieldValues extends FieldValues> = {
  /**
   * Get the entire form values when no argument is supplied to this function.
   *
   * @remarks
   * [API](https://react-hook-form.com/docs/useform/getvalues) • [Demo](https://codesandbox.io/s/react-hook-form-v7-ts-getvalues-txsfg)
   *
   * @returns form values
   *
   * @example
   * ```tsx
   * <button onClick={() => getValues()}>getValues</button>
   *
   * <input {...register("name", {
   *   validate: (value, formValues) => formValues.otherField === value;
   * })} />
   * ```
   */
  (name?: undefined, config?: GetValuesConfig): TFieldValues;
  /**
   * Get a single field value.
   *
   * @remarks
   * [API](https://react-hook-form.com/docs/useform/getvalues) • [Demo](https://codesandbox.io/s/react-hook-form-v7-ts-getvalues-txsfg)
   *
   * @param name - the path name to the form field value.
   * @param config - return touched or dirty fields
   *
   * @returns the single field value
   *
   * @example
   * ```tsx
   * <button onClick={() => getValues("name")}>getValues</button>
   *
   * <input {...register("name", {
   *   validate: () => getValues('otherField') === "test";
   * })} />
   * ```
   */
  <TFieldName extends FieldPath<TFieldValues>>(name: TFieldName, config?: GetValuesConfig): FieldPathValue<TFieldValues, TFieldName>;
  /**
   * Get an array of field values.
   *
   * @remarks
   * [API](https://react-hook-form.com/docs/useform/getvalues) • [Demo](https://codesandbox.io/s/react-hook-form-v7-ts-getvalues-txsfg)
   *
   * @param names - an array of field names
   * @param config - return touched or dirty fields
   *
   * @returns An array of field values
   *
   * @example
   * ```tsx
   * <button onClick={() => getValues(["name", "name1"])}>getValues</button>
   *
   * <input {...register("name", {
   *   validate: () => getValues(["fieldA", "fieldB"]).includes("test");
   * })} />
   * ```
   */
  <TFieldNames extends FieldPath<TFieldValues>[]>(names: readonly [...TFieldNames], config?: GetValuesConfig): [...FieldPathValues<TFieldValues, TFieldNames>];
};
/**
 * This method will return individual field states. It will be useful when you are trying to retrieve the nested value field state in a typesafe approach.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/getfieldstate) • [Demo](https://codesandbox.io/s/getfieldstate-jvekk)
 *
 * @param name - the path name to the form field value.
 *
 * @returns invalid, isDirty, isTouched and error object
 *
 * @example
 * ```tsx
 * // those formState has to be subscribed
 * const { formState: { dirtyFields, errors, touchedFields } } = formState();
 * getFieldState('name')
 * // Get field state when form state is not subscribed yet
 * getFieldState('name', formState)
 *
 * // It's ok to combine with useFormState
 * const formState = useFormState();
 * getFieldState('name')
 * getFieldState('name', formState)
 * ```
 */
type UseFormGetFieldState<TFieldValues extends FieldValues> = <TFieldName extends FieldPath<TFieldValues>>(name: TFieldName, formState?: FormState<TFieldValues>) => {
  invalid: boolean;
  isDirty: boolean;
  isTouched: boolean;
  isValidating: boolean;
  error?: FieldError;
};
/**
 * This method will allow you to subscribe to formState without component render
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/subscribe) • [Demo](https://codesandbox.io/s/subscribe)
 *
 * @param options - subscription options on which formState subscribe to
 *
 * @example
 * ```tsx
const { subscribe } = useForm()

useEffect(() => {
 subscribe({
   formState: { isDirty: true },
   callback: () => {}
 })
})
 * ```
 */
type UseFormSubscribe<TFieldValues extends FieldValues> = <TFieldNames extends readonly FieldPath<TFieldValues>[]>(payload: {
  name?: readonly [...TFieldNames] | TFieldNames[number];
  formState?: Partial<ReadFormState>;
  callback: (data: Partial<FormState<TFieldValues>> & {
    values: TFieldValues;
    name?: InternalFieldName;
    type?: EventType;
  }) => void;
  exact?: boolean;
}) => () => void;
type UseFormWatch<TFieldValues extends FieldValues> = {
  /**
   * Watch and subscribe to the entire form update/change based on onChange and re-render at the useForm.
   *
   * @remarks
   * [API](https://react-hook-form.com/docs/useform/watch) • [Demo](https://codesandbox.io/s/react-hook-form-watch-v7-ts-8et1d) • [Video](https://www.youtube.com/watch?v=3qLd69WMqKk)
   *
   * @returns return the entire form values
   *
   * @example
   * ```tsx
   * const formValues = watch();
   * ```
   */
  (): TFieldValues;
  /**
   * Watch and subscribe to an array of fields used outside of render.
   *
   * @remarks
   * [API](https://react-hook-form.com/docs/useform/watch) • [Demo](https://codesandbox.io/s/react-hook-form-watch-v7-ts-8et1d) • [Video](https://www.youtube.com/watch?v=3qLd69WMqKk)
   *
   * @param names - an array of field names
   * @param defaultValue - defaultValues for the entire form
   *
   * @returns return an array of field values
   *
   * @example
   * ```tsx
   * const [name, name1] = watch(["name", "name1"]);
   * ```
   */
  <TFieldNames extends readonly FieldPath<TFieldValues>[]>(names: readonly [...TFieldNames], defaultValue?: DeepPartial$1<TFieldValues>): FieldPathValues<TFieldValues, TFieldNames>;
  /**
   * Watch and subscribe to a single field used outside of render.
   *
   * @remarks
   * [API](https://react-hook-form.com/docs/useform/watch) • [Demo](https://codesandbox.io/s/react-hook-form-watch-v7-ts-8et1d) • [Video](https://www.youtube.com/watch?v=3qLd69WMqKk)
   *
   * @param name - the path name to the form field value.
   * @param defaultValue - defaultValues for the entire form
   *
   * @returns return the single field value
   *
   * @example
   * ```tsx
   * const name = watch("name");
   * ```
   */
  <TFieldName extends FieldPath<TFieldValues>>(name: TFieldName, defaultValue?: FieldPathValue<TFieldValues, TFieldName>): FieldPathValue<TFieldValues, TFieldName>;
  /**
   * Subscribe to field update/change without trigger re-render
   *
   * @remarks
   * [API](https://react-hook-form.com/docs/useform/watch) • [Demo](https://codesandbox.io/s/react-hook-form-watch-v7-ts-8et1d) • [Video](https://www.youtube.com/watch?v=3qLd69WMqKk)
   *
   * @param callback - call back function to subscribe all fields change and return unsubscribe function
   * @param defaultValues - defaultValues for the entire form
   *
   * @returns unsubscribe function
   *
   * @example
   * ```tsx
   * useEffect(() => {
   *   const { unsubscribe } = watch((value) => {
   *     console.log(value);
   *   });
   *   return () => unsubscribe();
   * }, [watch])
   * ```
   */
  (callback: WatchObserver<TFieldValues>, defaultValues?: DeepPartial$1<TFieldValues>): Subscription;
};
/**
 * Trigger field or form validation
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/trigger) • [Demo](https://codesandbox.io/s/react-hook-form-v7-ts-triggervalidation-forked-xs7hl) • [Video](https://www.youtube.com/watch?v=-bcyJCDjksE)
 *
 * @param name - provide empty argument will trigger the entire form validation, an array of field names will validate an array of fields, and a single field name will only trigger that field's validation.
 * @param options - should focus on the error field
 *
 * @returns validation result
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   trigger();
 * }, [trigger])
 *
 * <button onClick={async () => {
 *   const result = await trigger(); // result will be a boolean value
 * }}>
 *  trigger
 *  </button>
 * ```
 */
type UseFormTrigger<TFieldValues extends FieldValues> = (name?: FieldPath<TFieldValues> | FieldPath<TFieldValues>[] | readonly FieldPath<TFieldValues>[], options?: TriggerConfig) => Promise<boolean>;
/**
 * Clear the entire form errors.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/clearerrors) • [Demo](https://codesandbox.io/s/react-hook-form-v7-ts-clearerrors-w3ymx)
 *
 * @param name - the path name to the form field value.
 *
 * @example
 * Clear all errors
 * ```tsx
 * clearErrors(); // clear the entire form error
 * clearErrors(["name", "name1"]) // clear an array of fields' error
 * clearErrors("name2"); // clear a single field error
 * ```
 */
type UseFormClearErrors<TFieldValues extends FieldValues> = (name?: FieldPath<TFieldValues> | FieldPath<TFieldValues>[] | readonly FieldPath<TFieldValues>[] | `root.${string}` | 'root') => void;
/**
 * Set a single field value, or a group of fields value.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/setvalue) • [Demo](https://codesandbox.io/s/react-hook-form-v7-ts-setvalue-8z9hx) • [Video](https://www.youtube.com/watch?v=qpv51sCH3fI)
 *
 * @param name - the path name to the form field value.
 * @param value - field value
 * @param options - should validate or update form state
 *
 * @example
 * ```tsx
 * // Update a single field
 * setValue('name', 'value', {
 *   shouldValidate: true, // trigger validation
 *   shouldTouch: true, // update touched fields form state
 *   shouldDirty: true, // update dirty and dirty fields form state
 * });
 *
 * // Update a group fields
 * setValue('root', {
 *   a: 'test', // setValue('root.a', 'data')
 *   b: 'test1', // setValue('root.b', 'data')
 * });
 *
 * // Update a nested object field
 * setValue('select', { label: 'test', value: 'Test' });
 * ```
 */
type UseFormSetValue<TFieldValues extends FieldValues> = <TFieldName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>(name: TFieldName, value: FieldPathValue<TFieldValues, TFieldName>, options?: SetValueConfig) => void;
/**
 * Set an error for the field. When set an error which is not associated to a field then manual `clearErrors` invoke is required.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/seterror) • [Demo](https://codesandbox.io/s/react-hook-form-v7-ts-seterror-nfxxu) • [Video](https://www.youtube.com/watch?v=raMqvE0YyIY)
 *
 * @param name - the path name to the form field value.
 * @param error - an error object which contains type and optional message
 * @param options - whether or not to focus on the field
 *
 * @example
 * ```tsx
 * // when the error is not associated with any fields, `clearError` will need to invoke to clear the error
 * const onSubmit = () => setError("serverError", { type: "server", message: "Error occurred"})
 *
 * <button onClick={() => setError("name", { type: "min" })} />
 *
 * // focus on the input after setting the error
 * <button onClick={() => setError("name", { type: "max" }, { shouldFocus: true })} />
 * ```
 */
type UseFormSetError<TFieldValues extends FieldValues> = (name: FieldPath<TFieldValues> | `root.${string}` | 'root', error: ErrorOption, options?: {
  shouldFocus: boolean;
}) => void;
/**
 * Unregister a field reference and remove its value.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/unregister) • [Demo](https://codesandbox.io/s/react-hook-form-unregister-4k2ey) • [Video](https://www.youtube.com/watch?v=TM99g_NW5Gk&feature=emb_imp_woyt)
 *
 * @param name - the path name to the form field value.
 * @param options - keep form state options
 *
 * @example
 * ```tsx
 * register("name", { required: true })
 *
 * <button onClick={() => unregister("name")} />
 * // there are various keep options to retain formState
 * <button onClick={() => unregister("name", { keepErrors: true })} />
 * ```
 */
type UseFormUnregister<TFieldValues extends FieldValues> = (name?: FieldPath<TFieldValues> | FieldPath<TFieldValues>[] | readonly FieldPath<TFieldValues>[], options?: Omit<KeepStateOptions, 'keepIsSubmitted' | 'keepSubmitCount' | 'keepValues' | 'keepDefaultValues' | 'keepErrors'> & {
  keepValue?: boolean;
  keepDefaultValue?: boolean;
  keepError?: boolean;
}) => void;
/**
 * Validate the entire form. Handle submit and error callback.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/handlesubmit) • [Demo](https://codesandbox.io/s/react-hook-form-handlesubmit-ts-v7-lcrtu) • [Video](https://www.youtube.com/watch?v=KzcPKB9SOEk)
 *
 * @param onValid - callback function invoked after form pass validation
 * @param onInvalid - callback function invoked when form failed validation
 *
 * @returns callback - return callback function
 *
 * @example
 * ```tsx
 * const onSubmit = (data) => console.log(data);
 * const onError = (error) => console.log(error);
 *
 * <form onSubmit={handleSubmit(onSubmit, onError)} />
 * ```
 */
type UseFormHandleSubmit<TFieldValues extends FieldValues, TTransformedValues = TFieldValues> = (onValid: SubmitHandler<TTransformedValues>, onInvalid?: SubmitErrorHandler<TFieldValues>) => (e?: React$1.BaseSyntheticEvent) => Promise<void>;
/**
 * Reset a field state and reference.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/resetfield) • [Demo](https://codesandbox.io/s/priceless-firefly-d0kuv) • [Video](https://www.youtube.com/watch?v=IdLFcNaEFEo)
 *
 * @param name - the path name to the form field value.
 * @param options - keep form state options
 *
 * @example
 * ```tsx
 * <input {...register("firstName", { required: true })} />
 * <button type="button" onClick={() => resetField("firstName"))}>Reset</button>
 * ```
 */
type UseFormResetField<TFieldValues extends FieldValues> = <TFieldName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>>(name: TFieldName, options?: ResetFieldConfig<TFieldValues, TFieldName>) => void;
type ResetAction<TFieldValues> = (formValues: TFieldValues) => TFieldValues;
/**
 * Reset at the entire form state.
 *
 * @remarks
 * [API](https://react-hook-form.com/docs/useform/reset) • [Demo](https://codesandbox.io/s/react-hook-form-reset-v7-ts-pu901) • [Video](https://www.youtube.com/watch?v=qmCLBjyPwVk)
 *
 * @param values - the entire form values to be reset
 * @param keepStateOptions - keep form state options
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   // reset the entire form after component mount or form defaultValues is ready
 *   reset({
 *     fieldA: "test"
 *     fieldB: "test"
 *   });
 * }, [reset])
 *
 * // reset by combine with existing form values
 * reset({
 *   ...getValues(),
 *  fieldB: "test"
 *});
 *
 * // reset and keep form state
 * reset({
 *   ...getValues(),
 *}, {
 *   keepErrors: true,
 *   keepDirty: true
 *});
 * ```
 */
type UseFormReset<TFieldValues extends FieldValues> = (values?: DefaultValues<TFieldValues> | TFieldValues | ResetAction<TFieldValues>, keepStateOptions?: KeepStateOptions) => void;
type WatchInternal<TFieldValues> = (fieldNames?: InternalFieldName | InternalFieldName[], defaultValue?: DeepPartial$1<TFieldValues>, isMounted?: boolean, isGlobal?: boolean) => FieldPathValue<FieldValues, InternalFieldName> | FieldPathValues<FieldValues, InternalFieldName[]>;
type GetIsDirty = <TName extends InternalFieldName, TData>(name?: TName, data?: TData) => boolean;
type FormStateSubjectRef<TFieldValues extends FieldValues> = Subject<Partial<FormState<TFieldValues>> & {
  name?: InternalFieldName;
  values?: TFieldValues;
  type?: EventType;
}>;
type Subjects<TFieldValues extends FieldValues = FieldValues> = {
  array: Subject<{
    name?: InternalFieldName;
    values?: FieldValues;
  }>;
  state: FormStateSubjectRef<TFieldValues>;
};
type Names = {
  mount: InternalNameSet;
  unMount: InternalNameSet;
  disabled: InternalNameSet;
  array: InternalNameSet;
  watch: InternalNameSet;
  focus?: InternalFieldName;
  watchAll?: boolean;
};
type BatchFieldArrayUpdate = <T extends Function, TFieldValues extends FieldValues, TFieldArrayName extends FieldArrayPath<TFieldValues> = FieldArrayPath<TFieldValues>>(name: InternalFieldName, updatedFieldArrayValues?: Partial<FieldArray<TFieldValues, TFieldArrayName>>[], method?: T, args?: Partial<{
  argA: unknown;
  argB: unknown;
}>, shouldSetValue?: boolean, shouldUpdateFieldsAndErrors?: boolean) => void;
type FromSubscribe<TFieldValues extends FieldValues> = <TFieldNames extends readonly FieldPath<TFieldValues>[]>(payload: {
  name?: readonly [...TFieldNames] | TFieldNames[number];
  formState?: Partial<ReadFormState>;
  callback: (data: Partial<FormState<TFieldValues>> & {
    values: TFieldValues;
    name?: InternalFieldName;
  }) => void;
  exact?: boolean;
  reRenderRoot?: boolean;
}) => () => void;
type Control<TFieldValues extends FieldValues = FieldValues, TContext = any, TTransformedValues = TFieldValues> = {
  _subjects: Subjects<TFieldValues>;
  _removeUnmounted: Noop;
  _names: Names;
  _state: {
    mount: boolean;
    action: boolean;
    watch: boolean;
  };
  _reset: UseFormReset<TFieldValues>;
  _options: UseFormProps<TFieldValues, TContext, TTransformedValues>;
  _getDirty: GetIsDirty;
  _resetDefaultValues: Noop;
  _formState: FormState<TFieldValues>;
  _setValid: (shouldUpdateValid?: boolean) => void;
  _fields: FieldRefs;
  _formValues: FieldValues;
  _proxyFormState: ReadFormState;
  _defaultValues: Partial<DefaultValues<TFieldValues>>;
  _getWatch: WatchInternal<TFieldValues>;
  _setFieldArray: BatchFieldArrayUpdate;
  _getFieldArray: <TFieldArrayValues>(name: InternalFieldName) => Partial<TFieldArrayValues>[];
  _setErrors: (errors: FieldErrors<TFieldValues>) => void;
  _setDisabledField: (props: {
    disabled?: boolean;
    name: FieldName<any>;
  }) => void;
  _runSchema: (names: InternalFieldName[]) => Promise<{
    errors: FieldErrors;
  }>;
  _updateIsValidating: (names?: InternalFieldName[], isValidating?: boolean) => void;
  _focusError: () => boolean | undefined;
  _disableForm: (disabled?: boolean) => void;
  _subscribe: FromSubscribe<TFieldValues>;
  register: UseFormRegister<TFieldValues>;
  handleSubmit: UseFormHandleSubmit<TFieldValues, TTransformedValues>;
  unregister: UseFormUnregister<TFieldValues>;
  getFieldState: UseFormGetFieldState<TFieldValues>;
  setError: UseFormSetError<TFieldValues>;
};
type WatchObserver<TFieldValues extends FieldValues> = (value: DeepPartialSkipArrayKey<TFieldValues>, info: {
  name?: FieldPath<TFieldValues>;
  type?: EventType;
  values?: unknown;
}) => void;
type UseFormReturn<TFieldValues extends FieldValues = FieldValues, TContext = any, TTransformedValues = TFieldValues> = {
  watch: UseFormWatch<TFieldValues>;
  getValues: UseFormGetValues<TFieldValues>;
  getFieldState: UseFormGetFieldState<TFieldValues>;
  setError: UseFormSetError<TFieldValues>;
  clearErrors: UseFormClearErrors<TFieldValues>;
  setValue: UseFormSetValue<TFieldValues>;
  trigger: UseFormTrigger<TFieldValues>;
  formState: FormState<TFieldValues>;
  resetField: UseFormResetField<TFieldValues>;
  reset: UseFormReset<TFieldValues>;
  handleSubmit: UseFormHandleSubmit<TFieldValues, TTransformedValues>;
  unregister: UseFormUnregister<TFieldValues>;
  control: Control<TFieldValues, TContext, TTransformedValues>;
  register: UseFormRegister<TFieldValues>;
  setFocus: UseFormSetFocus<TFieldValues>;
  subscribe: UseFormSubscribe<TFieldValues>;
};
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/utils.d.ts
type Noop = () => void;
interface File$1 extends Blob {
  readonly lastModified: number;
  readonly name: string;
}
interface FileList$1 {
  readonly length: number;
  item(index: number): File$1 | null;
  [index: number]: File$1;
}
type Primitive = null | undefined | string | number | boolean | symbol | bigint;
type BrowserNativeObject = Date | FileList$1 | File$1;
type NonUndefined<T> = T extends undefined ? never : T;
type LiteralUnion<T extends U, U extends Primitive> = T | (U & {
  _?: never;
});
type ExtractObjects<T> = T extends (infer U) ? U extends object ? U : never : never;
type IsPrimitiveLike<T> = T extends Primitive ? true : T extends Primitive & object ? true : false;
type DeepPartial$1<T> = IsPrimitiveLike<T> extends true ? T : T extends BrowserNativeObject | NestedValue ? T : { [K in keyof T]?: ExtractObjects<T[K]> extends never ? T[K] : DeepPartial$1<T[K]>; };
type DeepPartialSkipArrayKey<T> = IsPrimitiveLike<T> extends true ? T : T extends BrowserNativeObject | NestedValue ? T : T extends ReadonlyArray<any> ? { [K in keyof T]: DeepPartialSkipArrayKey<T[K]>; } : { [K in keyof T]?: DeepPartialSkipArrayKey<T[K]>; };
/**
 * Checks whether the type is any
 * See {@link https://stackoverflow.com/a/49928360/3406963}
 * @typeParam T - type which may be any
 * ```
 * IsAny<any> = true
 * IsAny<string> = false
 * ```
 */
type IsAny<T> = 0 extends 1 & T ? true : false;
/**
 * Checks whether T1 can be exactly (mutually) assigned to T2
 * @typeParam T1 - type to check
 * @typeParam T2 - type to check against
 * ```
 * IsEqual<string, string> = true
 * IsEqual<'foo', 'foo'> = true
 * IsEqual<string, number> = false
 * IsEqual<string, number> = false
 * IsEqual<string, 'foo'> = false
 * IsEqual<'foo', string> = false
 * IsEqual<'foo' | 'bar', 'foo'> = boolean // 'foo' is assignable, but 'bar' is not (true | false) -> boolean
 * ```
 */
type IsEqual<T1, T2> = T1 extends T2 ? (<G>() => G extends T1 ? 1 : 2) extends (<G>() => G extends T2 ? 1 : 2) ? true : false : false;
type DeepMap<T, TValue> = IsAny<T> extends true ? any : T extends BrowserNativeObject | NestedValue ? TValue : T extends object ? { [K in keyof T]: DeepMap<NonUndefined<T[K]>, TValue>; } : TValue;
type IsFlatObject<T extends object> = Extract<Exclude<T[keyof T], NestedValue | Date | FileList$1>, any[] | object> extends never ? true : false;
type Merge<A, B> = { [K in keyof A | keyof B]?: K extends keyof A & keyof B ? [A[K], B[K]] extends [object, object] ? Merge<A[K], B[K]> : B[K] : K extends keyof A ? A[K] : K extends keyof B ? B[K] : never; };
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/fields.d.ts
type InternalFieldName = string;
type FieldName<TFieldValues extends FieldValues> = IsFlatObject<TFieldValues> extends true ? Extract<keyof TFieldValues, string> : string;
type CustomElement<TFieldValues extends FieldValues> = Partial<HTMLElement> & {
  name: FieldName<TFieldValues>;
  type?: string;
  value?: any;
  disabled?: boolean;
  checked?: boolean;
  options?: HTMLOptionsCollection;
  files?: FileList | null;
  focus?: Noop;
};
type FieldValues = Record<string, any>;
type FieldElement<TFieldValues extends FieldValues = FieldValues> = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | CustomElement<TFieldValues>;
type Ref = FieldElement;
type Field = {
  _f: {
    ref: Ref;
    name: InternalFieldName;
    refs?: HTMLInputElement[];
    mount?: boolean;
  } & RegisterOptions;
};
type FieldRefs = Partial<{
  [key: InternalFieldName]: Field | FieldRefs;
}>;
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/errors.d.ts
type Message = string;
type MultipleFieldErrors = { [K in keyof RegisterOptions]?: ValidateResult; } & {
  [key: string]: ValidateResult;
};
type FieldError = {
  type: LiteralUnion<keyof RegisterOptions, string>;
  root?: FieldError;
  ref?: Ref;
  types?: MultipleFieldErrors;
  message?: Message;
};
type ErrorOption = {
  message?: Message;
  type?: LiteralUnion<keyof RegisterOptions, string>;
  types?: MultipleFieldErrors;
};
type DeepRequired<T> = T extends BrowserNativeObject | Blob ? T : { [K in keyof T]-?: NonNullable<DeepRequired<T[K]>>; };
type FieldErrorsImpl<T extends FieldValues = FieldValues> = { [K in keyof T]?: T[K] extends BrowserNativeObject | Blob ? FieldError : K extends 'root' | `root.${string}` ? GlobalError : T[K] extends object ? Merge<FieldError, FieldErrorsImpl<T[K]>> : FieldError; };
type GlobalError = Partial<{
  type: string | number;
  message: Message;
}>;
type FieldErrors<T extends FieldValues = FieldValues> = Partial<FieldValues extends IsAny<FieldValues> ? any : FieldErrorsImpl<DeepRequired<T>>> & {
  root?: Record<string, GlobalError> & GlobalError;
};
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/validator.d.ts
type ValidationValue = boolean | number | string | RegExp;
type ValidationRule<TValidationValue extends ValidationValue = ValidationValue> = TValidationValue | ValidationValueMessage<TValidationValue>;
type ValidationValueMessage<TValidationValue extends ValidationValue = ValidationValue> = {
  value: TValidationValue;
  message: Message;
};
type ValidateResult = Message | Message[] | boolean | undefined;
type Validate<TFieldValue, TFormValues> = (value: TFieldValue, formValues: TFormValues) => ValidateResult | Promise<ValidateResult>;
type RegisterOptions<TFieldValues extends FieldValues = FieldValues, TFieldName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>> = Partial<{
  required: Message | ValidationRule<boolean>;
  min: ValidationRule<number | string>;
  max: ValidationRule<number | string>;
  maxLength: ValidationRule<number>;
  minLength: ValidationRule<number>;
  validate: Validate<FieldPathValue<TFieldValues, TFieldName>, TFieldValues> | Record<string, Validate<FieldPathValue<TFieldValues, TFieldName>, TFieldValues>>;
  value: FieldPathValue<TFieldValues, TFieldName>;
  setValueAs: (value: any) => any;
  shouldUnregister?: boolean;
  onChange?: (event: any) => void;
  onBlur?: (event: any) => void;
  disabled: boolean;
  deps: FieldPath<TFieldValues> | FieldPath<TFieldValues>[];
}> & ({
  pattern?: ValidationRule<RegExp>;
  valueAsNumber?: false;
  valueAsDate?: false;
} | {
  pattern?: undefined;
  valueAsNumber?: false;
  valueAsDate?: true;
} | {
  pattern?: undefined;
  valueAsNumber?: true;
  valueAsDate?: false;
});
//#endregion
//#region ../../node_modules/.pnpm/react-hook-form@7.71.2_react@19.2.7/node_modules/react-hook-form/dist/types/controller.d.ts
type UseControllerProps<TFieldValues extends FieldValues = FieldValues, TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>, TTransformedValues = TFieldValues> = {
  name: TName;
  rules?: Omit<RegisterOptions<TFieldValues, TName>, 'valueAsNumber' | 'valueAsDate' | 'setValueAs' | 'disabled'>;
  shouldUnregister?: boolean;
  defaultValue?: FieldPathValue<TFieldValues, TName>;
  control?: Control<TFieldValues, any, TTransformedValues>;
  disabled?: boolean;
  exact?: boolean;
};
//#endregion
//#region src/types.d.ts
interface UseControllerComponentProps {
  useControllerProps: UseControllerProps<any>;
  formFieldProps?: FormFieldProps;
}
//#endregion
//#region src/components/form/ControlledSelect/index.d.ts
type SelectValue<T extends 'single' | 'multiple'> = T extends 'single' ? string : string[];
type SelectRenderValue<T extends 'single' | 'multiple'> = T extends 'single' ? (value: string) => ReactNode : (value: string[]) => ReactNode;
interface Props$2<T extends 'single' | 'multiple' = 'single'> extends Omit<ComponentProps<typeof Select>, 'onChange' | 'defaultValue'>, UseControllerComponentProps {
  hideError?: boolean;
  loading?: boolean;
  onChange?: (value: SelectValue<T>) => void;
  kind?: T;
  renderValue?: SelectRenderValue<T>;
}
export declare const ControlledSelect: <T extends 'single' | 'multiple' = 'single'>({ kind, loading, hideError, onChange, renderValue, formFieldProps, useControllerProps, ...selectProps }: Props$2<T>) => import("react/jsx-runtime").JSX.Element;
//#endregion
//#region src/components/form/ControlledTextArea/index.d.ts
interface Props$1 extends Omit<ComponentProps<typeof TextArea>, 'onChange'>, UseControllerComponentProps {
  label?: string;
  onChange?: (value: string) => void;
}
export declare const ControlledTextArea: ({ useControllerProps, label, formFieldProps, attributes, onChange, ...props }: Props$1) => import("react/jsx-runtime").JSX.Element;
//#endregion
//#region src/components/form/ControlledTextInput/index.d.ts
interface Props extends ComponentProps<typeof TextInput>, UseControllerComponentProps {
  label?: ReactNode | string;
  slotEnd?: ReactNode;
  /**
   * When true, selects all text in the input when focused.
   * Useful for pre-populated inputs where users typically want to replace the entire value.
   */
  selectOnFocus?: boolean;
  /** When true, validation errors are not shown on the field (e.g. parent shows a summary). */
  hideError?: boolean;
  /**
   * When true, renders the value as a masked password field with a visibility toggle.
   * Incompatible with `type="number"`.
   */
  masked?: boolean;
}
export declare const ControlledTextInput: ({ useControllerProps, label, 'aria-label': ariaLabel, formFieldProps, required, slotEnd, selectOnFocus, hideError, masked, attributes: textInputAttributes, ...props }: Props) => import("react/jsx-runtime").JSX.Element;
//#endregion
//#region src/hooks/useStudioDataViewState/index.d.ts
/**
 * Extract the options type from DataView.useDataViewState (excluding undefined)
 */
type DataViewStateOptions = NonNullable<Parameters<typeof useDataViewState>[0]>;
/**
 * Options for useStudioDataViewState hook.
 * Accepts the same options as DataView.useDataViewState, plus URL sync defaults.
 * Note: `pagination` and `sorting` are read from URL params, not from options.
 */
interface UseStudioDataViewStateOptions extends Omit<DataViewStateOptions, 'pagination' | 'sorting'> {
  /** Default page number (1-based) when not specified in URL. Defaults to 1. */
  defaultPage?: number;
  /** Default page size when not specified in URL. Defaults to 50. */
  defaultPageSize?: number;
  /**
   * Default sorting when not specified in URL. Always an ordered list — a single-column default is
   * just an array of one. Uses the same format as DataView's sorting state.
   * Example: [{ id: 'created_at', desc: true }] for descending by created_at.
   */
  defaultSort?: SortEntry[];
  /**
   * Enable multi-column sort. When true, the `sort` URL param and the sorting state round-trip an
   * ordered, comma-separated list of fields (e.g. shift-click a second column header). Defaults to
   * false — single-sort, exactly as before.
   */
  multiSort?: boolean;
  /** Maps a filter column id to the API key it's emitted under (id used as-is when absent). Also
   * accepts a function `(id) => key | undefined` for dynamic ids, e.g. `latency_ms`→`latency_ms.mean`. */
  filterFieldMap?: Record<string, string> | ((id: string) => string | undefined);
}
interface SortEntry {
  id: string;
  desc: boolean;
}
/**
 * API filter object exposed by the hook. The `filter` shape is parameterized by
 * `FilterType` so consumers get type-checked filter keys and values without
 * needing `as` casts at the call site.
 *
 * Defaults to `Record<string, unknown>` for callers that don't supply a type.
 */
interface ApiFilter<FilterType = Record<string, unknown>> {
  searchText?: string;
  filter?: Partial<FilterType>;
}
/**
 * Extended DataView state that includes helper functions for common table operations.
 */
interface StudioDataViewState<FilterType = Record<string, unknown>> extends DataViewState {
  /**
   * Resets pagination to page 1 and clears row selection.
   * Call this when search/filter criteria change to ensure users see results from the beginning.
   */
  resetPagination: () => void;
  /** Debounced search bar value (300ms). Use this for API queries. */
  debouncedSearchBar: string;
  /** Debounced column filters (300ms). Use this for API queries. */
  debouncedColumnFilters: ColumnFiltersState;
  /**
   * Convention-mapped API filter object built from debounced columnFilters and searchBar.
   * - `columnFilters` entries map to `filter` keys: `{id, value}` → `filter[id] = value`,
   *   unless a `filterFieldMap` entry remaps the id to a different API key.
   * - `searchBar` is exposed as `searchText` when non-empty. Consumers are responsible for
   *   mapping `searchText` onto the appropriate filter field (and wrapping it in an operator
   *   like `$like` if fuzzy matching is desired).
   */
  apiFilter: ApiFilter<FilterType>;
  /** Clears searchBar, columnFilters, and resets pagination. */
  resetFilters: () => void;
}
/**
 * Opinionated React hook that wraps {@link DataView.useDataViewState} to synchronize
 * table state (pagination and sorting) with URL search parameters.
 *
 * Ensures DataView's pagination and sorting state and the URL's query params are always in sync,
 * so that users can share/bookmark current view state, and can use browser navigation
 * controls (back/forward) seamlessly.
 *
 * @param {UseStudioDataViewStateOptions} [options] - Accepts all options for DataView's useDataViewState
 *   except `pagination` and `sorting`, plus `defaultPage`, `defaultPageSize`, and `defaultSort`
 *   for setting fallback URL values.
 *
 * @returns {StudioDataViewState} - Extended DataView state object with URL sync and helper functions.
 *
 * @remarks
 * - Page numbers in the URL are 1-based, while DataView expects 0-based pageIndex.
 * - Sort in URL uses string format: "field" for ascending, "-field" for descending.
 * - defaultSort uses DataView's object format: { id: 'field', desc: boolean }.
 * - Pagination automatically resets to page 1 when sorting changes.
 * - Use `resetPagination()` when search/filter criteria change.
 * - Intended for table/data grid views where browser-driven navigation is desirable.
 */
export declare const useStudioDataViewState: <FilterType = Record<string, unknown>>(options?: UseStudioDataViewStateOptions) => StudioDataViewState<FilterType>;
//#endregion
//#region src/api/filterOperators.d.ts
/**
 * Mongo-style comparison operators accepted by NeMo Platform's unified filter
 * syntax (e.g. `{ name: { $like: '%foo%' } }`, `{ created_at: { $gte, $lte } }`).
 *
 * The OpenAPI-generated SDK types model filter fields as bare scalars
 * (`name?: string`) and do not expose the operator-object form. Use
 * {@link WithFilterOperators} to widen a generated filter shape so call sites
 * can build operator filters without `as unknown as` casts.
 */
interface FilterOperators<V> {
  $eq?: V;
  $ne?: V;
  $in?: readonly V[];
  $nin?: readonly V[];
  $gt?: V;
  $gte?: V;
  $lt?: V;
  $lte?: V;
  /** Substring match (e.g. `%foo%`). String-typed regardless of `V`. */
  $like?: string;
}
/**
 * Widens each field of `F` so it accepts either its original scalar value or
 * a {@link FilterOperators} object. Intended for API call sites that need
 * operator-object filters; coerce back to the generated filter type once at
 * the SDK boundary.
 *
 * @example
 * type ModelFilterInput = WithFilterOperators<ModelEntityFilter>;
 * const filter: ModelFilterInput = { name: { $like: '%foo%' } };
 */
type WithFilterOperators<F> = { [K in keyof F]?: F[K] | FilterOperators<NonNullable<F[K]>>; };
/**
 * Build a typed operator-filter and coerce it to the generated SDK filter type
 * in one step. Centralizes the unavoidable cast at the SDK boundary so call
 * sites can be written with full type-checking on operator objects.
 *
 * @example
 * filter: withOperators<FilesetFilter>({ name: { $like: `%${search}%` } })
 */
export declare const withOperators: <F>(filter: WithFilterOperators<F>) => F;
//#endregion
//#region src/api/fetchAllPages.d.ts
interface PaginatedResponse<T> {
  data?: T[];
  pagination?: {
    total_pages?: number;
  };
}
interface FetchAllPagesOptions {
  pageSize?: number;
  maxPages?: number;
}
/**
 * Drains a page-numbered list endpoint. Takes the fetcher rather than the SDK
 * function so it stays free of `@nemo/sdk` and can ship in the plugin surface.
 */
export declare const fetchAllPages: <T>(fetchPage: (page: number, pageSize: number) => Promise<PaginatedResponse<T>>, { pageSize, maxPages }?: FetchAllPagesOptions) => Promise<T[]>;
//#endregion
//#region src/utils/query.d.ts
export declare const getJobRefetchInterval: (status?: PlatformJobStatus) => number | false;
export declare const getSortParam: (sortingState: SortingState) => string;
/**
 * Maps DataView sorting state to a sort query value when the API only allows specific fields.
 * Table columns often use client-only ids (e.g. model_name); URL bookmarking can also reference invalid ids.
 *
 * Pass a generated sort-field enum as `allowedFieldIds` to get that enum back rather than a bare
 * `string`, so the result drops straight into the SDK params. Descending sort prefixes an allowed
 * field with `-`; the enum is expected to carry both directions, as the generated ones do.
 */
export declare const getSortParamWithWhitelist: <S extends string = string>(sortingState: SortingState, allowedFieldIds: readonly S[], fallbackWhenEmptyOrInvalid: S) => S;
//#endregion
//#region src/utils/file.d.ts
/**
 * Triggers a browser download for a file.
 * @param data - The data to be downloaded (e.g., ArrayBuffer, Blob, File).
 * @param filename - The name to give the downloaded file.
 */
export declare const triggerDownload: (data: BlobPart, filename: string) => void;
//#endregion
//#region src/utils/entityName.d.ts
export declare const ENTITY_NAME_HELP = "Must start with a lowercase letter and be 2–63 characters. Lowercase letters, numbers, and - _ . @ + only. No consecutive or trailing hyphens.";
/** Zod string schema enforcing `ENTITY_NAME_REGEXP` with per-rule error messages. */
export declare function entityNameSchema(label?: string): ZodEffects<ZodString, string, string>;
//#endregion
//#region src/utils/error.d.ts
/** Display message from an unknown error; falls back when it is not an Error. */
export declare function getErrorMessage(error: unknown, defaultMessage?: string): string;
//#endregion
//#region src/utils/forms/error.d.ts
/**
 * Use as a sane default for form errors.
 * Handles parsing the errors and logging them to the website logger.
 * @param errors - The errors to handle. Any shape
 */
type HandleFormErrorGenericProps = {
  title?: string;
};
export declare const handleFormErrorsGeneric: ({ title }: HandleFormErrorGenericProps) => (errors: FieldErrors) => void;
//#endregion
//#region src/utils/logger.d.ts
/**
 * Wrapper class around the global OpenTelemetry Logger. Logs will be transported to:
 * 1. The console
 * 2. OpenTelemetry Collector (except in local and test envs - see `telemetry.ts`)
 *
 * We export a global instance of this class below (`websiteLogger`). That variable
 * should be used by individual modules, in-place of `console.log`.
 *
 * Example usage:
 *
 * ```
 * import { logger } from '@nemo/common/src/utils/logger';
 * logger.info('Some message to log');
 * ```
 */
declare class WebsiteLogger {
  private log;
  debug(message: string, cause?: unknown): void;
  error(message: string, cause?: unknown): void;
  info(message: string, cause?: unknown): void;
  warn(message: string, cause?: unknown): void;
}
export declare const logger: WebsiteLogger;
/**
 * Converts a unknown error to an Error object.
 */
export declare function toError(err: unknown): Error;
//#endregion
//#region src/constants/pagination.d.ts
/** Default page size used in any table, i.e. projects, models, datasets */
export declare const DEFAULT_PAGE_SIZE_OPTIONS: number[];
/** Default page used in any table, i.e. projects, models, datasets */
export declare const DEFAULT_PAGE = 1;
/** Default page size used in any table, i.e. projects, models, datasets */
export declare const DEFAULT_PAGE_SIZE = 50;
//#endregion
//#region src/constants/query.d.ts
export declare const CJobCancellableStatuses: PlatformJobStatus[];
export declare const CJobLaunchableStatuses: PlatformJobStatus[];
export declare const CJobTerminalStatuses: PlatformJobStatus[];
export declare const PlatformJobTerminalStatuses: PlatformJobStatus[];
//#endregion
export { type AccordionSectionProps, type ApiFilter, type AssistantChatMessageContentProps, type AssistantChatProps, type BadgeStatus, type CreateSecretFormData, type CreateSecretModalProps, index_d_exports as DataView, type FetchAllPagesOptions, type FileTagProps, type FileTagStatus, type FileUploadProps, type FilterOperators, type FormModalProps, type NotifyFn, type NotifyType, type PaginatedResponse, type QuickActionItem, type RadioCardProps, type RenderFileTagFn, type StatusConfigEntry, type StudioDataViewState, type StudioDataViewToolbarProps, type UseStudioDataViewStateOptions, type WithFilterOperators };