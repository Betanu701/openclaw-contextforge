export type ContextForgeMode = "off" | "contextforge" | "hybrid";

export type ContextForgeConfig = {
  serviceUrl: string;
  namespacePrefix: string;
  mode: ContextForgeMode;
  budgetRatio: number;
  autoRecall: boolean;
  autoCapture: boolean;
  recallMaxTokens: number;
  recallMaxChars: number;
  captureMaxChars: number;
  autoRecallTimeoutMs: number;
  timeoutMs: number;
  category: string;
  autoCaptureTriggers: string[];
};

export type ContextForgeNamespace = {
  namespace: string;
  sessionId?: string;
  sessionKey?: string;
  agentId?: string;
  channelId?: string;
  userId?: string;
  workspaceDir?: string;
};

export type ContextForgeSource = {
  id: string;
  path: string;
  title: string;
  category: string;
  score: number;
  tokens: number;
  matchedTerms: string[];
};

export type RecallRequest = {
  namespace: ContextForgeNamespace;
  query: string;
  conversationContext?: string;
  category?: string;
  maxTokens?: number;
  limit?: number;
};

export type RecallResponse = {
  context: string;
  sources: ContextForgeSource[];
  totalTokens: number;
  latencyMs: number;
};

export type RememberRequest = {
  namespace: ContextForgeNamespace;
  text: string;
  title?: string;
  category?: string;
  metadata?: Record<string, unknown>;
};

export type RememberResponse = {
  id: string;
  path: string;
  title: string;
  category: string;
  tokens: number;
};

export type PrepareContextRequest = {
  namespace: ContextForgeNamespace;
  query: string;
  maxContextTokens?: number;
};

export type PreparedContext = {
  context: string;
  sources: ContextForgeSource[];
  totalTokens: number;
  latencyMs: number;
};

export type RecordTurnRequest = {
  namespace: ContextForgeNamespace;
  success: boolean;
  latestUserText?: string;
  runId?: string;
};

export type LongTermContextPlugin = {
  prepareContext(
    request: PrepareContextRequest,
    signal?: AbortSignal,
  ): Promise<PreparedContext | undefined>;
  recordTurn(
    request: RecordTurnRequest,
    signal?: AbortSignal,
  ): Promise<RememberResponse | undefined>;
};

export type IngestRequest = {
  namespace: ContextForgeNamespace;
  text?: string;
  path?: string;
  title?: string;
  category?: string;
  metadata?: Record<string, unknown>;
};

export type IngestResponse = {
  count: number;
  ids: string[];
};

export type ForgetRequest = {
  namespace: ContextForgeNamespace;
  memoryId?: string;
  query?: string;
  confirmTopMatch?: boolean;
  limit?: number;
};

export type ForgetResponse = {
  deleted: string[];
  candidates: ContextForgeSource[];
};

export type StatsResponse = {
  dbPath: string;
  namespace?: string;
  totalNodes: number;
  indexedNodes: number;
  indexedTerms: number;
  categories: Record<string, number>;
  cache: Record<string, unknown>;
};
