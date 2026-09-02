export type ContextForgeConfig = {
  serviceUrl: string;
  namespacePrefix: string;
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

export type ContextRequest = RecallRequest & {
  includePermanent?: boolean;
};

export type ContextResponse = RecallResponse & {
  permanentTokens: number;
  branchPaths: string[];
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

export type PermanentContextRequest = {
  namespace: ContextForgeNamespace;
  text: string;
  title?: string;
};

export type PermanentContextResponse = {
  id: string;
  tokens: number;
};

export type SessionRequest = {
  namespace: ContextForgeNamespace;
  sessionId?: string;
  metadata?: Record<string, unknown>;
};

export type SessionMessageRequest = {
  namespace: ContextForgeNamespace;
  sessionId?: string;
  role: string;
  content: string;
};

export type SessionResponse = {
  id: string;
  resumed: boolean;
  messageCount: number;
  totalTokens: number;
  metadata: Record<string, unknown>;
};

export type SessionsResponse = {
  sessions: SessionResponse[];
};

export type ChatRequest = {
  namespace: ContextForgeNamespace;
  message: string;
  sessionId?: string;
  category?: string;
  maxTokens?: number;
  limit?: number;
  modelKwargs?: Record<string, unknown>;
};

export type ChatResponse = {
  response: string;
  sessionId: string;
  context: ContextResponse;
  latencyMs: number;
};

export type AnalyzeRequest = ChatRequest & {
  maxPasses?: number;
};

export type AnalyzeResponse = {
  response: string;
  sessionId: string;
  contexts: ContextResponse[];
  latencyMs: number;
};

export type StatsResponse = {
  dbPath: string;
  namespace?: string;
  totalNodes: number;
  indexedNodes: number;
  indexedTerms: number;
  categories: Record<string, number>;
  cache: Record<string, unknown>;
  sessions?: number;
  permanentContextTokens?: number;
  modelConfigured?: boolean;
};
