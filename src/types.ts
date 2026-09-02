export type ContextForgeConfig = {
  serviceUrl: string;
  namespacePrefix: string;
  autoRecall: boolean;
  autoCapture: boolean;
  recallMaxTokens: number;
  autoRecallLimit: number;
  recallMaxChars: number;
  captureMaxChars: number;
  autoRecallTimeoutMs: number;
  timeoutMs: number;
  category: string;
  includePermanentContext: boolean;
  allowedCategories: string[];
  blockedCategories: string[];
  minScore?: number;
  maxSourceTokens?: number;
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

export type ContextPlanEntry = {
  id: string;
  path: string;
  title: string;
  category: string;
  tier: string;
  score: number;
  originalTokens: number;
  plannedTokens: number;
  disposition: string;
  reason: string;
  matchedTerms: string[];
};

export type ContextPlan = {
  strategy: string;
  maxTokens: number;
  recallBudget: number;
  requestedLimit: number;
  candidateCount: number;
  selectedCount: number;
  droppedCount: number;
  compactedCount: number;
  totalTokens: number;
  budgets: Record<string, number>;
  items: ContextPlanEntry[];
  dropped: ContextPlanEntry[];
};

export type RecallRequest = {
  namespace: ContextForgeNamespace;
  query: string;
  conversationContext?: string;
  category?: string;
  allowedCategories?: string[];
  blockedCategories?: string[];
  minScore?: number;
  maxSourceTokens?: number;
  maxTokens?: number;
  limit?: number;
};

export type RecallResponse = {
  context: string;
  sources: ContextForgeSource[];
  totalTokens: number;
  latencyMs: number;
  plan: ContextPlan;
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
  allowedCategories?: string[];
  blockedCategories?: string[];
  minScore?: number;
  maxSourceTokens?: number;
  includePermanent?: boolean;
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
