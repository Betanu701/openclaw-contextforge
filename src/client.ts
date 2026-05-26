import type {
  ForgetRequest,
  ForgetResponse,
  IngestRequest,
  IngestResponse,
  RecallRequest,
  RecallResponse,
  RememberRequest,
  RememberResponse,
  StatsResponse,
} from "./types.js";

type TimeoutSignal = {
  signal: AbortSignal;
  cleanup: () => void;
};

function withTimeout(timeoutMs: number, upstream?: AbortSignal): TimeoutSignal {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${timeoutMs}ms`)), timeoutMs);
  const abortFromUpstream = () => controller.abort(upstream?.reason);
  if (upstream?.aborted) {
    abortFromUpstream();
  } else {
    upstream?.addEventListener("abort", abortFromUpstream, { once: true });
  }
  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer);
      upstream?.removeEventListener("abort", abortFromUpstream);
    },
  };
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`ContextForge sidecar returned ${response.status}: ${text.slice(0, 800)}`);
  }
  if (!text.trim()) {
    throw new Error("ContextForge sidecar returned an empty response");
  }
  return JSON.parse(text) as T;
}

export class ContextForgeClient {
  readonly serviceUrl: string;

  constructor(serviceUrl: string) {
    this.serviceUrl = serviceUrl.replace(/\/+$/, "");
  }

  async health(signal: AbortSignal | undefined, timeoutMs: number): Promise<StatsResponse> {
    return await this.request<StatsResponse>("GET", "/healthz", undefined, signal, timeoutMs);
  }

  async recall(
    request: RecallRequest,
    signal: AbortSignal | undefined,
    timeoutMs: number,
  ): Promise<RecallResponse> {
    return await this.request<RecallResponse>("POST", "/recall", request, signal, timeoutMs);
  }

  async remember(
    request: RememberRequest,
    signal: AbortSignal | undefined,
    timeoutMs: number,
  ): Promise<RememberResponse> {
    return await this.request<RememberResponse>("POST", "/remember", request, signal, timeoutMs);
  }

  async ingest(
    request: IngestRequest,
    signal: AbortSignal | undefined,
    timeoutMs: number,
  ): Promise<IngestResponse> {
    return await this.request<IngestResponse>("POST", "/ingest", request, signal, timeoutMs);
  }

  async forget(
    request: ForgetRequest,
    signal: AbortSignal | undefined,
    timeoutMs: number,
  ): Promise<ForgetResponse> {
    return await this.request<ForgetResponse>("POST", "/forget", request, signal, timeoutMs);
  }

  async stats(namespace: string | undefined, signal: AbortSignal | undefined, timeoutMs: number) {
    const suffix = namespace ? `?namespace=${encodeURIComponent(namespace)}` : "";
    return await this.request<StatsResponse>("GET", `/stats${suffix}`, undefined, signal, timeoutMs);
  }

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body: unknown,
    signal: AbortSignal | undefined,
    timeoutMs: number,
  ): Promise<T> {
    const timeout = withTimeout(timeoutMs, signal);
    try {
      const response = await fetch(`${this.serviceUrl}${path}`, {
        method,
        signal: timeout.signal,
        headers: body === undefined ? undefined : { "content-type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      return await parseJsonResponse<T>(response);
    } finally {
      timeout.cleanup();
    }
  }
}

