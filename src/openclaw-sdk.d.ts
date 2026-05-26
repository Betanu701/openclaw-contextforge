declare module "openclaw/plugin-sdk/plugin-entry" {
  export function definePluginEntry<T>(entry: T): T;
}

declare module "openclaw/plugin-sdk" {
  export type AgentToolResult = {
    content: Array<{ type: "text"; text: string }>;
    details?: unknown;
  };

  export type AnyAgentTool = {
    name: string;
    label?: string;
    description?: string;
    parameters: unknown;
    execute(
      toolCallId: string,
      params: unknown,
      signal?: AbortSignal,
      onUpdate?: unknown,
    ): Promise<AgentToolResult>;
  };

  export type OpenClawPluginApi = {
    id: string;
    pluginConfig?: Record<string, unknown>;
    logger: {
      info(message: string): void;
      warn(message: string): void;
      error?(message: string): void;
      debug?(message: string): void;
    };
    registerService(service: { id: string; start: () => void | Promise<void> }): void;
    registerTool(
      tool: AnyAgentTool | ((ctx: Record<string, unknown>) => AnyAgentTool | AnyAgentTool[]),
      opts?: { name?: string; names?: string[]; optional?: boolean },
    ): void;
    on(
      hookName: string,
      handler: (event: any, ctx: any) => unknown,
      opts?: { priority?: number; timeoutMs?: number },
    ): void;
  };
}
