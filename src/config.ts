import type { ContextForgeConfig, ContextForgeMode } from "./types.js";

export const DEFAULT_CONTEXTFORGE_CONFIG: ContextForgeConfig = {
  serviceUrl: "http://contextforge:8765",
  namespacePrefix: "openclaw",
  mode: "contextforge",
  budgetRatio: 0.25,
  autoRecall: true,
  autoCapture: true,
  recallMaxTokens: 4096,
  recallMaxChars: 2000,
  captureMaxChars: 4000,
  autoRecallTimeoutMs: 750,
  timeoutMs: 5000,
  category: "openclaw",
  autoCaptureTriggers: ["remember this", "save this", "contextforge remember"],
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function resolveEnvTemplate(value: string, key: string): string {
  const match = /^\$\{([A-Z0-9_]+)\}$/.exec(value.trim());
  if (!match) {
    return value;
  }
  const envValue = process.env[match[1]];
  if (!envValue) {
    throw new Error(`ContextForge config ${key} references unset environment variable ${match[1]}`);
  }
  return envValue;
}

const CONTEXTFORGE_MODES = new Set<ContextForgeMode>(["off", "contextforge", "hybrid"]);

function readString(
  record: Record<string, unknown>,
  key: keyof ContextForgeConfig,
  fallback: string,
): string {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "string") {
    throw new Error(`ContextForge config ${String(key)} must be a string`);
  }
  const resolved = resolveEnvTemplate(value, String(key)).trim();
  if (!resolved) {
    throw new Error(`ContextForge config ${String(key)} must not be empty`);
  }
  return resolved;
}

function readBoolean(
  record: Record<string, unknown>,
  key: keyof ContextForgeConfig,
  fallback: boolean,
): boolean {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "boolean") {
    throw new Error(`ContextForge config ${String(key)} must be a boolean`);
  }
  return value;
}

/**
 * Reads a bounded numeric config value.
 * Use integer=false for ratio-style values that must preserve decimals.
 */
function readNumber(
  record: Record<string, unknown>,
  key: keyof ContextForgeConfig,
  fallback: number,
  min: number,
  max: number,
  integer = true,
): number {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`ContextForge config ${String(key)} must be a finite number`);
  }
  if (value < min || value > max) {
    throw new Error(`ContextForge config ${String(key)} must be between ${min} and ${max}`);
  }
  return integer ? Math.floor(value) : value;
}

function readMode(
  record: Record<string, unknown>,
  key: keyof ContextForgeConfig,
  fallback: ContextForgeMode,
): ContextForgeMode {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "string" || !CONTEXTFORGE_MODES.has(value as ContextForgeMode)) {
    throw new Error(`ContextForge config ${String(key)} must be one of: off, contextforge, hybrid`);
  }
  return value as ContextForgeMode;
}

function readStringArray(
  record: Record<string, unknown>,
  key: keyof ContextForgeConfig,
  fallback: string[],
): string[] {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (!Array.isArray(value)) {
    throw new Error(`ContextForge config ${String(key)} must be an array`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== "string" || !entry.trim()) {
      throw new Error(`ContextForge config ${String(key)}[${index}] must be a non-empty string`);
    }
    return entry.trim();
  });
}

export function parseContextForgeConfig(value: unknown): ContextForgeConfig {
  const record = asRecord(value);
  const cfg: ContextForgeConfig = {
    serviceUrl: readString(record, "serviceUrl", DEFAULT_CONTEXTFORGE_CONFIG.serviceUrl).replace(
      /\/+$/,
      "",
    ),
    mode: readMode(record, "mode", DEFAULT_CONTEXTFORGE_CONFIG.mode),
    budgetRatio: readNumber(
      record,
      "budgetRatio",
      DEFAULT_CONTEXTFORGE_CONFIG.budgetRatio,
      0.01,
      1,
      false,
    ),
    namespacePrefix: readString(
      record,
      "namespacePrefix",
      DEFAULT_CONTEXTFORGE_CONFIG.namespacePrefix,
    ),
    autoRecall: readBoolean(record, "autoRecall", DEFAULT_CONTEXTFORGE_CONFIG.autoRecall),
    autoCapture: readBoolean(record, "autoCapture", DEFAULT_CONTEXTFORGE_CONFIG.autoCapture),
    recallMaxTokens: readNumber(record, "recallMaxTokens", 4096, 256, 65536, true),
    recallMaxChars: readNumber(record, "recallMaxChars", 2000, 100, 20000, true),
    captureMaxChars: readNumber(record, "captureMaxChars", 4000, 100, 20000, true),
    autoRecallTimeoutMs: readNumber(record, "autoRecallTimeoutMs", 750, 100, 10000, true),
    timeoutMs: readNumber(record, "timeoutMs", 5000, 500, 60000, true),
    category: readString(record, "category", DEFAULT_CONTEXTFORGE_CONFIG.category),
    autoCaptureTriggers: readStringArray(
      record,
      "autoCaptureTriggers",
      DEFAULT_CONTEXTFORGE_CONFIG.autoCaptureTriggers,
    ),
  };

  try {
    new URL(cfg.serviceUrl);
  } catch (error) {
    throw new Error(`ContextForge config serviceUrl must be a valid URL: ${String(error)}`);
  }

  return cfg;
}

export const contextForgeConfigSchema = {
  parse: parseContextForgeConfig,
  uiHints: {
    serviceUrl: {
      label: "ContextForge service URL",
      placeholder: DEFAULT_CONTEXTFORGE_CONFIG.serviceUrl,
    },
    mode: { label: "Long-term context mode" },
    budgetRatio: { label: "Context budget ratio", advanced: true },
    namespacePrefix: { label: "Namespace prefix", advanced: true },
    autoRecall: { label: "Auto-recall" },
    autoCapture: { label: "Explicit auto-capture" },
    recallMaxTokens: { label: "Recall max tokens", advanced: true },
    recallMaxChars: { label: "Recall query max chars", advanced: true },
    captureMaxChars: { label: "Capture max chars", advanced: true },
    autoRecallTimeoutMs: { label: "Auto-recall timeout", advanced: true },
    timeoutMs: { label: "Tool timeout", advanced: true },
    category: { label: "Default category", advanced: true },
    autoCaptureTriggers: { label: "Auto-capture trigger phrases", advanced: true },
  },
};
