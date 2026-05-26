# OpenClaw ContextForge

OpenClaw ContextForge is an external OpenClaw memory plugin plus a Python HTTP sidecar that gives OpenClaw access to ContextForge hierarchical memory without patching OpenClaw core.

## Architecture

- OpenClaw loads the `contextforge` plugin as the active `plugins.slots.memory` implementation.
- The TypeScript plugin calls a local sidecar during `before_prompt_build` and injects bounded, delimited memory context.
- Explicit memory capture happens through `contextforge_remember` or user turns containing phrases such as `remember this`.
- The sidecar stores namespaced ContextForge nodes in SQLite and filters recall by OpenClaw session/user/channel namespace.

## Local development

```bash
npm install
npm run build
PYTHONPATH=../contextforge:. python3 -m pytest sidecar/tests -q
python3 benchmarks/needle_haystack.py self-test
```

Run the sidecar locally:

```bash
./scripts/run-sidecar-dev.sh
```

Install the plugin into a local OpenClaw config:

```bash
./scripts/install-openclaw-plugin.sh
```

Then configure OpenClaw:

```json5
{
  plugins: {
    load: { paths: ["/path/to/openclaw-contextforge"] },
    entries: {
      contextforge: {
        enabled: true,
        hooks: { allowPromptInjection: true, allowConversationAccess: true },
        config: { serviceUrl: "http://localhost:8765" }
      },
      "memory-lancedb": { enabled: false },
      "active-memory": { enabled: false }
    },
    slots: { memory: "contextforge" }
  }
}
```

## Overwatch compose

```bash
cd deploy/overwatch
docker compose up --build
```

The example compose file builds the plugin, starts the ContextForge sidecar with persistent `/data/contextforge.db`, and mounts this repo into the OpenClaw container at `/plugins/contextforge`. Copy the `openclaw.config.example.json5` settings into the active OpenClaw config for the container.

Useful sidecar environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONTEXTFORGE_DB_PATH` | `/data/contextforge.db` | SQLite persistence path. |
| `CONTEXTFORGE_MAX_CONTEXT_TOKENS` | `4096` | Maximum recall context returned by the sidecar. |
| `CONTEXTFORGE_MAX_NODE_TOKENS` | `768` | Ingest-time chunk size so large documents are indexed as retrievable snippets instead of one oversized node. |
| `CONTEXTFORGE_INGEST_ROOT` | unset | Required root directory for path ingestion. |

## Benchmark

Run the deterministic generator self-test:

```bash
python3 benchmarks/needle_haystack.py self-test
```

Run a retrieval-only needle-in-the-haystack check against a running sidecar:

```bash
python3 benchmarks/needle_haystack.py retrieval --sidecar-url http://localhost:8765 --tokens 10000 --needles 10
```

Run the long multi-turn conversation decay benchmark:

```bash
python3 benchmarks/needle_haystack.py conversation \
  --sidecar-url http://localhost:8765 \
  --turns 100 \
  --tokens-per-turn 12000 \
  --needles 12 \
  --control-needles 2 \
  --native-window-tokens 40960
```

This builds a roughly 1.2M-token transcript with most needles in the first 10 turns and a small control set near the end. The report compares ContextForge source-hit accuracy against whether each answer is still visible in a native recent-context window.

For model comparisons, keep the same OpenClaw model/provider for baseline and ContextForge-enabled runs, disable other memory plugins, and report retrieval source-hit rate, answer accuracy, and latency separately.
