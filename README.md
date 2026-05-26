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

## Installing from GitHub

Publishing is not required for OpenClaw to load this plugin. Clone the private repo, build it, and point OpenClaw at the local checkout:

```bash
git clone git@github.com:Betanu701/openclaw-contextforge.git
cd openclaw-contextforge
npm install
npm run build
```

On Windows, the plugin path in OpenClaw should be the cloned folder, for example:

```json5
{
  plugins: {
    load: { paths: ["C:/Users/you/src/openclaw-contextforge"] },
    entries: {
      contextforge: {
        enabled: true,
        hooks: { allowPromptInjection: true, allowConversationAccess: true },
        config: { serviceUrl: "http://192.168.3.8:8765" }
      }
    },
    slots: { memory: "contextforge" }
  }
}
```

Publish later only if you want one-command installation from npm or GitHub Packages.

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

Current benchmark result is committed at `benchmarks/results/conversation-large-100x12k.json`:

| Metric | Result |
| --- | ---: |
| Transcript size | 1,200,300 estimated tokens |
| Turns | 100 |
| Tokens per turn | 12,000 |
| Needles | 12 |
| ContextForge chunks | 3,085 |
| Recall budget | 4,096 tokens |
| ContextForge source hits | 12/12 |
| Native 40,960-token window visibility | 2/12 |

For model comparisons, keep the same OpenClaw model/provider for baseline and ContextForge-enabled runs, disable other memory plugins, and report retrieval source-hit rate, answer accuracy, and latency separately.
