# OpenClaw ContextForge

OpenClaw ContextForge is an external OpenClaw plugin plus a Python HTTP sidecar that gives OpenClaw access to ContextForge hierarchical memory, proactive context assembly, scoped sessions, permanent context, and optional model-backed analysis without patching OpenClaw core.

## Architecture

- OpenClaw loads the `contextforge` plugin as the active `plugins.slots.memory` implementation.
- The TypeScript plugin calls a local sidecar during `before_prompt_build` and injects bounded, delimited ContextForge working context.
- Explicit memory capture happens through `contextforge_remember` or user turns containing phrases such as `remember this`.
- The sidecar stores namespaced ContextForge nodes and sessions in SQLite and filters recall by OpenClaw session/user/channel namespace.
- Permanent context is namespace-scoped, so durable project rules/persona/contracts can be loaded before recalled branches without leaking across users or conversations.
- If the sidecar is configured with an LLM provider, OpenClaw can call ContextForge chat and multi-pass analysis directly; otherwise those tools are disabled with a clear configuration error.

## OpenClaw tools

The plugin exposes these tools to agents:

| Tool | Purpose |
| --- | --- |
| `contextforge_context` | Assemble permanent context plus relevant memory branches for a query. |
| `contextforge_recall` | Search scoped ContextForge memory and return sources. |
| `contextforge_remember` | Store a durable fact, decision, preference, or note. |
| `contextforge_permanent_context` | Set scoped permanent context for future turns. |
| `contextforge_ingest` | Ingest raw text or sidecar-approved files/directories. |
| `contextforge_session` | Start/resume/list sessions or append explicit session messages. |
| `contextforge_chat` | Ask the sidecar model through ContextForge context assembly and session memory. |
| `contextforge_analyze` | Run multi-pass analysis across matching memory categories and synthesize one answer. |
| `contextforge_forget` | Delete a memory by id or inspect deletion candidates. |
| `contextforge_stats` | Show node/index/session/permanent-context/model status. |

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
| `CONTEXTFORGE_LLM_PROVIDER` | unset | Optional provider for sidecar `chat`/`analyze`; e.g. `local`, `openai`, or `anthropic`. |
| `CONTEXTFORGE_LLM_BASE_URL` | unset | Optional OpenAI-compatible endpoint for local/hosted inference. |
| `CONTEXTFORGE_LLM_MODEL` | unset | Optional model/deployment name used by sidecar `chat`/`analyze`. |
| `CONTEXTFORGE_LLM_API_KEY` | unset | Optional API key for sidecar model calls. |

## Benchmark

Run the deterministic generator self-test:

```bash
python3 benchmarks/needle_haystack.py self-test
```

Run a retrieval-only needle-in-the-haystack check against a running sidecar:

```bash
python3 benchmarks/needle_haystack.py retrieval --sidecar-url http://localhost:8765 --tokens 10000 --needles 10
```

Run a Greg Kamradt official-compatible NIAH grid without vendoring the official corpus:

```bash
tmp="$(mktemp -d)"
git clone --depth 1 https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git "$tmp/niah"
python3 benchmarks/needle_haystack.py official \
  --sidecar-url http://localhost:8765 \
  --haystack-dir "$tmp/niah/needlehaystack/PaulGrahamEssays" \
  --context-lengths 4000,8000,16000,32000,40000,64000,128000 \
  --depths 0,10,25,50,75,90,100
rm -rf "$tmp"
```

This mode follows the official single-needle shape: Paul Graham essay haystack, the San Francisco/Dolores Park needle, document-depth insertion, context-length grid, and the original retrieval question. It is a protocol-compatible retrieval run against ContextForge, not a vendored copy of the official benchmark package.

Current official-compatible result is committed at `benchmarks/results/official-compatible-128k-grid.json`:

| Metric | Result |
| --- | ---: |
| Context lengths | 4K, 8K, 16K, 32K, 40K, 64K, 128K |
| Depths | 0, 10, 25, 50, 75, 90, 100 |
| Cases | 49 |
| Recall budget | 4,096 tokens |
| ContextForge source hits | 49/49 |
| Native 40,960-token full-context eligibility | 35/49 |
| Native 40,960-token tail visibility | 42/49 |
| ContextForge chunks | 3,332 |

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

Use both benchmark families in public claims:

| Benchmark | Claim it supports |
| --- | --- |
| Official-compatible NIAH | ContextForge can recover the canonical San Francisco needle across the standard context-length/depth grid. |
| Conversation decay | ContextForge can recover old conversational facts after many turns and far beyond the model's native recent-context window. |

For model comparisons, keep the same OpenClaw model/provider for baseline and ContextForge-enabled runs, disable other memory plugins, and report retrieval source-hit rate, answer accuracy, and latency separately.
