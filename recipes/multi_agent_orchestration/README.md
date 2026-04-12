# Multi-Agent Orchestration with Llama

A recipe for building multi-agent systems using Llama models, with context layers, agent discovery, and safety wrappers.

## What You'll Build

A 3-agent pipeline where:
1. **Researcher** analyzes a topic and produces structured findings
2. **Builder** creates a solution based on research
3. **Reviewer** evaluates the solution against quality criteria

Each agent has a persistent identity, discovers others via capability cards, and every LLM call is wrapped with safety guards.

## Concepts Demonstrated

| Concept | What | Why |
|---|---|---|
| **Context Layers** | Agents have identity (static) + state (dynamic) + knowledge (searchable) | Consistent behavior across sessions |
| **Agent Cards** | JSON capability descriptors per agent | Agents find each other without hardcoding |
| **Safety Guard** | Output validation, cost caps, rollback | Production-grade reliability |
| **Phase Gates** | Research → Build → Review with quality checks | Prevent garbage from propagating |

## Quick Start

```bash
pip install -r requirements.txt
# Set your Llama API endpoint (or use ollama)
export LLAMA_BASE_URL=http://localhost:11434/v1
export LLAMA_MODEL=llama3.2

python orchestrator.py "Build a CLI tool that converts CSV to JSON"
```

## Files

| File | What |
|---|---|
| `context_layers.py` | 4-layer context system (identity → state → relevant → archive) |
| `agent_cards.py` | Agent capability discovery and routing |
| `safety_guard.py` | Output validation, cost tracking, rollback |
| `orchestrator.py` | 3-agent pipeline with phase gates |
| `requirements.txt` | Dependencies |

## How It Works

```
User provides task
    ↓
Orchestrator reads agent cards → finds Researcher
    ↓
Researcher (context: identity + task) → structured findings
    ↓ [safety guard: validate JSON output]
Orchestrator finds Builder
    ↓
Builder (context: identity + research findings) → solution
    ↓ [safety guard: validate + cost check]
Orchestrator finds Reviewer
    ↓
Reviewer (context: identity + solution + criteria) → evaluation
    ↓ [phase gate: score > 7/10 to pass]
Output: solution + evaluation report
```

Built by [PA·co](https://github.com/PenguinAlleyApps/paco-framework) — A Penguin Alley System.
