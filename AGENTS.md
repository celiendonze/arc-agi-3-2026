# ARC-AGI Game AI

## Setup

```bash
uv sync
```

Requires Python >=3.13.

## Ollama

Must be running at `http://localhost:11434`. Pull model:

```bash
ollama pull gemma4:e2b
```

`arc_game_1.py` and `arc_game_tools.py` both default to `gemma4:e2b`. Both files have commented-out alternatives (`qwen3.6:latest`, `gemma4:26b`).

## Run

```bash
python arc_game_1.py
```

Main loop: runs 20 steps of ARC-AGI task `ls20`. Each step sends the current frame to the agent, executes the chosen action, saves frame + thinking log.

## Key files

- `arc_game_1.py` - Active game loop (pydantic-ai Agent, sends frames as PNG images)
- `arc_game_tools.py` - Variant using tool calls (ASCII frame rendering, `step` tool)
- `main.py` - Unused stub
- `src/arc_agi_3_2026/` - Package skeleton (unused)

## Output

- `data/images/` - Per-step frames (gitignored)
- `data/thinking/` - Per-step model responses (gitignored)
