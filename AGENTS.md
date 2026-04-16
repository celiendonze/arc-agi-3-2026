# Development

## Run
```bash
python arc_game_1.py
```

## Prerequisites
- **Ollama** must be running at `http://localhost:11434` with model `gemma4:e2b`
- `uv sync` to install dependencies

## Architecture
- `arc_game_1.py` - Main entry point (AI agent playing ARC-AGI game)
- `main.py` - Stub entry point (unused)
- `src/arc_agi_3_2026/` - Package source
- `data/images/` - Generated frames (gitignored)
- `data/thinking/` - Model thinking logs (gitignored)

## Virtual Environment
`.venv/` is gitignored. Activate with `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)

## Notes
- Uses `uv` for dependency management
- Python 3.13 required
- `uv.lock` is committed
