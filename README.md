# ARC-AGI Game AI

An AI agent that plays ARC-AGI games using a vision-capable language model via Ollama.

## Setup

```bash
uv sync
```

### Ollama

Install [Ollama](https://ollama.ai/) and pull the model:

```bash
ollama pull gemma4:e2b
```

Ensure Ollama is running at `http://localhost:11434`.

## Run

```bash
python arc_game_1.py
```

The agent will:
1. Launch an ARC-AGI game environment (ls20 task)
2. For each step, send the current frame to the AI
3. Execute the action chosen by the AI
4. Save frames to `data/images/`
5. Display a grid of all frames at the end

## Dependencies

- **arc-agi** - ARC-AGI game environment
- **pydantic-ai** - AI agent framework
- **matplotlib** - Frame visualization
- **opencv-python** - Image processing
- **loguru** - Logging
