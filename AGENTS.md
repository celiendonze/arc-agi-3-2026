# Development

## Run
```bash
python main.py
```

## Setup
```bash
uv sync
```

## Virtual Environment
`.venv/` is gitignored. Activate it before running:
- Windows: `.venv\Scripts\activate`
- Unix: `source .venv/bin/activate`

## Notes
- Uses `uv` for dependency management (not pip)
- Python 3.13 required
- `uv.lock` is committed (do not ignore)
