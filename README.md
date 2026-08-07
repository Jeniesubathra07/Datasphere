# Datasphere

A data platform starter with a FastAPI development API.

## Development

Install dependencies:

```bash
bash .cursor/scripts/install.sh
```

Run the API server:

```bash
python3 -m uvicorn datasphere.app:app --host 0.0.0.0 --port 8000 --reload
```

Run tests:

```bash
python3 -m pytest
```

## API

- `GET /health` — service health check
- `GET /api/datasets` — list sample datasets
