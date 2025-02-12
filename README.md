# i2_bridge

Project structure:
``` bash
i2-bridge/
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── pyproject.toml
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── telemetry.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       └── health.py
│   │   ├── controllers/
│   │   │   ├── __init__.py
│   │   │   ├── base_controller.py
│   │   │   └── health_controller.py
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── base_repository.py
│   │   │   └── health_repository.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── base_model.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── health.py
│   ├── deploy/
│   │   └── helm/
│   │       └── i2-bridge/
│   │           ├── Chart.yaml
│   │           ├── values.yaml
│   │           └── templates/
│   │               ├── deployment.yaml
│   │               ├── service.yaml
│   │               ├── configmap.yaml
│   │               └── ingress.yaml
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       └── api/
│           └── v1/
│               └── test_health.py 
``` 

## Setup

```bash
pip install poetry
poetry install
```

## Run locally
```bash
cp .env.example .env
poetry run uvicorn app.main:app --reload
```

## Deploy
```bash
docker build -t i2-bridge:1.0.0 .
helm install i2-bridge ./deploy/helm/i2-bridge
```

## Test
```bash
poetry run pytest