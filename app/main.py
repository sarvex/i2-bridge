from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import Settings
from app.core.telemetry import setup_telemetry
from app.api.v1.routes import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    setup_telemetry()
    yield
    # Cleanup
    pass

def create_app() -> FastAPI:
    settings = Settings()
    
    app = FastAPI(
        title="i2-bridge",
        version=settings.VERSION,
        description="Integration bridge service",
        docs_url=f"{settings.API_V1_STR}/docs",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(
        health.router,
        prefix=settings.API_V1_STR,
        tags=["health"]
    )

    return app

app = create_app() 