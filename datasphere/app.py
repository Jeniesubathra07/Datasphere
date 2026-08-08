"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from datasphere.api.routes import health, model, predict
from datasphere.core.config import APP_VERSION, CORS_ORIGINS, MODELS_PATH
from datasphere.core.model_loader import load_production_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.loaded_model = load_production_model(MODELS_PATH)
        app.state.model_load_error = None
    except Exception as exc:
        app.state.loaded_model = None
        app.state.model_load_error = str(exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="EduRisk Intelligence",
        version=APP_VERSION,
        description="Child Education Risk Intelligence Platform API",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(model.router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "message": "EduRisk Intelligence API",
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict/explain",
        }

    return app


app = create_app()
