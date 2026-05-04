from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .routers import advisor, games, research, sim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_game.json"


def create_app() -> FastAPI:
    app = FastAPI(title="Coup Detection API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(games.router)
    app.include_router(advisor.router)
    app.include_router(sim.router)
    app.include_router(research.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/sample-game")
    def sample_game() -> JSONResponse:
        if not SAMPLE_PATH.exists():
            return JSONResponse({"detail": "sample missing"}, status_code=404)
        return FileResponse(SAMPLE_PATH, media_type="application/json")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
