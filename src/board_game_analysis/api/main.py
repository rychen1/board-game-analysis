"""Minimal FastAPI application."""

from fastapi import FastAPI

app = FastAPI(title="board-game-analysis")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
