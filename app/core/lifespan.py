from contextlib import asynccontextmanager

from .ai_loader import ai_models


@asynccontextmanager
async def lifespan(app):

    print("Starting server...")

    ai_models.load()

    yield

    print("Server stopped.")