from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import settings
from app.core.lifespan import lifespan
from app.api.voice_baseline import router as audio_router
from app.api.counting import router as counting_router
from app.api.sustained_phonation import router as sustained_phonation_router
from app.api.resonatory_control import router as resonatory_control_router
from app.api.reading import router as reading_router
from app.api.facial_articulation import router as facial_articulation_router


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.include_router(health_router)
app.include_router(audio_router)
app.include_router(counting_router)
app.include_router(sustained_phonation_router)
app.include_router(resonatory_control_router)
app.include_router(reading_router)
app.include_router(facial_articulation_router)

@app.get("/")
def root():
    return {
        "message": settings.APP_NAME
    }