from functools import lru_cache

from faster_whisper import WhisperModel


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    """
    Loads the Whisper model once per process and reuses it for every
    transcription request across all exercises (counting, reading, etc).

    Without this, instantiating WhisperModel per-request reloads the
    model from disk on every API call, which is slow and wasteful.
    """

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )