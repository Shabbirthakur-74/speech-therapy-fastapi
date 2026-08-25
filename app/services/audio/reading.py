import re

from app.services.audio.common import AudioProcessor
from app.services.audio.whisper_engine import get_whisper_model


class ReadingAnalyzer:
    """
    Reading exercise.

    Extracts only what this exercise needs:
      - velocity (words per minute)
      - interruptions (pauses between speech segments over threshold)

    Skips paragraph-matching accuracy scoring and the full
    AudioProcessor feature set (only duration() is needed here).
    """

    def __init__(
        self,
        filepath: str,
        patient_id: int,
        session_id: str,
        language: str = "hi",
        pause_threshold_seconds: float = 0.5
    ):

        self.filepath = filepath
        self.patient_id = patient_id
        self.session_id = session_id
        self.language = language
        self.pause_threshold_seconds = pause_threshold_seconds

        self.processor = AudioProcessor(filepath)

        self.transcription = ""
        self.duration_seconds = 0.0

        self.wpm = 0.0
        self.interruptions = 0

    # --------------------------------------------------
    # Speech To Text (with segment timing for pause detection)
    # --------------------------------------------------

    def transcribe(self):

        model = get_whisper_model()

        segments, _ = model.transcribe(
            self.filepath,
            language=self.language
        )

        text = ""
        interruptions = 0
        previous_end = None

        for segment in segments:

            text += segment.text + " "

            if previous_end is not None:

                gap = segment.start - previous_end

                if gap >= self.pause_threshold_seconds:
                    interruptions += 1

            previous_end = segment.end

        self.transcription = text.strip()
        self.interruptions = interruptions

        return self.transcription

    # --------------------------------------------------
    # Normalize Text (for word counting)
    # --------------------------------------------------

    def normalize(self, text):

        text = text.lower()

        text = re.sub(r"[^\u0900-\u097Fa-zA-Z0-9\s]", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    # --------------------------------------------------
    # Velocity (Words Per Minute)
    # --------------------------------------------------

    def calculate_wpm(self):

        total_words = len(
            self.normalize(self.transcription).split()
        )

        if self.duration_seconds <= 0:
            self.wpm = 0.0
        else:
            self.wpm = round(
                total_words * 60 / self.duration_seconds, 2
            )

        return self.wpm

    # --------------------------------------------------
    # Main Analysis
    # --------------------------------------------------

    def analyze(self) -> dict:

        self.processor.load()
        self.processor.validate()
        self.duration_seconds = self.processor.duration()

        self.transcribe()
        self.calculate_wpm()

        return self.build_response()

    def build_response(self) -> dict:

        return {
            "success": True,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "assessment_type": "prosody_reading",
            "result_data": {
                "velocity_wpm": self.wpm,
                "interruptions": self.interruptions
            }
        }