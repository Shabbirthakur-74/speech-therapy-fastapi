from app.services.audio.common import AudioProcessor


class SustainedPhonationAnalyzer:
    """
    Sustained phonation exercise.

    Extracts only what this exercise needs:
      - intensity (dBFS, RMS-based average loudness)
      - duration (seconds)

    Skips pitch stability (librosa.pyin - expensive), loudness
    stability, voice-break detection, and scoring entirely, since
    none are used here.
    """

    def __init__(self, filepath: str, patient_id: int, session_id: str,):

        self.filepath = filepath
        self.patient_id = patient_id
        self.session_id = session_id
        self.processor = AudioProcessor(filepath)
        self.intensity_db = None
        self.duration_seconds = None

    # -------------------------------------------------------
    # Analyze
    # -------------------------------------------------------

    def analyze(self) -> dict:

        self.processor.load()
        self.processor.validate()

        self.intensity_db = self.processor.rms_db()
        self.duration_seconds = round(self.processor.duration(), 2)

        return self.build_response()

    # -------------------------------------------------------
    # Build Response
    # -------------------------------------------------------

    def build_response(self) -> dict:

        return {
            "success": True,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "assessment_type": "phonation",
            "result_data": {
            "intensity_db": self.intensity_db,
            "duration_seconds": self.duration_seconds
            }
        }