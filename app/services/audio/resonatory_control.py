from app.services.audio.common import AudioProcessor

class ResonatoryControlAnalyzer:
    """
    Resonatory control exercise.

    Extracts only what this exercise needs:
      - peak pressure (dBFS, loudest short-term RMS window)

    Skips scoring and the blocking Laravel call entirely - the
    router handles forwarding the result asynchronously.
    """

    def __init__(self, filepath: str, patient_id: int, session_id: str,):

        self.filepath = filepath
        self.patient_id = patient_id
        self.session_id = session_id
        self.processor = AudioProcessor(filepath)

        self.peak_pressure_db = None

    # -------------------------------------------------------
    # Analyze
    # -------------------------------------------------------

    def analyze(self) -> dict:

        self.processor.load()
        self.processor.validate()

        self.peak_pressure_db = self.processor.peak_rms_db()

        return self.build_response()

    # -------------------------------------------------------
    # Build Response
    # -------------------------------------------------------

    def build_response(self) -> dict:

        return {
            "success": True,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "assessment_type": "resonatory_control",
            "result_data": {
            "peak_pressure_db": self.peak_pressure_db
            }
        }