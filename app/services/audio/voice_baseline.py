from app.services.audio.common import AudioProcessor


class VoiceBaselineAnalyzer:
    """
    Voice baseline exercise.

    Extracts only what this exercise needs:
      - peak intensity (dBFS)
      - mean pitch F0 (Hz)

    Skips duration, rms, silence_ratio, spectral features and MFCC
    entirely, since none are used here - this avoids the heaviest
    librosa calls (spectral_* and mfcc) for an endpoint that doesn't
    need them.
    """

    def __init__(self, filepath: str, patient_id: int, session_id: str,):

        self.filepath = filepath
        self.patient_id = patient_id
        self.session_id = session_id
        self.processor = AudioProcessor(filepath)

        self.peak_intensity_db = None
        self.mean_pitch_hz = None

    # -------------------------------------------------------
    # Analyze
    # -------------------------------------------------------

    def analyze(self) -> dict:

        self.processor.load()
        self.processor.validate()

        self.peak_intensity_db = self.processor.peak_db()
        self.mean_pitch_hz = round(self.processor.pitch(), 2)

        return self.build_response()

    # -------------------------------------------------------
    # Build Response
    # -------------------------------------------------------

    def build_response(self) -> dict:

        return {
            "success": True,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "assessment_type": "voice_baseline",
            "result_data": {
            "peak_intensity_db": self.peak_intensity_db,
            "mean_pitch_hz": self.mean_pitch_hz
            }
        }   
    