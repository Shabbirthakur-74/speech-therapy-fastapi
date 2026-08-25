import json
import re
from app.services.audio.common import AudioProcessor
from app.services.audio.whisper_engine import get_whisper_model


class CountingAnalyzer:
    """
    Counting exercise.

    Transcribes the patient's spoken numbers and reports:
      - how many numbers were detected out of the expected total
      - how many were correct against the expected sequence
      - peak vocal force (dBFS) during the recording

    Does NOT run the full AudioProcessor feature set (spectral
    features, MFCC) - only peak_db() is needed here.
    """

    NUMBER_MAP = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20
    }

    def __init__(
        self,
        filepath: str,
        patient_id: int,
        session_id: str,
        total_expected: int,
        sequence: str
    ):

        self.filepath = filepath
        self.patient_id = patient_id
        self.session_id = session_id
        self.total_expected = total_expected
        self.sequence = sequence

        self.processor = AudioProcessor(filepath)

        self.transcription = ""
        self.spoken_numbers = []
        self.detected_count = 0
        self.peak_force_db = None
        self.sequence_result = {}

    # --------------------------------------------------
    # Speech To Text
    # --------------------------------------------------

    def transcribe(self):

        model = get_whisper_model()

        segments, _ = model.transcribe(self.filepath)

        text = ""

        for segment in segments:
            text += segment.text + " "

        self.transcription = text.lower()

        return self.transcription

    # --------------------------------------------------
    # Words/Digits To Numbers
    # --------------------------------------------------

    def extract_numbers(self):

        words = re.findall(r'\w+', self.transcription)

        numbers = []

        for word in words:

            if word in self.NUMBER_MAP:
                numbers.append(self.NUMBER_MAP[word])

            elif word.isdigit():
                numbers.append(int(word))

        self.spoken_numbers = numbers
        self.detected_count = len(numbers)

        return numbers

    # --------------------------------------------------
    # Parse Expected Sequence
    # --------------------------------------------------

    def parse_expected_sequence(self):
        try:
            parsed = json.loads(self.sequence)

            if isinstance(parsed, list):
                return [int(x) for x in parsed]

        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback: comma or whitespace separated string
        raw = re.split(r'[,\s]+', self.sequence.strip())

        return [int(x) for x in raw if x.strip().lstrip('-').isdigit()]

    # --------------------------------------------------
    # Compare Spoken Numbers Against Expected Sequence
    # --------------------------------------------------

    def check_sequence(self) -> dict:

        expected = self.parse_expected_sequence()

        correct_count = 0

        for i, num in enumerate(self.spoken_numbers):
            if i < len(expected) and num == expected[i]:
                correct_count += 1

        self.sequence_result = {
            "expected_sequence": expected,
            "spoken_numbers": self.spoken_numbers,
            "correct_count": correct_count
        }

        return self.sequence_result

    # --------------------------------------------------
    # Final Analysis
    # --------------------------------------------------

    def analyze(self) -> dict:

        self.processor.load()
        self.processor.validate()
        self.peak_force_db = self.processor.peak_db()

        self.transcribe()
        self.extract_numbers()
        self.check_sequence()

        return self.build_response()

    def build_response(self) -> dict:
        return {
            "success": True,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "assessment_type": "counting",
            "result_data": {
                "detected_count": self.detected_count,
                "total_expected": self.total_expected,
                "correct_count": self.sequence_result["correct_count"],
                "peak_force_db": self.peak_force_db
            }
        }