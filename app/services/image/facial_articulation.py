import math
from app.services.image.landmarks import get_face_landmark_detector
class FacialArticulationAnalyzer:
    """
    Facial articulation exercise (wide_smile or o_shape).

    Uses the shared FaceLandmarkDetector singleton rather than
    creating a new MediaPipe FaceMesh instance per request.
    """
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_EYE = 33
    RIGHT_EYE = 263
    VALID_EXERCISES = {"wide_smile", "o_shape"}

    def __init__(
        self,
        image_path: str,
        patient_id: int,
        session_id: str,
        exercise: str
    ):

        self.image_path = image_path
        self.patient_id = patient_id
        self.session_id = session_id
        self.exercise = exercise.lower()

        if self.exercise not in self.VALID_EXERCISES:
            raise ValueError(
                f"Invalid exercise '{exercise}'. Must be one of: "
                f"{', '.join(sorted(self.VALID_EXERCISES))}"
            )
        self.detector = get_face_landmark_detector()
        self.points = None
        self.score = 0

    # --------------------------------------------------
    # Distance
    # --------------------------------------------------

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # --------------------------------------------------
    # Face Normalization
    # --------------------------------------------------

    def face_scale(self):
        return self.distance(
            self.points[self.LEFT_EYE],
            self.points[self.RIGHT_EYE]
        )
    # --------------------------------------------------
    # Mouth Width / Height
    # --------------------------------------------------

    def mouth_width(self):
        return self.distance(
            self.points[self.LEFT_MOUTH],
            self.points[self.RIGHT_MOUTH]
        )

    def mouth_height(self):
        return self.distance(
            self.points[self.UPPER_LIP],
            self.points[self.LOWER_LIP]
        )
    # --------------------------------------------------
    # Wide Smile
    # --------------------------------------------------

    def analyze_wide_smile(self):
        ratio = self.mouth_width() / self.face_scale()
        if ratio >= 0.65:
            score = 100
        elif ratio >= 0.60:
            score = 90
        elif ratio >= 0.55:
            score = 80
        elif ratio >= 0.50:
            score = 70
        else:
            score = 60
        self.score = score

    # --------------------------------------------------
    # O Shape
    # --------------------------------------------------

    def analyze_o_shape(self):
        ratio = self.mouth_height() / self.mouth_width()
        if ratio >= 0.60:
            score = 100
        elif ratio >= 0.50:
            score = 90
        elif ratio >= 0.40:
            score = 80
        elif ratio >= 0.30:
            score = 70
        else:
            score = 60
        self.score = score

    # --------------------------------------------------
    # Main Analysis
    # --------------------------------------------------

    def analyze(self) -> dict:
        self.points = self.detector.detect(self.image_path)
        if self.points is None:
            raise ValueError("Face not detected in the image.")
        if self.exercise == "wide_smile":
            self.analyze_wide_smile()
        else:
            self.analyze_o_shape()
        return self.build_response()

    def build_response(self) -> dict:
        return {
            "success": True,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "assessment_type": "facial_articulation",
            "result_data": {
            "exercise": self.exercise,
            "articulation_score": self.score
            }
        }