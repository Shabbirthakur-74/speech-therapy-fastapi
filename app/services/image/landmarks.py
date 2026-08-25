import threading
from functools import lru_cache

import cv2
import mediapipe as mp


class FaceLandmarkDetector:
    """
    Wraps MediaPipe FaceMesh. Instantiated once via
    get_face_landmark_detector() below and reused across requests -
    creating a new instance per request reloads the model from disk
    every time.

    .detect() is guarded by a lock since FaceMesh.process() is not
    guaranteed safe for concurrent calls on the same instance.
    """

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self._lock = threading.Lock()

    def detect(self, image_path: str):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Unable to load image.")

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        h, w, _ = image.shape

        with self._lock:
            results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        points = [
            (int(landmark.x * w), int(landmark.y * h))
            for landmark in landmarks
        ]

        return points


@lru_cache(maxsize=1)
def get_face_landmark_detector() -> FaceLandmarkDetector:
    """
    Loads the MediaPipe FaceMesh model once per process and reuses
    it across every facial exercise request.
    """

    return FaceLandmarkDetector()