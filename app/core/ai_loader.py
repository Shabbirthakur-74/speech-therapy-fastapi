import mediapipe as mp


class AIModels:

    def __init__(self):

        self.face_landmarker = None
        self.hand_landmarker = None

    def load(self):

        print("Loading MediaPipe...")

        self.face_landmarker = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        self.hand_landmarker = mp.solutions.hands.Hands(
            max_num_hands=2
        )

        print("MediaPipe Loaded")


ai_models = AIModels()