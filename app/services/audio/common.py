from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import os

import librosa
import numpy as np
from pydub import AudioSegment


class AudioProcessor:
    """
    Shared audio processing class for all speech therapy exercises.

    Responsibilities:
    - Load audio
    - Normalize formats that librosa/soundfile cannot read directly
    - Validate audio
    - Extract common audio metrics
    - Extract reusable speech features
    """

    SUPPORTED_EXTENSIONS = {
        ".wav",
        ".mp3",
        ".ogg",
        ".flac",
        ".m4a",
        ".webm"
    }

    def __init__(self, file_path: str):

        self.file_path = Path(file_path)

        self.audio = None
        self.sample_rate = None

        # Path to temporary converted WAV, if one was created.
        self._converted_file = None

    # -------------------------------------------------------
    # Audio Format Normalization
    # -------------------------------------------------------

    def _convert_to_wav(self) -> Path:
        """
        Convert the input audio to WAV using FFmpeg/pydub.

        This is used for formats such as M4A and WebM that may
        not be directly readable by soundfile/libsndfile.
        """

        print(
            f"Converting audio to WAV: {self.file_path}"
        )

        try:
            audio = AudioSegment.from_file(
                str(self.file_path)
            )

        except Exception as exc:
            raise ValueError(
                f"Could not decode audio file "
                f"'{self.file_path.name}': {exc}"
            ) from exc

        # Create a temporary WAV file.
        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        wav_path = Path(temp.name)

        temp.close()

        try:
            audio.export(
                str(wav_path),
                format="wav"
            )

        except Exception as exc:

            # Clean up if conversion failed.
            if wav_path.exists():
                try:
                    wav_path.unlink()
                except OSError:
                    pass

            raise ValueError(
                f"Could not convert audio to WAV: {exc}"
            ) from exc

        self._converted_file = wav_path

        print(
            f"Audio converted successfully: {wav_path}"
        )

        return wav_path

    def _get_load_path(self) -> Path:
        """
        Determine which file should be passed to librosa.

        WAV files are loaded directly.

        Other formats are converted to WAV first so that
        librosa/soundfile have a reliable input format.
        """

        extension = self.file_path.suffix.lower()

        if extension == ".wav":
            return self.file_path

        return self._convert_to_wav()

    # -------------------------------------------------------
    # Load Audio
    # -------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.

        Non-WAV formats are normalized to WAV using FFmpeg
        before being loaded by librosa.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {self.file_path}"
            )

        extension = self.file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {extension}"
            )

        load_path = self._get_load_path()

        print(
            f"Loading audio with librosa: {load_path}"
        )

        try:
            self.audio, self.sample_rate = librosa.load(
                str(load_path),
                sr=None,
                mono=True
            )

        except Exception:
            # If librosa fails, clean up converted file.
            self.cleanup()
            raise

        return self.audio, self.sample_rate

    # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------

    def cleanup(self):
        """
        Remove temporary converted audio file.
        """

        if self._converted_file is not None:

            try:

                if self._converted_file.exists():
                    self._converted_file.unlink()

                    print(
                        f"Deleted converted audio: "
                        f"{self._converted_file}"
                    )

            except OSError as exc:

                print(
                    f"Warning: could not delete converted "
                    f"audio file: {exc}"
                )

            finally:
                self._converted_file = None

    def __del__(self):
        """
        Best-effort cleanup when the processor is destroyed.
        """

        try:
            self.cleanup()
        except Exception:
            pass

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    def validate(self) -> bool:

        if self.audio is None:
            raise ValueError("Audio not loaded.")

        if len(self.audio) == 0:
            raise ValueError("Empty audio file.")

        return True

    # -------------------------------------------------------
    # Basic Metrics
    # -------------------------------------------------------

    def duration(self) -> float:

        return float(
            librosa.get_duration(
                y=self.audio,
                sr=self.sample_rate
            )
        )

    def rms(self) -> float:

        return float(
            np.sqrt(
                np.mean(self.audio ** 2)
            )
        )

    def peak(self) -> float:
        """
        Peak amplitude (linear, 0.0-1.0 range).
        """

        return float(
            np.max(
                np.abs(self.audio)
            )
        )

    def peak_db(self, floor_db: float = -100.0) -> float:
        """
        Peak intensity in dBFS (decibels relative to full scale).

        NOTE: This is dBFS, the standard peak-level metric for
        digital audio. It is NOT true dBm - dBm is a calibrated
        electrical power unit relative to 1 milliwatt and cannot
        be derived from a raw audio file alone.
        """

        peak_amp = self.peak()

        if peak_amp <= 0:
            return floor_db

        db = 20 * np.log10(peak_amp)

        return round(max(db, floor_db), 2)

    def rms_db(self, floor_db: float = -100.0) -> float:
        """
        Average intensity in dBFS, derived from RMS energy.
        """

        rms_amp = self.rms()

        if rms_amp <= 0:
            return floor_db

        db = 20 * np.log10(rms_amp)

        return round(max(db, floor_db), 2)

    def peak_rms_db(
        self,
        frame_length: int = 1024,
        hop_length: int = 512,
        floor_db: float = -100.0
    ) -> float:
        """
        Peak short-term RMS pressure in dBFS.
        """

        rms_frames = librosa.feature.rms(
            y=self.audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        peak_rms = float(np.max(rms_frames))

        if peak_rms <= 0:
            return floor_db

        db = 20 * np.log10(peak_rms)

        return round(max(db, floor_db), 2)

    def silence_ratio(
        self,
        threshold: float = 0.01
    ) -> float:

        silent = np.abs(self.audio) < threshold

        return float(np.mean(silent))

    # -------------------------------------------------------
    # Pitch
    # -------------------------------------------------------

    def pitch(self) -> float:
        """
        Estimate average pitch (mean F0 in Hz)
        using librosa.pyin().
        """

        f0, _, _ = librosa.pyin(
            self.audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )

        if np.all(np.isnan(f0)):
            return 0.0

        return float(np.nanmean(f0))

    # -------------------------------------------------------
    # Spectral Features
    # -------------------------------------------------------

    def spectral_features(self) -> Dict:

        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=self.audio,
                    sr=self.sample_rate
                )
            )
        )

        bandwidth = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    y=self.audio,
                    sr=self.sample_rate
                )
            )
        )

        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=self.audio,
                    sr=self.sample_rate
                )
            )
        )

        zcr = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    self.audio
                )
            )
        )

        return {
            "spectral_centroid": round(centroid, 2),
            "spectral_bandwidth": round(bandwidth, 2),
            "spectral_rolloff": round(rolloff, 2),
            "zero_crossing_rate": round(zcr, 4)
        }

    # -------------------------------------------------------
    # MFCC
    # -------------------------------------------------------

    def mfcc(self) -> List[float]:

        mfcc = librosa.feature.mfcc(
            y=self.audio,
            sr=self.sample_rate,
            n_mfcc=13
        )

        return [
            round(float(value), 2)
            for value in np.mean(mfcc, axis=1)
        ]

    # -------------------------------------------------------
    # Full Feature Extraction
    # -------------------------------------------------------

    def extract_features(self) -> Dict:

        self.validate()

        return {
            "duration": round(self.duration(), 2),
            "sample_rate": self.sample_rate,
            "rms": round(self.rms(), 4),
            "peak": round(self.peak(), 4),
            "peak_db": self.peak_db(),
            "silence_ratio": round(
                self.silence_ratio(),
                4
            ),
            "pitch": round(self.pitch(), 2),
            **self.spectral_features(),
            "mfcc": self.mfcc()
        }

    # -------------------------------------------------------
    # Analyze
    # -------------------------------------------------------

    def analyze(self) -> Dict:

        try:

            self.load()

            return self.extract_features()

        finally:

            self.cleanup()