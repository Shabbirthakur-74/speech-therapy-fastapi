import os
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.audio.common import AudioProcessor
from app.services.audio.resonatory_control import ResonatoryControlAnalyzer
from app.services.laravel_client import send_result_to_laravel, LaravelClientError

router = APIRouter(
    prefix="/api/resonatory-control",
    tags=["Resonatory Control"]
)


@router.post("/analyze")
async def analyze_resonatory_control(
    patient_id: int = Form(...),
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a resonatory control recording, measure peak pressure,
    and store the result via Laravel.
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix not in AudioProcessor.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format '{suffix}'. "
                f"Supported formats: "
                f"{', '.join(sorted(AudioProcessor.SUPPORTED_EXTENSIONS))}"
            )
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:
        temp.write(await file.read())
        filepath = temp.name

    try:
        analyzer = ResonatoryControlAnalyzer(
            filepath=filepath,
            patient_id=patient_id,
            session_id=session_id,
        )

        result = analyzer.analyze()

    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file could not be read."
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Audio analysis failed. Please try recording again."
        )

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

       # Forward result to Laravel for storage
    try:
        await send_result_to_laravel(
            endpoint="assessment-results",
            payload=result
        )
    except LaravelClientError as exc:
        if exc.status_code is None:
            # Laravel unreachable - this really is a server-side failure
            raise HTTPException(
                status_code=502,
                detail="Analysis succeeded but saving the result failed. Please try again."
            )
        # Laravel responded with a real business error (already completed,
        # session expired, invalid session, etc.) - pass it through as-is
        # so the phone can distinguish these cases.
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail
        )

    return {
        "success": True,
        "message": "Resonatory control exercise recorded successfully."
    }