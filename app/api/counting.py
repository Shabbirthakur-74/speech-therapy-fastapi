import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.audio.common import AudioProcessor
from app.services.audio.counting import CountingAnalyzer
from app.services.laravel_client import send_result_to_laravel, LaravelClientError

router = APIRouter(
    prefix="/api/counting",
    tags=["Counting"]
)

@router.post("/analyze")
async def analyze_counting(
    patient_id: int = Form(...),
    session_id: str = Form(...),
    total_expected: int = Form(
        ..., description="How many numbers the patient was asked to count, e.g. 20"
    ),
    sequence: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a recording of the patient counting. Detects how many
    numbers were spoken out of the expected total, measures peak
    vocal force, and stores the result via Laravel.
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

    if total_expected <= 0:
        raise HTTPException(
            status_code=400,
            detail="total_expected must be greater than 0"
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:
        temp.write(await file.read())
        filepath = temp.name

    try:
        analyzer = CountingAnalyzer(
            filepath=filepath,
            patient_id=patient_id,
            session_id=session_id,
            total_expected=total_expected,
            sequence=sequence
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
        "message": "Counting exercise recorded successfully."
    }