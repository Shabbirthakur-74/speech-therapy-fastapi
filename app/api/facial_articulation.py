import os
import tempfile
import traceback

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.image.facial_articulation import FacialArticulationAnalyzer
from app.services.laravel_client import (
    send_result_to_laravel,
    LaravelClientError,
)


router = APIRouter(
    prefix="/api/facial-articulation",
    tags=["Facial Articulation"],
)


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


@router.post("/analyze")
async def analyze_facial_articulation(
    patient_id: int = Form(...),
    session_id: str = Form(...),
    exercise: str = Form(
        ...,
        description="'wide_smile' or 'o_shape'",
    ),
    file: UploadFile = File(...),
):
    """
    Upload a facial photo for the wide_smile or o_shape exercise,
    analyze facial articulation, and store the result via Laravel.
    """

    filepath = None

    try:
        # ---------------------------------------------------------
        # 1. Validate uploaded file
        # ---------------------------------------------------------

        print("\n========== FACIAL ARTICULATION REQUEST ==========")
        print("Patient ID:", patient_id)
        print("Session ID:", session_id)
        print("Exercise:", exercise)
        print("Filename:", file.filename)
        print("Content type:", file.content_type)

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file uploaded",
            )

        suffix = os.path.splitext(file.filename)[1].lower()

        print("File extension:", suffix)

        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported image format '{suffix}'. "
                    f"Supported formats: "
                    f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
                ),
            )

        # ---------------------------------------------------------
        # 2. Save uploaded file temporarily
        # ---------------------------------------------------------

        print("Saving uploaded image...")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:

            file_contents = await file.read()

            print("Uploaded bytes:", len(file_contents))

            temp.write(file_contents)

            filepath = temp.name

        print("Temporary image:", filepath)

        # ---------------------------------------------------------
        # 3. Run facial articulation analysis
        # ---------------------------------------------------------

        print("Creating FacialArticulationAnalyzer...")

        analyzer = FacialArticulationAnalyzer(
            image_path=filepath,
            patient_id=patient_id,
            session_id=session_id,
            exercise=exercise,
        )

        print("Analyzer created successfully.")

        print("Starting image analysis...")

        result = analyzer.analyze()

        print("Image analysis completed.")
        print("Analysis result:", result)

        # ---------------------------------------------------------
        # 4. Send result to Laravel
        # ---------------------------------------------------------

        print("Sending result to Laravel...")

        try:
            await send_result_to_laravel(
                endpoint="assessment-results",
                payload=result,
            )

            print("Laravel save completed successfully.")

        except LaravelClientError as exc:

            print("LaravelClientError occurred.")
            print("Status code:", exc.status_code)
            print("Detail:", exc.detail)

            if exc.status_code is None:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Analysis succeeded but saving the result "
                        "failed. Please try again."
                    ),
                )

            # Laravel returned a business error such as:
            # - already completed
            # - session expired
            # - invalid session
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
            )

        except Exception as exc:

            print("Unexpected Laravel error:")
            traceback.print_exc()

            raise HTTPException(
                status_code=502,
                detail=(
                    "Analysis succeeded but saving the result "
                    "failed. Please try again."
                ),
            )

        # ---------------------------------------------------------
        # 5. Success
        # ---------------------------------------------------------

        print("========== FACIAL ARTICULATION SUCCESS ==========\n")

        return {
            "success": True,
            "message": "Facial articulation exercise recorded successfully.",
        }

    except HTTPException:
        # Let our intentional HTTP errors pass through unchanged.
        raise

    except ValueError as exc:

        print("ValueError during facial articulation analysis:")
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        # ---------------------------------------------------------
        # This is the important part for debugging.
        # ---------------------------------------------------------

        print("\n========== FACIAL ARTICULATION ERROR ==========")
        print("Exception type:", type(exc).__name__)
        print("Exception message:", str(exc))
        print("Full traceback:")

        traceback.print_exc()

        print("===============================================\n")

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(exc)}",
        )

    finally:

        # ---------------------------------------------------------
        # 6. Always delete temporary image
        # ---------------------------------------------------------

        if filepath and os.path.exists(filepath):

            try:
                os.remove(filepath)
                print("Deleted temporary image:", filepath)

            except Exception as exc:
                print(
                    "Warning: Could not delete temporary image:",
                    filepath,
                )
                print("Delete error:", exc)