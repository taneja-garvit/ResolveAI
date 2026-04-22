import os
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.rag_service import process_document

router = APIRouter()

UPLOAD_DIR = "data/docs"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def sanitize_upload_name(filename: str) -> str:
    original_name = Path(filename or "").name
    suffix = Path(original_name).suffix.lower()
    stem = SAFE_FILENAME_RE.sub("_", Path(original_name).stem).strip("._")

    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    if not stem:
        raise HTTPException(status_code=400, detail="Uploaded file name is invalid.")

    return f"{uuid4().hex}_{stem}{suffix}"


@router.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    safe_filename = sanitize_upload_name(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        with open(file_path, "wb") as f:
            f.write(contents)

        process_document(file_path)
    except ValueError as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        ) from exc
    finally:
        await file.close()

    return {"message": "Document uploaded and processed", "filename": safe_filename}