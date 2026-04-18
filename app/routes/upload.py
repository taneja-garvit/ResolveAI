from fastapi import APIRouter, UploadFile, File
import os
from app.services.rag_service import process_document

router = APIRouter()

UPLOAD_DIR = "data/docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # process for RAG
    process_document(file_path)

    return {"message": "Document uploaded and processed"}