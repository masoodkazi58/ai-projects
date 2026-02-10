from fastapi import FastAPI,UploadFile,File,Form,Request,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from modules import model_question
from config import PDF_DIR,DATA_DIR
from extractor import extract_pdf_text_to_txt
from embedding import build_embeddings_for_txt

app = FastAPI(title="Ragapp")

class Userinput(BaseModel):
    text : str
    doc_name: str

@app.post("/ask")
async def ask(payload:Userinput):
    answer = await model_question(question=payload.text,doc_name=payload.doc_name)
   
    return {"answer":answer}

@app.post("/uploads")
async def upload_pdfs(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only pdf or txt files are allowed"
        )

    # -------- TXT FILE --------
    if file.filename.lower().endswith(".txt"):
        file_path = DATA_DIR / file.filename

        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        embeddings_path = build_embeddings_for_txt(
            txt_file_path=file_path
        )

        text_file_path = file_path

    # -------- PDF FILE --------
    else:
        file_path = PDF_DIR / file.filename

        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        # ❗ extract_pdf_text_to_txt is SYNC → do NOT await
        text_file_path = extract_pdf_text_to_txt(
            pdf_path=file_path,
            output_dir=DATA_DIR
        )

        # ❗ embed the TXT, not the PDF
        embeddings_path = build_embeddings_for_txt(
            txt_file_path=text_file_path
        )

    return {
        "filename": file.filename,
        "saved_to": str(text_file_path),
        "vectorstore_path": str(embeddings_path)
    }
