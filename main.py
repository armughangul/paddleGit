import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from google.cloud import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "awsummit-f2ce2-5cbe291d4779.json")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", CREDENTIALS_PATH)

app = FastAPI(title="Google OCR API")

client = vision.ImageAnnotatorClient()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["ja", "en"])
    response = client.document_text_detection(image=image, image_context=image_context)

    if response.error.message:
        raise HTTPException(status_code=502, detail=f"Vision API error: {response.error.message}")

    extracted_text = response.full_text_annotation.text if response.full_text_annotation else ""

    return {"filename": file.filename, "text": extracted_text}
