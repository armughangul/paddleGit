"""
PaddleOCR HTTP microservice.

Wraps the PaddleOCR pipeline behind a small FastAPI app so it can be called
from other stacks (e.g. a Laravel app) over HTTP.

Usage:
    source venv/bin/activate
    uvicorn service:app --host 127.0.0.1 --port 8001

Endpoints:
    GET  /health          -> liveness check
    POST /ocr              -> multipart upload, field name "document"
                               returns {"blocks": [{"text", "confidence", "box"}]}
"""
import io
import os
import subprocess
import tempfile
from pathlib import Path

# PaddlePaddle's CPU backend (MKLDNN/OpenMP) sizes its thread pool off the
# *host* machine's core count, not any container CPU limit — on a cloud box
# with far more cores than the container actually gets, this blows up RAM
# far past what the workload needs (observed: 12GB+ on Railway vs ~3GB
# locally on a modest Mac). Pin it explicitly. setdefault() so an operator
# can still override via a real env var without editing code.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("FLAGS_cpu_num_threads", "2")

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from paddleocr import PaddleOCR

app = FastAPI(title="PaddleOCR Service")

print("Loading PaddleOCR model...")
# Auction-sheet/vehicle-document scans are consistently upright and flat, so
# the page-level orientation classifier and perspective-unwarping model are
# dead weight here — each is a separate loaded model with its own inference
# workspace, not just the few MB its weights take on disk. Cutting these two
# (of 5 total) measurably reduces both load time and steady-state RAM.
# use_textline_orientation stays on since individual text lines within an
# otherwise-upright scan can still run sideways/rotated on some documents.
# enable_mkldnn=False: MKLDNN/oneDNN's own thread pool and primitive cache is
# a second, independent contributor to the same host-core-count memory blowup
# OMP_NUM_THREADS/FLAGS_cpu_num_threads address above — belt and suspenders.
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="en",
    enable_mkldnn=False,
)
print("Model loaded. Ready to serve requests.")


@app.get("/health")
def health():
    return {"status": "ok"}


def _first_page_from_pdf(raw: bytes) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "input.pdf"
        pdf_path.write_bytes(raw)
        output_prefix = Path(tmp_dir) / "page"

        result = subprocess.run(
            ["pdftoppm", "-png", "-r", "200", "-f", "1", "-l", "1", str(pdf_path), str(output_prefix)],
            capture_output=True,
        )

        if result.returncode != 0:
            raise HTTPException(status_code=400, detail="Failed to convert PDF to an image.")

        pages = sorted(Path(tmp_dir).glob("page*.png"))
        if not pages:
            raise HTTPException(status_code=400, detail="PDF conversion produced no image.")

        return Image.open(pages[0]).convert("RGB")


@app.post("/ocr")
async def run_ocr(document: UploadFile = File(...)):
    raw = await document.read()
    filename = (document.filename or "").lower()

    try:
        if filename.endswith(".pdf") or document.content_type in ("application/pdf", "application/x-pdf"):
            image = _first_page_from_pdf(raw)
        else:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image or PDF.")

    image_array = np.array(image)
    results = ocr.predict(image_array)

    if not results or len(results[0]["rec_texts"]) == 0:
        return {"blocks": []}

    res = results[0]
    texts = res["rec_texts"]
    scores = res["rec_scores"]
    boxes = res["rec_polys"]

    blocks = [
        {
            "text": text,
            "confidence": round(float(score), 4),
            "box": [[float(x), float(y)] for x, y in box],
        }
        for text, score, box in zip(texts, scores, boxes)
    ]

    return {"blocks": blocks}
