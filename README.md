# PaddleOCR Demo

Minimal project to test PaddleOCR.

## Setup

Already set up in this directory (Python 3.11 venv):

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source venv/bin/activate
python demo.py                     # uses images/sample.png
python demo.py path/to/other.png   # or any image of your choice
```

First run downloads PaddleOCR's detection/recognition/angle-classifier
models (a few hundred MB) into `~/.paddleocr/`. Subsequent runs are fast.

Output:
- Console: detected text lines with confidence scores.
- `output/annotated.png`: input image with red boxes around detected text.

## Web UI (Gradio)

```bash
source venv/bin/activate
python app.py
```

Opens a local Gradio app (default: http://127.0.0.1:7860) to drag/drop an image and
see boxes + recognized text in the browser.

## HTTP microservice (for integrating with other apps, e.g. Laravel)

```bash
source venv/bin/activate
uvicorn service:app --host 127.0.0.1 --port 8001
```

- `GET /health` — liveness check.
- `POST /ocr` — multipart upload, field name `document` (image or PDF). Returns:
  ```json
  { "blocks": [ { "text": "...", "confidence": 0.99, "box": [[x,y],[x,y],[x,y],[x,y]] } ] }
  ```

This is what `otoz-ops` (the Laravel app) calls via `PADDLEOCR_ENDPOINT` — start this
service before using the "PaddleOCR Extract" button in that app.

## Files

- `demo.py` — CLI script: runs OCR on an image and prints/saves results.
- `app.py` — Gradio web UI for interactive testing.
- `service.py` — FastAPI microservice for integrating with other apps over HTTP.
- `images/sample.png` — generated test image with simple text.
- `output/` — annotated results get saved here (from `demo.py`).
