"""
Web UI for testing PaddleOCR.

Usage:
    python app.py

Opens a local Gradio app (default: http://127.0.0.1:7860) where you can
upload/drag-drop an image and see detected text + confidence + boxes.
"""
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR

print("Loading PaddleOCR model (first run may download weights)...")
ocr = PaddleOCR(use_textline_orientation=True, lang="en")
print("Model loaded.")

print("testing PaddleOCR with a sample image...")

def run_ocr(image):
    if image is None:
        return None, "No image provided."

    # gradio gives a numpy array (RGB); PaddleOCR accepts numpy arrays directly.
    results = ocr.predict(image)

    if not results or len(results[0]["rec_texts"]) == 0:
        return Image.fromarray(image), "No text detected."

    res = results[0]
    texts = res["rec_texts"]
    scores = res["rec_scores"]
    boxes = res["rec_polys"]

    # Draw boxes on a copy of the image.
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        points = [tuple(p) for p in box]
        draw.polygon(points, outline="red", width=2)

    # Build a readable text report.
    lines = [f"[{score:.3f}]  {text}" for text, score in zip(texts, scores)]
    report = f"Detected {len(texts)} line(s):\n\n" + "\n".join(lines)

    return img, report


demo = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="numpy", label="Upload or drag an image"),
    outputs=[
        gr.Image(type="pil", label="Detected text boxes"),
        gr.Textbox(label="Recognized text", lines=15),
    ],
    title="PaddleOCR Demo",
    description="Upload an image to test PaddleOCR's text detection + recognition.",
    examples=["images/sample.png"],
)

if __name__ == "__main__":
    demo.launch()
