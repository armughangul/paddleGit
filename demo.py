"""
Simple PaddleOCR demo.

Usage:
    python demo.py [path/to/image.png]

If no path is given, uses images/sample.png (a generated test image).
Prints detected text lines + confidence, and saves an annotated image
to output/annotated.png.
"""
import sys
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images/sample.png")
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"Loading PaddleOCR (this may download models on first run)...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)

    print(f"Running OCR on: {image_path}")
    results = ocr.predict(str(image_path))

    if not results:
        print("No text detected.")
        return

    res = results[0]
    texts = res["rec_texts"]
    scores = res["rec_scores"]
    boxes = res["rec_polys"]

    print(f"\nDetected {len(texts)} text line(s):\n")
    for text, score in zip(texts, scores):
        print(f"  [{score:.3f}] {text}")

    # Draw boxes + text on a copy of the image for visual confirmation.
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        points = [tuple(p) for p in box]
        draw.polygon(points, outline="red", width=2)

    output_path = Path("output/annotated.png")
    output_path.parent.mkdir(exist_ok=True)
    img.save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
