"""
This module provides a function to process a single image
"""
from __future__ import annotations
import json

from PIL import Image

from lib.tesseract import get_boxes
from lib.gcloud import ocr


def parse_png(img: Image.Image) -> str:
    _, boxes_info = get_boxes(img, preprocess=True)

    boxes = [img.crop(rect.to_pillow()) for (rect, _, _) in boxes_info]

    # set format from parent
    for box in boxes:
        box.format = img.format

    print(f'Running Cloud Vision OCR on {len(boxes)} images')
    if len(boxes) > 100 and input('Continue? [y/n]').strip() == 'y':
        return ''
    annotations_raw = map(ocr.annotate, boxes)

    annotations_json = map(json.loads, annotations_raw)
    annotations = map(ocr.Annotation, annotations_json)
    annotations = filter(lambda x: x is not None and x.full_text, annotations)
    annotations = filter(lambda x: len(x.full_text) > 30, annotations)
    return '\n'.join(box.full_text for box in annotations)
