from __future__ import annotations

import scipy.stats
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw
import cv2

from lib.rectangle import Rectangle


def pillow_to_opencv(img: Image.Image) -> np.array:
    """PIL RGB image to BGR np.array"""
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # cv2 uses BGR, not RGB


def opencv_to_pillow(arr: np.array) -> Image.Image:
    """BGR np.array to RGB PIL image"""
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def binarize(img: Image.Image) -> Image.Image:
    """
    binarize the image, apply basic image morphology (dilation, opening etc.)
    """

    img = pillow_to_opencv(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply morphological open
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # find the Otsu threshold, but don't binarize the image yet
    # otsu_th, _ = cv2.threshold(
    #     img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # use the Otsu threshold to find thresholds for the Canny edge detection alg
    # canny_low = otsu_th / 2
    # canny_high = otsu_th
    # img = cv2.Canny(img, canny_low, canny_high)
    # img = ~img

    # binarize with Otsu's method
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # apply erosion after binarization
    # img = cv2.erode(img, kernel)

    img = opencv_to_pillow(img)
    return img


def split_vertical(img: Image.Image) -> list[int]:
    height = img.size[1]
    # 1 if there's something, 0 if pixels are empty
    img = ~np.array(binarize(img).convert('1'))

    fillness = img.sum(axis=0) / height
    return list(np.where(fillness < 0.05)[0])


def get_expected_letter_size(img: Image.Image) -> tuple[int, int]:
    cv_img = np.array(img.convert('L'))

    contours, _ = cv2.findContours(
        cv_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    # draw = ImageDraw.Draw(img)
    widths, heights = [], []
    for contour in contours:
        # left, top, width, height = cv2.boundingRect(contour)
        # letter_rect = Rectangle(left, top, width, height).shift(anchor)
        # draw.rectangle(letter_rect.to_xy(), outline='green')
        _, _, width, height = cv2.boundingRect(contour)
        widths.append(width)
        heights.append(height)

    def stat(arr: list) -> int:
        quant = scipy.stats.mstats.mquantiles(arr, axis=None)[2]
        # interquartile mean
        trim_mean = scipy.stats.trim_mean(arr, proportiontocut=0.25)
        return int(max(quant, trim_mean))

    return stat(widths), stat(heights)


def has_gutter(img: Image.Image, mean_w: int) -> bool:
    lonely = split_vertical(img)
    assert sorted(lonely) == lonely
    assert len(set(lonely)) == len(lonely)

    if lonely:
        gutter_start_ind = 0
        last_x = lonely[0]

        to_remove = []
        for i, x in enumerate(lonely[1:]):
            i += 1  # because lonely[1:]
            if x > last_x + 1:
                # gutter ended. check it's width and start a new one
                # if it's narrow, drop it
                if last_x - lonely[gutter_start_ind] < mean_w * 1.5:
                    to_remove.extend(lonely[gutter_start_ind: i])
                else:
                    return True
                gutter_start_ind = i
            last_x = x
    return False


def get_boxes(
    img: Image.Image,
    preprocess: bool,
    _recursive: bool = False
) -> tuple[Image.Image, list[tuple[Rectangle, str, tuple[int, int]]]]:
    """
    Returns [preprocessed] source img with rectangles drawn and
    the rectangles themselves
    """

    if not _recursive:
        # don't modify the source image in-place
        img = img.copy()

        # binarization
        # we want dark text on light background
        if preprocess:
            img = binarize(img)

    # good thing: all our pages have horizontal lines, are not skewed and the
    # overall quality of the images is quite good.
    annotations = pytesseract.image_to_data(
        img,
        output_type=Output.DICT,
        lang='pol',
        # oem=1: use LSTM engine only
        # psm=1: automatic page segmentation + osd
        config='--psm 1 --oem 1'
    )
    # print(annotations.keys())

    boxes = []
    for i, _ in enumerate(annotations['level']):
        # levels:
        # 3 - paragraph
        # 4 - line
        # 5 - word
        # 6 - letter
        if annotations['level'][i] != 3:
            continue

        rect = Rectangle.from_tesseract(annotations, i)
        anchor = (rect.left, rect.top)

        if not rect.get_area():
            continue

        inner_img = img.crop(rect.to_pillow())

        # drop boxes with no or little text
        txt = pytesseract.image_to_string(
            inner_img,
            lang='pol',
            config='--psm 1 --oem 1'
        )

        if len(txt) < 4:
            continue

        letter_size = mean_w, mean_h = get_expected_letter_size(inner_img)
        inner_boxes = [(rect, 'red', letter_size)]

        if not _recursive:
            if has_gutter(inner_img, mean_w):
                _, inner_boxes = get_boxes(inner_img, False, True)
                inner_rects = [rect for (rect, _, _) in inner_boxes]
                if len(inner_rects) > 1:
                    inner_boxes = [
                        (
                            rect.shift(anchor),
                            'green',
                            get_expected_letter_size(inner_img.crop(rect.to_pillow()))
                        )
                        for rect in inner_rects
                    ]

            # mean_w, mean_h = get_expected_letter_size(inner_img)
            # letter_rect = Rectangle(anchor[0], anchor[1], mean_w, mean_h)

            # draw orange vertical strips where vertical blank detected
            # lonely = split_vertical(inner_img)
            # assert sorted(lonely) == lonely
            # assert len(set(lonely)) == len(lonely)

            # if lonely:
            #     gutter_start_ind = 0
            #     last_x = lonely[0]

            #     to_remove = []
            #     for i, x in enumerate(lonely[1:]):
            #         i += 1  # because lonely[1:]
            #         if x > last_x + 1:
            #             # gutter ended. check it's width and start a new one
            #             # if it's narrow, drop it
            #             if last_x - lonely[gutter_start_ind] < mean_w * 2.5:
            #                 to_remove.extend(lonely[gutter_start_ind: i])
            #             else:
            #                 print('GUTTTER DEEEETECTED!!')
            #                 print(lonely[gutter_start_ind], last_x, gutter_start_ind, i)
            #             gutter_start_ind = i
            #         last_x = x

            #     # check if we have a gutter unfinished and drop it if so
            #     if gutter_start_ind <= len(lonely) - 1:
            #         to_remove.extend(lonely[gutter_start_ind:])

            #     for r in to_remove:
            #         lonely.remove(r)

            # draw = ImageDraw.Draw(img)
            # for x in lonely:
            #     x += rect.left
            #     y0 = rect.top
            #     y1 = y0 + inner_img.size[1]
            #     draw.line((x, y0, x, y1), fill='orange', width=1)

            # for x in to_remove:
            #     x += rect.left
            #     y0 = rect.top
            #     y1 = y0 + inner_img.size[1]
            #     draw.line((x, y0, x, y1), fill='green', width=1)
            # _, _inner_boxes = get_boxes(inner_img, False, _recursive=True)

            # if len(_inner_boxes) > 1:
            #     _inner_boxes = [box.shift(anchor) for box in _inner_boxes]
            #     boxes.extend(_inner_boxes)
            #     for inner_box in _inner_boxes:
            #         draw = ImageDraw.Draw(img)
            #         draw.rectangle(inner_box.to_xy(), outline='red')
            #     continue

            # draw = ImageDraw.Draw(img)
            # draw.rectangle(rect.to_xy(), outline='red')
            # draw.rectangle(letter_rect.to_xy(), outline='blue')
        boxes.extend(inner_boxes)

    # orig_boxes = boxes[:]

    # remove overlapping rectangles
    # this doesn't work! some articles have a bounding box
    # around the whole thing
    # to_remove = []
    # for rect1 in boxes:
    #     for rect2 in boxes:
    #         if rect1 == rect2:
    #             continue

    #         if rect1.is_inside(rect2, margin=0.75):
    #             to_remove.append(rect1)

    # for r in to_remove:
    #     try:
    #         boxes.remove(r)
    #     except ValueError:
    #         pass

    # assert boxes type
    assert type(boxes) is list
    for b in boxes:
        assert type(b) is tuple
        assert len(b) == 3
        assert type(b[0]) is Rectangle
        assert type(b[1]) is str
        assert type(b[2]) is tuple
        assert len(b[2]) == 2
        assert type(b[2][0]) is int and type(b[2][1]) is int

    # detect headlines
    def median(arr: list) -> int:
        return scipy.stats.trim_mean(arr, proportiontocut=0.25)

    def rel_close(x, y, rel_tol) -> bool:
        return abs(1 - x / y) < rel_tol

    def font_close(size1, size2, rel_tol) -> bool:
        return rel_close(size1[0], size2[0], rel_tol) \
            and rel_close(size1[1], size2[1], rel_tol)

    median_fontsize = median([size[0] for (_, _, size) in boxes]), median([size[1] for (_, _, size) in boxes])

    for i, (rect, col, size) in enumerate(boxes):
        if font_close(size, median_fontsize, 0.9):
            boxes[i] = (rect, 'orange', size)

    def get_str(img: Image.Image) -> str:
        return pytesseract.image_to_string(
            img,
            lang='pol',
            config='--psm 1 --oem 1'
        )

    # remove too big rectangles (with multiple children containing most of text)
    # for box in boxes.copy():
    #     rect, color, font_size = box
    #     children = [box for box in boxes if box[0].is_inside(rect, margin=0.95) and box[0] != rect]

    #     img2 = img.copy()

    #     orig_text = get_str(img2.crop(rect.to_pillow()))

    #     draw = ImageDraw.Draw(img2)

    #     for child in children:
    #         draw.rectangle(child[0].to_xy(), fill='black')

    #     new_text = get_str(img2.crop(rect.to_pillow()))

    #     if len(new_text) < 0.15 * len(orig_text):
    #         boxes.remove(box)

    # merge vertically connected rectangles

    def merged(box1, box2):
        rect1, rect2 = box1[0], box2[0]
        return (
            Rectangle(
                min(rect1.left, rect2.left),
                min(rect1.top, rect2.top),
                max(rect1.width, rect2.width),
                abs(rect2.top + rect2.height - rect1.top)
            ),
            box1[1],
            box1[2]
        )

    def should_be_merged(
        box1: tuple[Rectangle, str, tuple[int, int]],
        box2: tuple[Rectangle, str, tuple[int, int]]
    ):
        if box1 is box2:
            return False

        if not font_close(box1[2], box2[2], 0.9):
            return False

        rect1, rect2 = box1[0], box2[0]
        if rect1 == rect2:
            return True

        # rect1 has to be above
        if rect1.top > rect2.top:
            return False

        if (
            rect1.left + rect1.width < rect2.left
            or rect2.left + rect2.width < rect1.left
        ):
            return False

        if not (
            rect1.left - box1[2][0] * 2 <= rect2.left
            and rect1.left + rect1.width + box1[2][0] * 2 >= rect2.left + rect2.width
        ):
            return False

        # it's just not close enough
        if rect1.top + rect1.height + box1[2][1] < rect2.top:
            return False

        return True

    done = False
    while not done:
        done = True

        for box1 in boxes:
            for box2 in boxes:
                if should_be_merged(box1, box2):
                    done = False
                    # carefully here; changing running iterator
                    boxes.remove(box1)
                    boxes.remove(box2)
                    boxes.append(merged(box1, box2))
                    break
            else:
                continue
            break

    # slabe to jest ;(
    # to_remove = []
    # to_add = []
    # for rect1 in boxes:
    #     for rect2 in boxes:
    #         if rect1 == rect2:
    #             continue

    #         if abs(rect1.left - rect2.left) > 4:
    #             continue

    #         if abs(1 - rect1.width / rect2.width) > 0.2:
    #             continue

    #         if rect1.top + rect1.height + 15 > rect2.top:
    #             to_remove.extend([rect1, rect2])
    #             new_rect = Rectangle(
    #                 min(rect1.left, rect2.left),
    #                 min(rect1.top, rect2.top),
    #                 max(rect1.width, rect2.width),
    #                 abs(rect2.top + rect2.height - rect1.top)
    #             )
    #             to_add.append(new_rect)

    # for r in to_remove:
    #     try:
    #         boxes.remove(r)
    #     except ValueError:
    #         pass

    # boxes.extend(to_add)

    # boxes = orig_boxes
    for box, color, font_size in boxes:
        draw = ImageDraw.Draw(img)
        draw.rectangle(box.to_xy(), outline=color)
        draw.rectangle(Rectangle(box.left, box.top, *font_size).to_xy(), outline='blue')
    return img, boxes
