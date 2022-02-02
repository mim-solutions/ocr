
import unittest

from PIL import Image

from lib.parse_png import parse_png

RUN_OCR_TEST = True


class TestParsePng(unittest.TestCase):
    """
    Basic test case for the PNG pipeline
    """

    @unittest.skipIf(RUN_OCR_TEST is False, "skip test which costs money")
    def test_example(self):
        """
        Run a test of the whole OCR. It performs 7 requests to the
        Cloud Vision API, so run it carefully
        """
        img = Image.open('test/sample_data/example.png')
        result = parse_png(img)

        with open('test/sample_data/example.txt', encoding='utf-8') as doc:
            expected = doc.read()

        self.assertEqual(result, expected)
