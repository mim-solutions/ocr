"""
This module implements a wrapper class
"""
import io

from PIL import Image
from google.cloud import vision
# https://github.com/googleapis/python-vision/issues/70
import proto

from lib.gcloud.auth import get_credentials
from lib.rectangle import Rectangle


class Annotation:
    text_annotations: dict
    full_text_annotation: dict
    avg_confidence: float = None
    full_text: str = None

    def __init__(self, data: dict):
        all_keys = [
            'textAnnotations',
            'fullTextAnnotation',
            'faceAnnotations',
            'landmarkAnnotations',
            'logoAnnotations',
            'labelAnnotations',
            'localizedObjectAnnotations'
        ]

        assert set(data.keys()).issubset(set(all_keys))
        if set(data.keys()) != set(all_keys):
            return

        assert not any(data[key] for key in all_keys[2:])
        assert all(data[key] for key in all_keys[:2])

        self.text_annotations = data['textAnnotations']
        self.full_text_annotation = data['fullTextAnnotation']

        self.parse_text_annotations()
        self.parse_full_text_annotation()

    def parse_text_annotations(self):
        assert isinstance(self.text_annotations, list)

        for annotation in self.text_annotations:
            text, rect, locale, confidence = self.parse_annotation(annotation)

        # is_polish = [elem['locale'] == 'pl' for elem in self.text_annotations]
        # confidences = [elem['confidence'] for elem in self.text_annotations]

    def parse_full_text_annotation(self):
        assert isinstance(self.full_text_annotation, dict)
        assert set(self.full_text_annotation.keys()) == {'pages', 'text'}

        text = self.full_text_annotation['text']
        assert isinstance(text, str)
        self.full_text = text

        assert isinstance(self.full_text_annotation['pages'], list)

        # we only gcloud after cutting to boxes
        assert len(self.full_text_annotation['pages']) == 1
        page = self.full_text_annotation['pages'][0]

        assert isinstance(page, dict)

        assert set(page.keys()).issubset({'property', 'width', 'height', 'blocks', 'confidence'})
        assert 'blocks' in page

        if 'property' in page:
            assert set(page['property'].keys()) == {'detectedLanguages'}
            for lang in page['property']['detectedLanguages']:
                assert set(lang.keys()) == {
                    'languageCode', 'confidence'
                }
            # assert page['property']['detectedLanguages'][0]['languageCode'] == 'pl'
            # assert page['property']['detectedLanguages'][0]['confidence'] > 0.5  # FIXME

        for block in page['blocks']:
            assert set(block.keys()).issubset(
                {
                    'property',
                    'boundingBox',
                    'paragraphs',
                    'blockType',
                    'confidence'
                }
            )
            if 'property' in block:
                assert set(block['property'].keys()) == {'detectedLanguages'}
                for lang in block['property']['detectedLanguages']:
                    assert set(lang.keys()) == {
                        'languageCode', 'confidence'
                    }
                    lang['confidence']  # FIXME: use this

        paragraphs = [
            par
            for block in page['blocks']
            for par in block['paragraphs']
        ]
        words = [word for par in paragraphs for word in par['words']]
        symbols = [sym for word in words for sym in word['symbols']]
        confidences = [sym['confidence'] for sym in symbols]
        self.avg_confidence = sum(confidences) / len(confidences)

    def parse_annotation(self, elem: dict):
        assert isinstance(elem, dict)
        all_keys = [
            'locale',
            'description',
            'boundingPoly',
            'mid',
            'score',
            'confidence',
            'topicality',
            'locations',
            'properties'
        ]

        assert sorted(list((elem.keys()))) == sorted(all_keys)

        unused_keys = [all_keys[i] for i in [3, 4, 6, 7, 8]]

        assert not any(elem[key] for key in unused_keys)
        # confidence might be 0.0, locale might not be detected
        assert all(
            elem[key]
            for key in all_keys
            if key not in unused_keys + ['confidence', 'locale']
        )

        rect = Rectangle.from_gcloud(elem['boundingPoly'])
        return elem['description'], rect, elem['locale'], elem['confidence']


def annotate(img: Image.Image) -> str:
    """Returns JSON"""
    scopes = ['https://www.googleapis.com/auth/cloud-vision']
    credentials = get_credentials(scopes)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # with io.open(img_path, 'rb') as doc:
    #     content = doc.read()
    buf = io.BytesIO()
    img.save(buf, format=img.format)
    content = buf.getvalue()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message)
        )

    # with open(result_path, 'w+') as doc:
    json_string = proto.Message.to_json(response)
    return json_string
    #     doc.write(json_string)
