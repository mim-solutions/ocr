from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Rectangle:
    left: int
    top: int
    width: int
    height: int

    @staticmethod
    def from_tesseract(data, i: int) -> Rectangle:
        try:
            left, top, width, height = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            )

            return Rectangle(left, top, width, height)
        except KeyError:
            left, top, right, bottom = (
                data['left'][i],
                data['top'][i],
                data['right'][i],
                data['bottom'][i]
            )

            return Rectangle(left, top, right - left, bottom - top)

    @staticmethod
    def from_gcloud(data) -> Rectangle:
        assert not data['normalizedVertices']
        assert len(data['vertices']) == 4

        def is_close(a: int, b: int) -> bool:
            return abs(b - a) < 4000  # FIXME: fix that margin to something ~3?

        lt, rt, rb, lb = data['vertices']
        assert is_close(lt['x'], lb['x'])
        assert is_close(rt['x'], rb['x'])
        assert is_close(lt['y'], rt['y'])
        assert is_close(lb['y'], rb['y'])

        left, top, width, height = (
            lt['x'],
            lt['y'],
            rt['x'] - lt['x'],
            lb['y'] - lt['y']
        )

        return Rectangle(left, top, width, height)

    def to_pillow(self) -> tuple[int]:
        return (
            self.left,
            self.top,
            self.left + self.width,
            self.top + self.height
        )

    def to_matplotlib(self) -> tuple[tuple[int, int], int, int]:
        return (
            (self.left, self.top),
            self.width,
            self.height
        )

    def to_xy(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (self.left, self.top),
            (self.left + self.width, self.top + self.height)
        )

    def shift(self, vector: tuple[int, int]) -> Rectangle:
        return Rectangle(
            self.left + vector[0],
            self.top + vector[1],
            self.width,
            self.height
        )

    def get_area(self) -> int:
        return self.width * self.height

    def is_inside(self, rect: Rectangle, margin: float) -> bool:
        """margin in [0, 1], percentage of area contained in rect"""
        left = max(self.left, rect.left)
        right = min(self.left + self.width, rect.left + rect.width)
        if not left < right:
            return False

        top = max(self.top, rect.top)
        bottom = min(self.top + self.height, rect.top + rect.height)
        if not top < bottom:
            return False

        assert left >= 0
        assert top >= 0
        assert right - left >= 0
        assert bottom - top >= 0
        common_rect = Rectangle(left, top, right - left, bottom - top)

        assert left >= self.left
        assert right <= self.left + self.width
        assert top >= self.top
        assert bottom <= self.top + self.height
        assert left <= right
        assert top <= bottom

        # print(common_rect.get_area(), self.get_area())
        assert common_rect.get_area() <= self.get_area()
        return common_rect.get_area() / self.get_area() >= margin
