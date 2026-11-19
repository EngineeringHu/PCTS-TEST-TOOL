"""Optional earlier green onset from color changes on the same letter strokes."""

from collections import deque

import cv2
import numpy as np

MIN_LEAD_FRAMES = 6


def interior_glyph(glyph):
    """Discard components clipped by the ROI, including dark strips along the bezel."""
    _, labels = cv2.connectedComponents((glyph > 0).astype(np.uint8))
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return ((glyph > 0) & ~np.isin(labels, border)).astype(np.uint8)


class GreenOnsetProbe:
    """Consume an original-frame crop at a time during the existing touch/OCR pass."""

    def __init__(self, glyph, pose, blink2, baseline):
        x, y, scale = pose
        self.template = cv2.resize(glyph, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        height, width = self.template.shape
        self.bounds = (slice(y, y + height), slice(x, x + width))
        self.blink2 = blink2
        self.start = blink2 - 2
        # A candidate needs four following frames to complete its five-frame evidence window.
        self.stop = baseline - MIN_LEAD_FRAMES + 4
        self.reference = []
        self.sectors = []
        self.support = deque(maxlen=5)
        self.candidate = None

    def observe(self, index, search):
        patch = search[self.bounds]
        if index <= self.blink2:
            self.reference.append(patch.copy())
            if index == self.blink2:
                self._set_reference()
            return

        pixels = patch[self.y, self.x]
        delta = pixels.astype(np.int16) - self.base
        dg = delta[:, 1]
        # Twice the change in G - (B + R)/2. Integer units preserve half-level changes.
        chroma = 2 * dg - delta[:, 0] - delta[:, 2]
        weak = (dg >= 12) & (chroma >= 12)
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        bright = cv2.inRange(hsv, (35, 65, 110), (90, 255, 255)).ravel() > 0
        strong = (dg >= 20) & (chroma >= 20) & bright
        weak_sectors = strong_sectors = 0
        for strokes, ring in self.sectors:
            contrast = np.median(chroma[strokes]) - np.median(chroma[ring])
            weak_sectors += int(contrast >= 6 and weak[strokes].mean() >= 0.15)
            strong_sectors += int(contrast >= 16 and strong[strokes].mean() >= 0.5)
        self.support.append((weak_sectors >= 2, strong_sectors >= 2))
        if len(self.support) == 5:
            weak_votes, strong_votes = zip(*self.support)
            if all(weak_votes[:3]) and sum(strong_votes) >= 3:
                self.candidate = index - 4
                self.stop = index

    def _set_reference(self):
        base = np.median(self.reference, axis=0).astype(np.int16)
        self.reference.clear()
        strokes = (self.template > 0) & (base.max(axis=2) < 110)
        ring = (cv2.dilate(self.template, np.ones((7, 7), np.uint8)) > 0) & (self.template == 0)
        self.y, self.x = np.nonzero(strokes | ring)
        self.base = base[self.y, self.x]
        ink = strokes[self.y, self.x]
        for columns in np.array_split(np.arange(self.template.shape[1]), 3):
            if columns.size:
                sector = (self.x >= columns[0]) & (self.x <= columns[-1])
                inside, outside = np.flatnonzero(sector & ink), np.flatnonzero(sector & ~ink)
                if inside.size >= 24 and outside.size >= 12:
                    self.sectors.append((inside, outside))
        if len(self.sectors) < 2:
            self.stop = self.blink2
