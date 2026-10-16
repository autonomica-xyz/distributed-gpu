"""Bound Argos MiniSBD's CPU fallback without disabling CUDA translation."""

from __future__ import annotations

import os

from argostranslate import sbd


def _minisbd_threads() -> int:
    try:
        return max(1, int(os.getenv("SPAREHOUR_MINISBD_THREADS", "1")))
    except ValueError:
        return 1


def _bounded_detector(self):
    if self.detector is None:
        self.detector = sbd.SBDetect(
            self.lang,
            use_gpu=False,
            max_threads=_minisbd_threads(),
        )
    return self.detector


sbd.MiniSBDSentencizer.lazy_detector = _bounded_detector
