from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Event, TouchCalibration


@dataclass
class TouchSample:
    frame: int
    gap_px: float | None = None
    x: float | None = None
    y: float | None = None
    front_width_px: float | None = None
    area_px: int = 0
    quality: float = 0.0


class ContactTracker:
    """Original-pixel contour relative to a user-confirmed contact reference."""

    def __init__(self, reference: np.ndarray, roi: list[float], calibration: TouchCalibration):
        height, width = reference.shape[:2]
        scale = np.array([width - 1, height - 1])
        polygon = np.rint(np.asarray(roi).reshape(4, 2) * scale).astype(np.int32)
        self.origin = polygon.min(axis=0)
        right = polygon.max(axis=0) + 1
        self.bounds = (slice(self.origin[1], right[1]), slice(self.origin[0], right[0]))
        self.reference = reference[self.bounds].copy()
        self.mask = np.zeros(self.reference.shape[:2], np.uint8)
        cv2.fillConvexPoly(self.mask, polygon - self.origin, 1)
        self.line = np.asarray(calibration.line).reshape(2, 2) * scale - self.origin
        self.seed = np.asarray(calibration.skin_point) * scale - self.origin
        tangent = self.line[1] - self.line[0]
        self.tangent = tangent / np.linalg.norm(tangent)
        self.normal = np.array([-self.tangent[1], self.tangent[0]])
        if (self.seed - self.line[0]) @ self.normal < 0:
            self.normal *= -1
        lab = cv2.cvtColor(self.reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        sx, sy = np.rint(self.seed).astype(int)
        swatch = lab[max(0, sy - 2) : sy + 3, max(0, sx - 2) : sx + 3].reshape(-1, 3)
        self.color = np.median(swatch, axis=0)
        spread = np.median(np.linalg.norm(swatch[:, 1:] - self.color[1:], axis=1))
        self.color_tolerance = float(np.clip(4 * spread + 10, 12, 30))
        foreground = self._foreground(lab)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        label = labels[sy, sx]
        if label == 0 or stats[label, cv2.CC_STAT_AREA] < 12:
            raise ValueError("指腹取样点附近没有稳定轮廓，请换一张清晰接触画面并重新取样")
        self.reference_mask = (labels == label).astype(np.uint8)
        self.reference_area = int(stats[label, cv2.CC_STAT_AREA])
        if self.reference_area > np.count_nonzero(self.mask) * 0.85:
            raise ValueError("取样颜色与背景难以区分，请调整触摸区域或指腹取样点")
        yy, xx = np.nonzero(self.reference_mask)
        coords = np.column_stack((xx, yy))
        self.reference_front = float(np.quantile((coords - self.line[0]) @ self.normal, 0.02))
        self.reference_center = coords.mean(axis=0)
        self.previous_center = self.reference_center.copy()

    def _foreground(self, lab: np.ndarray) -> np.ndarray:
        chroma = np.linalg.norm(lab[:, :, 1:] - self.color[1:], axis=2)
        return (
            (chroma <= self.color_tolerance)
            & (np.abs(lab[:, :, 0] - self.color[0]) <= 75)
            & (self.mask > 0)
        ).astype(np.uint8)

    def measure(self, frame: int, image: np.ndarray) -> TouchSample:
        lab = cv2.cvtColor(image[self.bounds], cv2.COLOR_BGR2LAB).astype(np.float32)
        foreground = self._foreground(lab)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
        candidates = [
            i
            for i in range(1, count)
            if max(8, self.reference_area * 0.08) <= stats[i, cv2.CC_STAT_AREA]
        ]
        if not candidates:
            return TouchSample(frame)
        label = min(candidates, key=lambda i: np.linalg.norm(centers[i] - self.previous_center))
        yy, xx = np.nonzero(labels == label)
        coords = np.column_stack((xx, yy))
        distances = (coords - self.line[0]) @ self.normal
        front = float(np.quantile(distances, 0.02))
        leading = coords[distances <= front + 4]
        across = (leading - self.line[0]) @ self.tangent
        width = float(np.quantile(across, 0.95) - np.quantile(across, 0.05))
        area = int(stats[label, cv2.CC_STAT_AREA])
        color_error = float(np.median(np.linalg.norm(lab[yy, xx, 1:] - self.color[1:], axis=1)))
        quality = min(1.0, area / (self.reference_area * 0.25)) * max(
            0.0, 1 - color_error / (self.color_tolerance * 1.5)
        )
        self.previous_center = centers[label]
        x, y = centers[label] + self.origin
        return TouchSample(
            frame, front - self.reference_front, float(x), float(y), width, area, quality
        )


def locate_contact(samples: list[TouchSample], fps: float, latest_press: int) -> tuple[Event, dict]:
    """Reference arrival is necessary; later hold/deformation confirms, never dates, it."""
    confirm = max(2, round(fps * 0.0125))
    lookback = max(6, round(fps * 0.10))
    tolerance = 1.5
    candidates = []
    arrived = [
        s.gap_px is not None and s.quality >= 0.35 and -3 <= s.gap_px <= tolerance for s in samples
    ]
    for i, sample in enumerate(samples):
        if sample.frame > latest_press:
            break
        if not arrived[i]:
            continue
        if i > 0 and arrived[i - 1] and samples[i - 1].frame + 1 == sample.frame:
            continue
        run_end = i
        while (
            run_end < len(samples)
            and arrived[run_end]
            and samples[run_end].frame == sample.frame + run_end - i
        ):
            run_end += 1
        if run_end - i < confirm:
            continue
        before = [
            s
            for s in samples[max(0, i - lookback) : i]
            if s.gap_px is not None and s.quality >= 0.35
        ]
        far = [s for s in before if s.gap_px > tolerance + 2]
        if len(far) < 2:
            continue
        closing = float(np.median([s.gap_px for s in far[-3:]]) - sample.gap_px)
        if closing < 2:
            continue
        widths = [s.front_width_px for s in far[-3:] if s.front_width_px is not None]
        old_width = float(np.median(widths)) if widths else 0.0
        last_far = next(s.frame for s in reversed(before) if s.gap_px > tolerance)
        unknown = any(
            (s.gap_px is None or s.quality < 0.35)
            for s in samples
            if last_far < s.frame < sample.frame
        )
        for end in range(i + confirm, run_end + 1):
            after = samples[end - confirm : end]
            speed = float(np.median(np.abs(np.diff([s.gap_px for s in after]))))
            new_width = float(np.median([s.front_width_px for s in after]))
            deformation = old_width > 0 and new_width - old_width >= max(1.5, old_width * 0.12)
            stopped = speed <= 0.7
            if stopped or deformation:
                candidates.append(
                    {
                        "first_candidate": sample.frame,
                        "confirmed_at": after[-1].frame,
                        "last_clear_approach": last_far,
                        "gap_px": sample.gap_px,
                        "hold_speed_px_per_frame": speed,
                        "stopped": stopped,
                        "deformation": bool(deformation),
                        "front_width_before": old_width,
                        "front_width_after": new_width,
                        "unobserved_before_contact": unknown,
                    }
                )
                break
        if candidates:
            break
    diagnostic = {
        "method": "calibrated_original_pixel_contour",
        "reference_tolerance_px": tolerance,
        "confirmation_frames": confirm,
        "sample_count": len(samples),
        "observed_frames": sum(s.gap_px is not None for s in samples),
        "candidates": candidates,
        "latest_press": latest_press,
    }
    if not candidates:
        return Event(
            reason="未找到接近参考位置后保持或形变的连续证据，请复核触摸区域和校准"
        ), diagnostic
    chosen = candidates[0]
    support = (
        "停留" + ("、指腹展宽" if chosen["deformation"] else "")
        if chosen["stopped"]
        else "指腹展宽"
    )
    reason = (
        f"原图轮廓到达接触参考位置，后续{support}确认；回溯至第{chosen['first_candidate']}帧。"
        f"请复核{chosen['last_clear_approach']}～{chosen['first_candidate']}帧；图像参考不能单独证明物理接触"
    )
    if chosen["unobserved_before_contact"]:
        reason += "；中间有轮廓不可观察帧"
    return Event(chosen["first_candidate"], "review", reason), diagnostic
