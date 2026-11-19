from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Event, Round
from .video import crop


class FeatureExtractor:
    """Measure original pixels inside polygons; no resampling of event boundaries."""

    def __init__(self, width: int, height: int, rois: dict[str, list[float]]):
        self.regions = {}
        for name in ("flash", "green"):
            points = np.rint(np.asarray(rois[name]).reshape(4, 2) * [width - 1, height - 1]).astype(
                np.int32
            )
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
            cv2.fillConvexPoly(mask, points - [x0, y0], 1)
            self.regions[name] = ((slice(y0, y1 + 1), slice(x0, x1 + 1)), mask.astype(bool))

    def __call__(self, image: np.ndarray) -> tuple[float, float, float]:
        bounds, mask = self.regions["flash"]
        light = cv2.cvtColor(image[bounds], cv2.COLOR_BGR2GRAY)[mask]
        bounds, mask = self.regions["green"]
        text = cv2.cvtColor(image[bounds], cv2.COLOR_BGR2HSV)
        green = cv2.inRange(text, (35, 65, 35), (90, 255, 255))[mask]
        black = text[:, :, 2][mask] < 110
        return float(np.mean(light)), float(np.mean(green > 0)), float(np.mean(black))


class TextColorTracker:
    """Match this recording's letter shape, not the color of its animated background."""

    def __init__(self, reference: np.ndarray, roi: list[float]):
        patch = crop(reference, roi)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        black = (hsv[:, :, 2] < 110).astype(np.uint8) * 255
        green = cv2.inRange(hsv, (35, 65, 35), (90, 255, 255))
        ink = black if 0.02 < np.mean(black > 0) < 0.6 else green
        if not 0.02 < np.mean(ink > 0) < 0.75:
            raise ValueError("文字区缺少可用字形参考，请让有效起帧包含清晰的PRESS HERE")
        x, y, w, h = cv2.boundingRect(ink)
        x, y = max(0, x - 2), max(0, y - 2)
        self.glyph = ink[y : min(ink.shape[0], y + h + 4), x : min(ink.shape[1], x + w + 4)]
        if self.glyph.shape[0] < 8 or self.glyph.shape[1] < 20 or self.glyph.std() == 0:
            raise ValueError("文字字形太小或缺少轮廓，请调整文字选区")
        points = np.asarray(roi).reshape(4, 2)
        center = points.mean(axis=0)
        self.search_roi = np.clip(center + (points - center) * 1.3, 0, 1).reshape(-1).tolist()
        search = crop(reference, self.search_roi)
        self.templates = []
        for scale in (0.9, 0.95, 1.0, 1.05, 1.1):
            template = cv2.resize(
                self.glyph, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            )
            if template.shape[0] <= search.shape[0] and template.shape[1] <= search.shape[1]:
                self.templates.append((scale, template))
        self.last_matches = {}

    def __call__(self, image: np.ndarray) -> tuple[float, float]:
        self.last_matches = self.match(image)
        return self.last_matches["green"]["score"], self.last_matches["black"]["score"]

    def match(self, image: np.ndarray) -> dict:
        """Match one frame using the read-only templates; poses belong to this frame."""
        hsv = cv2.cvtColor(crop(image, self.search_roi), cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (35, 65, 35), (90, 255, 255))
        black = ((hsv[:, :, 2] < 110) & (green == 0)).astype(np.uint8) * 255
        matches = {}
        for name, mask in (("green", green), ("black", black)):
            best = (-1.0, 1.0, (0, 0))
            for scale, template in self.templates:
                _, score, _, position = cv2.minMaxLoc(
                    cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
                )
                if score > best[0]:
                    best = (score, scale, position)
            matches[name] = {"score": best[0], "scale": best[1], "position": best[2]}
        return matches


def refine_green_start(
    tracker: TextColorTracker,
    read_at,
    proof: dict,
    fps: float,
    start_frame: int = 0,
    cancelled=lambda: False,
    progress=lambda frame: None,
    record=lambda value: None,
) -> dict:
    """Revisit confirmed responses using the same previously dark letter pixels."""
    anchor = proof["last_black_frame"]
    coarse = proof["first_green_candidate"]
    confirm = max(2, round(fps * 0.025))
    result = {
        "coarse_frame": coarse,
        "reference_frame": anchor,
        "candidate_frame": None,
        "confirmed_at": None,
        "reason": "没有更早的连续笔画变色证据",
        "thresholds": {
            "bright_green_value_min": 110,
            "min_stroke_pixels": 24,
            "min_sector_pixels": 8,
            "min_supported_sectors": 2,
            "bright_green_fraction": 0.5,
            "chroma_contrast": 10,
            "confirmation_frames": confirm,
            "max_gap_frames": 2,
            "translation_radius_px": 4,
            "min_alignment_score": 0.1,
        },
    }
    if coarse <= anchor + 1:
        result["reason"] = "粗检候选已紧邻最后黑字，无提前区间"
        return result
    if anchor - 2 < start_frame:
        result["reason"] = "有效区间内不足三帧黑字参考"
        return result
    if cancelled():
        raise InterruptedError("已取消")
    progress(anchor)
    reference = read_at(anchor)
    tracker(reference)
    pose = tracker.last_matches["black"]
    if pose["score"] < 0.25:
        result["reason"] = "黑字参考匹配不足，保留原候选"
        return result
    template = next(t for scale, t in tracker.templates if scale == pose["scale"]) > 0
    pattern = template.astype(np.uint8) * 255
    height, width = template.shape
    x, y = pose["position"]
    core = template.copy()
    for index in range(anchor - 2, anchor + 1):
        if cancelled():
            raise InterruptedError("已取消")
        progress(index)
        image = reference if index == anchor else read_at(index)
        patch = crop(image, tracker.search_roi)[y : y + height, x : x + width]
        core &= cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2] < 110
        record({"kind": "reference_sample", "frame": index, "stroke_pixels": int(core.sum())})
    ring = (cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~template
    edges = np.linspace(0, width, 4, dtype=int)
    sectors = [core[:, a:b] for a, b in zip(edges, edges[1:])]
    sector_sizes = [int(mask.sum()) for mask in sectors]
    result.update(
        {
            "reference_pose": {"position": [x, y], "scale": pose["scale"], "score": pose["score"]},
            "stroke_pixels": int(core.sum()),
            "ring_pixels": int(ring.sum()),
            "sector_pixels": sector_sizes,
        }
    )
    record({"kind": "reference", **result})
    if core.sum() < 24 or ring.sum() < 8 or sum(n >= 8 for n in sector_sizes) < 2:
        result["reason"] = "可用笔画或空间分区不足，保留原候选"
        return result
    first, support, gap = None, 0, 0
    for index in range(anchor + 1, proof["confirmed_at"] + 1):
        if cancelled():
            raise InterruptedError("已取消")
        progress(index)
        search = crop(read_at(index), tracker.search_roi)
        hsv_search = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        bright_mask = cv2.inRange(hsv_search, (35, 65, 110), (90, 255, 255))
        dark_mask = (hsv_search[:, :, 2] < 110).astype(np.uint8) * 255
        # Register the same glyph locally; a shifted dark word must not expose
        # green background under the old stroke coordinates and look like an onset.
        x0, y0 = max(0, x - 4), max(0, y - 4)
        x1, y1 = min(search.shape[1] - width, x + 4), min(search.shape[0] - height, y + 4)
        dark_scores = cv2.matchTemplate(
            dark_mask[y0 : y1 + height, x0 : x1 + width], pattern, cv2.TM_CCOEFF_NORMED
        )
        green_scores = cv2.matchTemplate(
            bright_mask[y0 : y1 + height, x0 : x1 + width], pattern, cv2.TM_CCOEFF_NORMED
        )
        _, alignment, _, position = cv2.minMaxLoc(np.maximum(dark_scores, green_scores))
        px, py = x0 + position[0], y0 + position[1]
        patch = search[py : py + height, px : px + width]
        hsv = hsv_search[py : py + height, px : px + width]
        bright = bright_mask[py : py + height, px : px + width] > 0
        color = patch.astype(np.float32)
        chroma = color[:, :, 1] - np.maximum(color[:, :, 0], color[:, :, 2])
        coverage = float(bright[core].mean())
        contrast = float(chroma[core].mean() - chroma[ring].mean())
        sector_coverage = [
            float(bright[:, a:b][mask].mean()) if size >= 8 else None
            for a, b, mask, size in zip(edges, edges[1:], sectors, sector_sizes)
        ]
        supported = sum(value is not None and value >= 0.5 for value in sector_coverage)
        passed = alignment >= 0.1 and coverage >= 0.5 and contrast >= 10 and supported >= 2
        record(
            {
                "kind": "sample",
                "frame": index,
                "bright_green_fraction": coverage,
                "dark_fraction": float((hsv[:, :, 2][core] < 110).mean()),
                "chroma_contrast": contrast,
                "sector_bright_green": sector_coverage,
                "supported_sectors": supported,
                "position": [px, py],
                "alignment_score": alignment,
                "passed": passed,
            }
        )
        if passed:
            if first is None:
                first = index
            support += 1
            gap = 0
        else:
            gap += 1
            if gap > 2:
                first, support = None, 0
        if support >= confirm:
            if first < coarse:
                result.update(
                    candidate_frame=first, confirmed_at=index, reason="连续亮绿笔画支持更早边界"
                )
            else:
                result["reason"] = "笔画证据没有支持早于原候选的帧"
            return result
    return result


def stable_runs(values: np.ndarray, minimum: int = 2) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
        if b - a >= minimum
    ]


@dataclass
class Detections:
    rounds: list[Round]
    green_ends: list[int]
    notes: list[str]
    diagnostics: dict


def detect_rounds(
    light: list[float],
    green: list[float],
    black: list[float],
    fps: float = 240.0,
    start_frame: int = 0,
    stop_frame: int | None = None,
) -> Detections:
    brightness, color = np.asarray(light), np.asarray(green)
    stop_frame = len(color) if stop_frame is None else stop_frame
    notes = []
    # Follow the illuminated level only while on; retain one-frame dark pulses.
    follow = max(2, round(fps * 0.05))
    level = float(brightness[start_frame])
    armed, bright_count = False, 0
    offs = []
    for i in range(start_frame, stop_frame):
        value = float(brightness[i])
        if value > level:
            level = value
        if value < level - max(8.0, level * 0.05):
            if armed:
                offs.append(i)
            armed, bright_count = False, 0
        else:
            level += (value - level) / follow
            bright_count += 1
            if bright_count >= 2:
                armed = True
    if not offs:
        notes.append("未找到显著的灯光下降沿，请检查灯光选区")

    # Positive and negative letter structure suppresses unrelated colored circles.
    black_scores = np.asarray(black)
    green_state = (color >= 0.25) & (color > black_scores + 0.04)
    black_state = (black_scores >= 0.25) & (black_scores > color + 0.04)
    confirm = max(2, round(fps * 0.025))
    reset = max(3, round(fps * 0.05))
    greens = []
    ready = bool(black_state[start_frame])
    active = None
    green_count, black_count = 0, 0
    green_start, green_gap = None, 0
    last_black = start_frame - 1
    candidate_last_black = last_black
    evidence = {}
    for i in range(start_frame, stop_frame):
        is_black = bool(black_state[i])
        black_count = black_count + 1 if is_black else 0
        if is_black:
            last_black = i
        if active is not None:
            if black_count >= reset:
                greens.append((active, i - reset + 1))
                active, ready, green_count = None, True, 0
                green_start, green_gap = None, 0
            continue
        if black_count >= reset:
            ready = True
        if green_state[i]:
            if green_start is None:
                green_start = i
                candidate_last_black = last_black
            green_count += 1
            green_gap = 0
        else:
            green_gap += 1
            if green_gap > 2:
                green_start, green_count = None, 0
        if ready and green_count >= confirm:
            active = green_start
            evidence[active] = {
                "last_black_frame": candidate_last_black,
                "first_green_candidate": active,
                "confirmed_at": i,
                "shape_score": float(np.median(color[active : i + 1])),
            }
            ready, black_count = False, 0
    if active is not None:
        greens.append((active, stop_frame))
    if not greens:
        notes.append("未找到稳定黑字复位后的变绿响应，请检查文字区域")

    rounds, ends, used, orphan_greens = [], [], set(), []
    previous = start_frame
    for start, end in greens:
        candidates = [i for i in offs if previous <= i <= start]
        previous = end
        if not candidates:
            orphan_greens.append(start)
            continue
        proof = evidence[start]
        uncertain = start - proof["last_black_frame"] > 2 or proof["shape_score"] < 0.4
        reason = (
            f"字形变色边界需复核：{proof['last_black_frame'] + 1}～{start}帧"
            if uncertain
            else "黑字变为绿字的首个匹配帧"
        )
        item = Round(
            green=Event(start, "review" if uncertain else "auto", reason), last_green_frame=end - 1
        )
        if len(candidates) == 2:
            item.blink1 = Event(candidates[0], "auto", "第一次亮到暗")
            item.blink2 = Event(candidates[1], "auto", "第二次亮到暗")
        else:
            selected = candidates[-2:]
            item.blink1 = Event(selected[0], "review", "本轮灭灯数量异常，候选待确认")
            if len(selected) == 2:
                item.blink2 = Event(selected[1], "review", "本轮灭灯数量异常，候选待确认")
            item.issues.append(f"此响应前有{len(candidates)}次灭灯，配对需复核")
        used.update(candidates[-2:])
        rounds.append(item)
        ends.append(end)
    unmatched = [i for i in offs if i not in used]
    if unmatched:
        notes.append(
            "未配对灭灯仅保留为异常候选，不自动新增轮次。帧号：" + ", ".join(map(str, unmatched))
        )
    if orphan_greens:
        notes.append(
            "缺少灯光对应的变绿候选需复核，未自动新增轮次。帧号："
            + ", ".join(map(str, orphan_greens))
        )
    notes.insert(
        0,
        f"检测到{len(offs)}次灭灯、{len(greens)}段复位后变绿、{len(rounds)}个有灯光对应的轮次候选；未按10轮截断。",
    )
    diagnostics = {
        "flash_off_frames": offs,
        "green_segments": greens,
        "unmatched_flash_frames": unmatched,
        "unmatched_green_frames": orphan_greens,
        "thresholds": {
            "sampling_fps": fps,
            "flash_method": "adaptive_on_level_drop",
            "flash_follow_frames": follow,
            "flash_min_drop": 8.0,
            "flash_drop_fraction": 0.05,
            "text_min_match": 0.25,
            "text_color_margin": 0.04,
            "green_confirm_frames": confirm,
            "initial_green_gap_frames": 2,
            "black_reset_frames": reset,
        },
        "raw_green_segments": [
            (a + start_frame, b + start_frame)
            for a, b in stable_runs(green_state[start_frame:stop_frame])
        ],
        "green_evidence": list(evidence.values()),
        "analysis_range": {"start": start_frame, "end": stop_frame - 1},
        "round_candidates": [
            {
                "blink1": r.blink1.frame,
                "blink2": r.blink2.frame,
                "green": r.green.frame,
                "green_end": end,
                "issues": r.issues,
            }
            for r, end in zip(rounds, ends)
        ],
    }
    return Detections(rounds, ends, notes, diagnostics)
