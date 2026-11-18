from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Event
from .touch import FingerRegion


@dataclass
class FingerObservation:
    frame: int
    mask: np.ndarray | None = None
    quality: float = 0.0
    flow_points: int = 0
    overlap: float = 0.0
    reason: str = "未跟踪到连续指腹"
    flow_x: float = 0.0
    flow_y: float = 0.0
    area_change: float = 0.0


def prepare_motion_tracker(
    read_at,
    roi,
    start,
    stop,
    blink1,
    fps,
    *,
    detector=None,
    sample=None,
    cancelled=lambda: False,
    progress=lambda frame: None,
):
    """Pick an appearance anchor, not a contact timestamp; include the full round window."""
    frames = sorted(
        {int(round(v)) for v in np.linspace(start, blink1, 9)}
        | {
            max(start, min(stop, blink1 + round(offset * fps)))
            for offset in (-0.1, -0.05, -0.025, 0.015)
        }
    )
    if sample and start <= sample.frame <= stop:
        frames = sorted(set(frames) | {sample.frame})
    report = {
        "source": "manual_sample" if sample else "hand_connected_color",
        "attempts": [],
        "reference_frame": None,
    }
    appearance = MotionTracker(read_at(sample.frame), roi, sample.skin_point) if sample else None
    best, chosen, chosen_frame = -1.0, None, None
    for frame in frames:
        if cancelled():
            raise InterruptedError("已取消")
        progress(frame)
        picture = read_at(frame)
        height, width = picture.shape[:2]
        if appearance:
            score, point = appearance.anchor_score(picture)
            options = (
                []
                if point is None or score < 0.3
                else [
                    {
                        "point": (np.asarray(point) / [width - 1, height - 1]).tolist(),
                        "appearance_score": score,
                    }
                ]
            )
        elif detector is not None:
            options = automatic_finger_samples(detector, picture, roi)
        else:
            options = []
        attempt = {"frame": frame, "seed_count": len(options), "valid_samples": []}
        for option in options:
            try:
                tracker = MotionTracker(picture, roi, option["point"])
            except ValueError as error:
                attempt.setdefault("rejected", []).append(str(error))
                continue
            score = option.get(
                "appearance_score",
                min(option.get("interior_radius", 0), 8) / tracker.color_tolerance,
            )
            attempt["valid_samples"].append(
                {
                    "point": option["point"],
                    "score": score,
                    "color_lab": tracker.color.tolist(),
                    "area_px": tracker.reference_area,
                }
            )
            if score > best:
                best, chosen, chosen_frame = score, tracker, frame
                report["point"] = option["point"]
        report["attempts"].append(attempt)
    report["reference_frame"] = chosen_frame
    if chosen:
        report.update(
            color_lab=chosen.color.tolist(),
            color_tolerance=chosen.color_tolerance,
            area_px=chosen.reference_area,
        )
    else:
        report["reason"] = "没有可靠的指腹取样，可在指腹设置中单点指定皮肤样本"
    return chosen, chosen_frame, report


def automatic_finger_samples(detector, image: np.ndarray, roi: list[float]) -> list[dict]:
    """A palm anchors identity; its connected color locates the missed extended fingertip."""
    import mediapipe as mp

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    height, width = image.shape[:2]
    scale = np.array([width - 1, height - 1])
    polygon = np.rint(np.asarray(roi).reshape(4, 2) * scale).astype(np.int32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidates = []
    for hand in result.hand_landmarks:
        points = np.array([[p.x, p.y] for p in hand]) * scale
        palm = points[[5, 9, 13, 17]].mean(axis=0)
        if np.linalg.norm(palm - polygon.mean(axis=0)) > max(
            150, np.ptp(points, axis=0).max() * 1.5
        ):
            continue
        sx, sy = np.rint(palm).astype(int)
        if not 2 <= sx < width - 2 or not 2 <= sy < height - 2:
            continue
        swatch = lab[sy - 2 : sy + 3, sx - 2 : sx + 3].reshape(-1, 3)
        color = np.median(swatch, axis=0)
        tolerance = float(
            np.clip(4 * np.median(np.linalg.norm(swatch[:, 1:] - color[1:], axis=1)) + 10, 12, 30)
        )
        combined = np.vstack((points, polygon))
        low = np.maximum(0, np.floor(combined.min(axis=0)).astype(int) - 10)
        high = np.minimum([width, height], np.ceil(combined.max(axis=0)).astype(int) + 11)
        patch = lab[low[1] : high[1], low[0] : high[0]]
        foreground = (
            (np.linalg.norm(patch[:, :, 1:] - color[1:], axis=2) <= tolerance)
            & (np.abs(patch[:, :, 0] - color[0]) <= 75)
        ).astype(np.uint8)
        _, labels = cv2.connectedComponents(foreground)
        label = labels[sy - low[1], sx - low[0]]
        if label == 0:
            continue
        roi_mask = np.zeros(foreground.shape, np.uint8)
        cv2.fillConvexPoly(roi_mask, polygon - low, 1)
        connected = ((labels == label) & (roi_mask > 0)).astype(np.uint8)
        if connected.sum() < 24:
            continue
        distance = cv2.distanceTransform(connected, cv2.DIST_L2, 5)
        _, radius, _, interior = cv2.minMaxLoc(distance)
        if radius < 2:
            continue
        point = (np.asarray(interior) + low) / scale
        candidates.append(
            {"point": point.tolist(), "interior_radius": radius, "palm_color_lab": color.tolist()}
        )
    return candidates


class MotionTracker(FingerRegion):
    """Trace one sampled finger bidirectionally; never reacquire arbitrary background."""

    def anchor_score(self, image: np.ndarray) -> tuple[float, list[float] | None]:
        lab = cv2.cvtColor(image[self.bounds], cv2.COLOR_BGR2LAB).astype(np.float32)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(self._foreground(lab))
        best = (0.0, None)
        pattern = self.reference_mask * 255
        pad = max(4, round(min(pattern.shape) * 0.25))
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if not self.reference_area * 0.2 <= area <= self.reference_area * 3:
                continue
            mask = (labels == label).astype(np.uint8)
            padded = cv2.copyMakeBorder(mask * 255, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            _, score, _, _ = cv2.minMaxLoc(cv2.matchTemplate(padded, pattern, cv2.TM_CCOEFF_NORMED))
            if score > best[0]:
                _, radius, _, point = cv2.minMaxLoc(cv2.distanceTransform(mask, cv2.DIST_L2, 5))
                if radius >= 2:
                    best = (score, (np.asarray(point) + self.origin).tolist())
        return best

    def trace(
        self, patches: list[tuple[int, np.ndarray]], anchor_frame: int, cancelled=lambda: False
    ) -> list[FingerObservation]:
        observations = [FingerObservation(frame) for frame, _ in patches]
        anchor = next(i for i, (frame, _) in enumerate(patches) if frame == anchor_frame)
        observations[anchor] = FingerObservation(
            anchor_frame, self.reference_mask.copy(), 1.0, reason="本轮指腹样本"
        )
        for direction in (-1, 1):
            previous_gray = cv2.cvtColor(patches[anchor][1], cv2.COLOR_BGR2GRAY)
            previous_mask = self.reference_mask.copy()
            previous_center = self.reference_center.copy()
            previous_frame = anchor_frame
            lost = 0
            for i in range(anchor + direction, len(patches) if direction > 0 else -1, direction):
                if cancelled():
                    raise InterruptedError("已取消")
                frame, picture = patches[i]
                gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
                lab = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB).astype(np.float32)
                count, labels, stats, centers = cv2.connectedComponentsWithStats(
                    self._foreground(lab)
                )
                flow, good_count = np.zeros(2), 0
                # Erosion selects interior texture only; measured contours remain unmodified.
                feature_mask = cv2.erode(previous_mask, np.ones((3, 3), np.uint8)) * 255
                features = cv2.goodFeaturesToTrack(previous_gray, 24, 0.01, 3, mask=feature_mask)
                if features is not None and len(features) >= 3:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(
                        previous_gray, gray, features, None, winSize=(15, 15), maxLevel=2
                    )
                    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                        gray, previous_gray, moved, None, winSize=(15, 15), maxLevel=2
                    )
                    error = np.linalg.norm(features.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
                    good = (status.ravel() > 0) & (backward_status.ravel() > 0) & (error <= 1.5)
                    good_count = int(good.sum())
                    if good_count >= 3:
                        flow = np.median((moved - features).reshape(-1, 2)[good], axis=0)
                transform = np.float32([[1, 0, flow[0]], [0, 1, flow[1]]])
                predicted = cv2.warpAffine(
                    previous_mask,
                    transform,
                    (gray.shape[1], gray.shape[0]),
                    flags=cv2.INTER_NEAREST,
                )
                best = None
                for label in range(1, count):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    if (
                        not max(8, self.reference_area * 0.06)
                        <= area
                        <= min(self.mask.sum() * 0.9, self.reference_area * 3)
                    ):
                        continue
                    mask = labels == label
                    overlap = float(
                        np.count_nonzero(mask & (predicted > 0))
                        / min(area, max(1, int(previous_mask.sum())))
                    )
                    shift = float(np.linalg.norm(centers[label] - previous_center - flow))
                    if overlap < 0.2 and not (
                        good_count >= 3 and shift <= max(3, np.sqrt(area) * 0.2)
                    ):
                        continue
                    color_error = float(
                        np.median(np.linalg.norm(lab[mask, 1:] - self.color[1:], axis=1))
                    )
                    quality = max(0.0, 1 - color_error / (self.color_tolerance * 1.5)) * min(
                        1.0, area / (self.reference_area * 0.25)
                    )
                    rank = (
                        overlap
                        + quality * 0.3
                        - shift / max(10, np.sqrt(self.reference_area)) * 0.15
                    )
                    if best is None or rank > best[0]:
                        best = (rank, mask.astype(np.uint8), centers[label], quality, overlap)
                if best is None or best[3] < 0.3:
                    lost += 1
                    observations[i].reason = "指腹轮廓失踪或连续性不足"
                    if lost >= 2:
                        break
                    continue
                _, mask, center, quality, overlap = best
                delta = frame - previous_frame
                observations[i] = FingerObservation(
                    frame,
                    mask,
                    quality,
                    good_count,
                    overlap,
                    "连续指腹轮廓",
                    float(flow[0] / delta),
                    float(flow[1] / delta),
                    (int(mask.sum()) - int(previous_mask.sum())) / delta,
                )
                previous_mask, previous_center, previous_gray = mask, center, gray
                previous_frame = frame
                lost = 0
        return observations


def locate_motion_press(
    observations: list[FingerObservation],
    fps: float,
    latest_press: int,
    origin: np.ndarray,
    roi_mask: np.ndarray | None = None,
) -> tuple[Event, dict, list[dict]]:
    diagnostic = {
        "method": "automatic_finger_motion",
        "direction": None,
        "candidates": [],
        "rejected_pauses": [],
        "rejected_clipping": [],
        "rejected_moving": [],
        "latest_press": latest_press,
    }
    valid = [o for o in observations if o.mask is not None and o.quality >= 0.35]
    rows = [
        {
            "frame": o.frame,
            "quality": o.quality,
            "area_px": int(o.mask.sum()) if o.mask is not None else 0,
            "flow_points": o.flow_points,
            "overlap": o.overlap,
            "reason": o.reason,
            "flow_x": o.flow_x,
            "flow_y": o.flow_y,
        }
        for o in observations
    ]
    diagnostic["observed_frames"] = len(valid)
    before = [o for o in valid if o.frame <= latest_press]
    if len(before) < 3 or len(valid) < 6:
        return (
            Event(reason="自动轨迹：可见指腹帧不足，可在指腹设置中点选清晰样本或扩大观察区"),
            diagnostic,
            rows,
        )
    centers = np.array([np.column_stack(np.nonzero(o.mask)[::-1]).mean(axis=0) for o in valid])
    areas = np.array([o.mask.sum() for o in valid], dtype=float)
    for row, observation in zip(rows, observations):
        if row["area_px"]:
            center = np.column_stack(np.nonzero(observation.mask)[::-1]).mean(axis=0) + origin
            row.update(x=float(center[0]), y=float(center[1]))
    centered = centers - centers.mean(axis=0)
    _, values, axes = np.linalg.svd(centered, full_matrices=False)
    moving = [o for o in valid if o.flow_points >= 3 and np.hypot(o.flow_x, o.flow_y) > 0.15]
    if len(moving) >= 3:
        vectors = np.array([[o.flow_x, o.flow_y] for o in moving])
        _, flow_values, flow_axes = np.linalg.svd(vectors, full_matrices=False)
        if flow_values[0] ** 2 / max(float(np.sum(flow_values**2)), 1e-6) >= 0.65:
            values, axes = flow_values, flow_axes
    if values[0] ** 2 / max(float(np.sum(values**2)), 1e-6) < 0.65:
        return Event(reason="自动轨迹：没有清楚的单一接近方向，保留人工复核"), diagnostic, rows
    axis = axes[0]
    peak = int(np.argmax(areas))
    first_center = np.median(centers[:3], axis=0)
    last_center = np.median(centers[-3:], axis=0)
    if (
        np.median(areas[:3]) < areas[peak] * 0.75
        and np.linalg.norm(centers[peak] - first_center) > 2
    ):
        reference = centers[peak] - first_center
        direction_source = "进入观察区的运动"
    elif (
        np.median(areas[-3:]) < areas[peak] * 0.75
        and np.linalg.norm(centers[peak] - last_center) > 2
    ):
        reference = centers[peak] - last_center
        direction_source = "回撤方向的反向"
    else:
        early = centers[: max(4, len(before) // 2)]
        reference = early[-1] - early[0]
        direction_source = "前段主运动，需核对方向"
    if reference @ axis < 0:
        axis = -axis
    if moving:
        entry_flow = sum((np.array([o.flow_x, o.flow_y]) @ axis) * o.area_change for o in moving)
        if abs(entry_flow) > 1:
            if entry_flow < 0:
                axis = -axis
            direction_source = "指腹纹理运动与进入观察区的方向"
    diagnostic.update(direction=axis.tolist(), direction_source=direction_source)
    across = np.array([-axis[1], axis[0]])
    if roi_mask is None:
        roi_mask = np.ones(valid[0].mask.shape, np.uint8)
    measured = {}
    for o, center in zip(valid, centers):
        points = np.column_stack(np.nonzero(o.mask)[::-1])
        projection = points @ axis
        front = float(np.quantile(projection, 0.98))
        leading = points[projection >= front - 4] @ across
        front_points = points[projection >= front - 1]
        ahead = np.rint(front_points + axis * 2).astype(int)
        inside = (
            (ahead[:, 0] >= 0)
            & (ahead[:, 0] < roi_mask.shape[1])
            & (ahead[:, 1] >= 0)
            & (ahead[:, 1] < roi_mask.shape[0])
        )
        outside = ~inside
        outside[inside] = roi_mask[ahead[inside, 1], ahead[inside, 0]] == 0
        measured[o.frame] = {
            "position_px": front,
            "front_clipped": bool(np.mean(outside) >= 0.4),
            "front_width_px": float(np.quantile(leading, 0.95) - np.quantile(leading, 0.05)),
            "x": float(center[0] + origin[0]),
            "y": float(center[1] + origin[1]),
        }
    for row in rows:
        row.update(measured.get(row["frame"], {}))
    travel = float(np.ptp([m["position_px"] for m in measured.values()]))
    diagnostic["travel_px"] = travel
    if travel < 3:
        return Event(reason="自动轨迹：指腹位移不足，无法判断触摸转折"), diagnostic, rows
    pre, post = max(5, round(fps * 0.035)), max(3, round(fps * 0.02))
    candidates = []
    for frame in sorted(measured):
        if frame > latest_press:
            continue
        before_count, after_count = 0, 0
        while before_count < pre and frame - before_count - 1 in measured:
            before_count += 1
        while after_count < post and frame + after_count + 1 in measured:
            after_count += 1
        if before_count < 2 or after_count < 3:
            continue
        t = np.arange(-before_count, after_count + 1, dtype=float)
        y = np.array([measured[frame + int(k)]["position_px"] for k in t])
        design = np.column_stack((np.ones(len(t)), np.minimum(t, 0), np.maximum(t, 0)))
        fit = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = float(np.sqrt(np.mean((design @ fit - y) ** 2)))
        noise = max(0.35, float(np.median(np.abs(np.diff(y, n=2)))) * 0.5)
        if (
            fit[1] < max(0.2, noise * 0.4)
            or abs(fit[2]) > max(0.3, fit[1] * 0.35)
            or fit[1] * before_count < 3
        ):
            continue
        if residual > max(1.2, noise * 2):
            continue
        if any(measured[f]["front_clipped"] for f in range(frame, frame + min(3, after_count) + 1)):
            diagnostic["rejected_clipping"].append(frame)
            continue
        later_flow = [
            o.flow_x * axis[0] + o.flow_y * axis[1]
            for o in valid
            if frame < o.frame <= frame + after_count and o.flow_points >= 3
        ]
        if len(later_flow) >= 3 and np.median(later_flow) > max(0.3, fit[1] * 0.35):
            diagnostic["rejected_moving"].append(frame)
            continue
        position = measured[frame]["position_px"]
        # A pause followed by substantial further approach is not the final action boundary.
        later = [
            measured[f]["position_px"]
            for f in sorted(measured)
            if frame + after_count < f <= latest_press
        ]
        if later and max(later) - position > max(4, travel * 0.25, noise * 3):
            diagnostic["rejected_pauses"].append(frame)
            continue
        old_width = float(
            np.median([measured[f]["front_width_px"] for f in range(frame - before_count, frame)])
        )
        new_width = float(
            np.median(
                [measured[f]["front_width_px"] for f in range(frame, frame + after_count + 1)]
            )
        )
        candidates.append(
            {
                "frame": frame,
                "confirmed_at": frame + after_count,
                "before_speed": float(fit[1]),
                "after_speed": float(fit[2]),
                "fit_error_px": residual,
                "noise_px": noise,
                "deformation": new_width - old_width > max(1.5, old_width * 0.12),
                "front_width_before": old_width,
                "front_width_after": new_width,
            }
        )
    diagnostic["candidates"] = candidates
    if not candidates:
        reason = "未找到连续接近后保持的明确转折；可能存在遮挡、快触或观察区不足"
        if diagnostic["rejected_clipping"]:
            reason = "指腹前缘被观察区边缘截断，无法确认停止；请扩大触摸区，保留完整指腹与接近过程"
        elif diagnostic["rejected_moving"]:
            reason = "轮廓看似停止，但可靠的指腹纹理仍在接近，保留缺失供复核"
        return (
            Event(reason="自动轨迹：" + reason),
            diagnostic,
            rows,
        )
    best = min(candidates, key=lambda c: c["fit_error_px"] / max(c["before_speed"], 0.2))
    nearby = [
        c
        for c in candidates
        if abs(c["frame"] - best["frame"]) <= max(2, round(fps * 0.025))
        and c["fit_error_px"] <= best["fit_error_px"] + best["noise_px"]
    ]
    interval = [min(c["frame"] for c in nearby), max(c["frame"] for c in nearby)]
    diagnostic.update(selected_frame=best["frame"], review_interval=interval)
    support = "，伴随局部展宽" if best["deformation"] else ""
    reason = f"自动方向的运动转折候选{best['frame']}帧{support}；请复核{interval[0]}～{interval[1]}帧。未直接确认物理接触"
    return Event(best["frame"], "review", reason), diagnostic, rows


def motion_preview(tracker: MotionTracker, decision: dict) -> np.ndarray:
    image = tracker.reference.copy()
    selected = tracker.reference_mask > 0
    image[selected] = (
        image[selected].astype(float) * 0.55 + np.array([50, 200, 50]) * 0.45
    ).astype(np.uint8)
    center = np.rint(tracker.reference_center).astype(int)
    if decision.get("direction") is not None:
        axis = np.array(decision["direction"])
        length = max(8, min(image.shape[:2]) * 0.35)
        start = tuple(np.rint(center - axis * length / 2).astype(int))
        end = tuple(np.rint(center + axis * length / 2).astype(int))
        cv2.arrowedLine(image, start, end, (0, 100, 255), 1, tipLength=0.25)
    return image
