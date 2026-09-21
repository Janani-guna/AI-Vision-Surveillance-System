# src/core/tracker.py
"""Lightweight object tracker using the supervision ByteTrack implementation."""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    import supervision as sv
    _HAS_SUPERVISION = True
except ImportError:
    _HAS_SUPERVISION = False
    logger.warning("supervision not installed — tracker will pass-through detections without tracking")


@dataclass
class TrackedObject:
    """A detection with a persistent track ID across frames."""
    track_id: int
    label: str                 # e.g. "face:22102027", "hazard:knife", "unknown_face"
    bbox: tuple                # (x1, y1, x2, y2)
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    alert_sent: bool = False   # prevents duplicate alerts per track
    metadata: dict = field(default_factory=dict)


class ObjectTracker:
    """Wraps supervision.ByteTrack to give persistent IDs to detections."""

    def __init__(self, frame_rate: int = 30, lost_buffer: int = 30):
        if _HAS_SUPERVISION:
            self._tracker = sv.ByteTrack(
                track_activation_threshold=0.4,
                lost_track_buffer=lost_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=frame_rate,
            )
        else:
            self._tracker = None
        self.tracks: dict[int, TrackedObject] = {}
        self._next_id = 0  # fallback counter when supervision unavailable
        self._fallback_boxes: dict[int, tuple] = {}

    def update(self, detections: list[dict], frame_id: int) -> list[TrackedObject]:
        """Update tracker with new detections and return tracked objects.

        Args:
            detections: [{'label': str, 'conf': float, 'box': (x1,y1,x2,y2)}]
            frame_id: monotonically increasing frame counter
        """
        if not detections:
            if self._tracker:
                self._tracker.update_with_detections(sv.Detections.empty())
            return []

        labels = [d["label"] for d in detections]
        boxes = np.array([d["box"] for d in detections], dtype=np.float32)
        confs = np.array([d["conf"] for d in detections], dtype=np.float32)

        if self._tracker is not None:
            sv_dets = sv.Detections(
                xyxy=boxes,
                confidence=confs,
                class_id=np.array([hash(l) % 10000 for l in labels], dtype=int),
            )
            tracked = self._tracker.update_with_detections(sv_dets)
            track_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        else:
            # Keep persistence useful when supervision is unavailable by matching
            # each detection to the closest recent box of the same label.
            track_ids = []
            used_ids: set[int] = set()
            for index, detection in enumerate(detections):
                best_id = None
                best_iou = 0.0
                for candidate_id, candidate_box in self._fallback_boxes.items():
                    if candidate_id in used_ids:
                        continue
                    candidate = self.tracks.get(candidate_id)
                    if candidate and candidate.label == detection["label"]:
                        overlap = self._iou(candidate_box, detection["box"])
                        if overlap > best_iou:
                            best_iou = overlap
                            best_id = candidate_id
                if best_id is None or best_iou < 0.2:
                    best_id = self._next_id
                    self._next_id += 1
                used_ids.add(best_id)
                self._fallback_boxes[best_id] = tuple(detection["box"])
                track_ids.append(best_id)

        results = []
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid not in self.tracks:
                self.tracks[tid] = TrackedObject(
                    track_id=tid,
                    label=labels[min(i, len(labels) - 1)],
                    bbox=tuple(boxes[i].astype(int)) if i < len(boxes) else (0, 0, 0, 0),
                    confidence=float(confs[i]) if i < len(confs) else 0.0,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                )
            else:
                t = self.tracks[tid]
                t.last_seen_frame = frame_id
                if i < len(boxes):
                    t.bbox = tuple(boxes[i].astype(int))
                if i < len(confs):
                    t.confidence = float(confs[i])
            results.append(self.tracks[tid])

        # Prune tracks not seen for a long time
        stale = [k for k, v in self.tracks.items()
                 if frame_id - v.last_seen_frame > 300]
        for k in stale:
            del self.tracks[k]
            self._fallback_boxes.pop(k, None)

        return results

    @staticmethod
    def _iou(first: tuple, second: tuple) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @property
    def active_count(self) -> int:
        return len(self.tracks)

    def should_alert(self, track: TrackedObject, min_frames: int = 5) -> bool:
        """Only alert once per track, and only after min_frames of persistence."""
        if track.alert_sent:
            return False
        duration = track.last_seen_frame - track.first_seen_frame
        if duration >= min_frames:
            track.alert_sent = True
            return True
        return False
