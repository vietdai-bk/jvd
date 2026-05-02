"""Violation detection and analysis"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from .models import DetectionEvent, ViolationRecord, TrackState, BoundingBox
from .tracker import VehicleTrackerManager


class ViolationAnalyzer:
    """Analyzes vehicle behavior and detects violations"""

    def __init__(self, roi_normalized: List[Tuple[float, float]], config: dict):
        self._norm_pts = roi_normalized
        self.states: Dict[int, TrackState] = {}

        self.stop_velocity_thresh = config.get('stop_velocity_thresh', 15.0)
        self.violation_time_sec = config.get('violation_time_sec', 3.0)
        self.block_dist_thresh = config.get('block_dist_thresh', 50.0)
        self.block_hiou_thresh = config.get('block_hiou_thresh', 0.3)
        self.stop_buffer_frames = config.get('stop_buffer_frames', 15)
        self.velocity_lookback = config.get('velocity_calc_lookback', 15)

        self._poly_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _get_polygon(self, width: int, height: int) -> np.ndarray:
        cache_key = (width, height)
        if cache_key not in self._poly_cache:
            self._poly_cache[cache_key] = np.array(
                [[int(nx * width), int(ny * height)] for nx, ny in self._norm_pts],
                dtype=np.int32
            )
        return self._poly_cache[cache_key]

    @staticmethod
    def _horizontal_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        inter_w = max(0.0, min(box1.x2, box2.x2) - max(box1.x1, box2.x1))
        union_w = max(box1.x2, box2.x2) - min(box1.x1, box2.x1)
        return inter_w / union_w if union_w > 0 else 0.0

    def update_params(self, velocity_thresh: float, violation_time: float):
        self.stop_velocity_thresh = velocity_thresh
        self.violation_time_sec = violation_time

    def analyze(self, events: List[DetectionEvent], tracker: VehicleTrackerManager,
                frame_width: int, frame_height: int) -> Tuple[List[ViolationRecord], Dict[int, TrackState]]:
        """Analyze events for violations"""
        # BUG FIX: violations dùng dict keyed by track_id để tránh duplicate records
        violations_dict: Dict[int, ViolationRecord] = {}
        current_stopped_events = []
        poly = self._get_polygon(frame_width, frame_height)

        # First pass: detect stopped vehicles
        for ev in events:
            if ev.track_id is None:
                continue

            state = self._get_or_create_state(ev.track_id, ev.frame_id)
            is_inside = self._check_inside_roi(ev, poly, state)

            if not is_inside:
                self._reset_state_outside_roi(state)
                continue

            self._check_stopped_vehicle(ev, tracker, state, current_stopped_events)

        # Second pass: check blocking and violations
        for ev in current_stopped_events:
            state = self.states[ev.track_id]

            if state.violation_triggered:
                # Cập nhật dwell time cho xe đang vi phạm
                if state.first_stop_time is not None and state.is_inside:
                    dwell = ev.timestamp - state.first_stop_time
                    state.last_dwell_time = dwell
                    violations_dict[ev.track_id] = ViolationRecord(ev, dwell)
                continue

            state.is_blocked = self._check_blocked(ev, current_stopped_events)

            if not state.is_blocked and state.first_stop_time is not None:
                dwell = ev.timestamp - state.first_stop_time
                state.last_dwell_time = dwell
                if dwell >= self.violation_time_sec:
                    state.violation_triggered = True
                    violations_dict[ev.track_id] = ViolationRecord(ev, dwell)

        if events:
            self._cleanup_stale_states(events[0].frame_id)

        return list(violations_dict.values()), self.states

    def _get_or_create_state(self, track_id: int, frame_id: int) -> TrackState:
        if track_id not in self.states:
            self.states[track_id] = TrackState()
        state = self.states[track_id]
        state.last_seen_frame = frame_id
        return state

    def _check_inside_roi(self, ev: DetectionEvent, poly: np.ndarray, state: TrackState) -> bool:
        px, py = ev.bbox.center
        is_inside = cv2.pointPolygonTest(poly, (px, py), False) >= 0
        state.is_inside = is_inside
        return is_inside

    def _reset_state_outside_roi(self, state: TrackState):
        if state.violation_triggered:
            state.violation_triggered = False
        state.first_stop_time = None
        state.is_blocked = False
        state.last_velocity = 0.0

    def _check_stopped_vehicle(self, ev: DetectionEvent, tracker: VehicleTrackerManager,
                               state: TrackState, stopped_events: List):
        history_len = len(tracker.get_history(ev.track_id))

        if history_len >= 5:
            velocity, _, _ = tracker.get_movement_stats(ev.track_id, self.velocity_lookback)
            state.last_velocity = velocity

            if velocity < self.stop_velocity_thresh:
                state.stop_buffer = self.stop_buffer_frames
                stopped_events.append(ev)
                if state.first_stop_time is None and not state.violation_triggered:
                    state.first_stop_time = ev.timestamp
            else:
                if state.stop_buffer > 0:
                    state.stop_buffer -= 1
                    stopped_events.append(ev)
                else:
                    if not state.violation_triggered:
                        state.first_stop_time = None
                        state.is_blocked = False

    def _check_blocked(self, ev: DetectionEvent, stopped_events: List) -> bool:
        for other_ev in stopped_events:
            if other_ev.track_id == ev.track_id:
                continue
            if other_ev.bbox.y2 <= ev.bbox.y1:
                vertical_dist = ev.bbox.y1 - other_ev.bbox.y2
                if vertical_dist < self.block_dist_thresh:
                    h_iou = self._horizontal_iou(ev.bbox, other_ev.bbox)
                    if h_iou > self.block_hiou_thresh:
                        return True
        return False

    def _cleanup_stale_states(self, current_frame: int, max_stale_frames: int = 120):
        stale_keys = [tid for tid, s in self.states.items()
                     if (current_frame - s.last_seen_frame) > max_stale_frames]
        for tid in stale_keys:
            del self.states[tid]