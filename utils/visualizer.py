"""Visualization utilities for the detection system"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from jvd.core.models import DetectionEvent, TrackState


class Visualizer:
    """Handles all visualization tasks"""

    def __init__(self, config: dict):
        self.config = config
        self.max_width = config['display']['max_width']
        self.max_height = config['display']['max_height']
        self.show_legend = config['display']['show_legend']
        self.resize_scale = config['display'].get('resize_scale', 0.5)

    def draw_roi(self, frame: np.ndarray, roi_points: np.ndarray) -> np.ndarray:
        cv2.polylines(frame, [roi_points], True, (0, 255, 255), 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_points], (0, 255, 255))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        return frame

    def draw_vehicle(self, frame: np.ndarray, event: DetectionEvent,
                     state: Optional[TrackState], is_violating: bool,
                     dwell_time: float, plate_text: str = "") -> np.ndarray:
        """Draw a single vehicle with all information including plate number."""
        x1, y1, x2, y2 = event.bbox.to_int()
        color, thickness = self._get_vehicle_color(event, state, is_violating)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Vẽ tâm bbox (dùng để check ROI)
        cx, cy = event.bbox.center
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

        self._draw_info_text(frame, event, state, is_violating, dwell_time, color, (x1, y1), plate_text)

        # Vẽ biển số nổi bật bên dưới bbox nếu xe đang vi phạm
        if is_violating and plate_text:
            self._draw_plate_overlay(frame, plate_text, (x1, y1, x2, y2))

        return frame

    def _draw_plate_overlay(self, frame: np.ndarray, plate_text: str,
                            bbox: Tuple[int, int, int, int]):
        """Vẽ biển số nổi bật ngay dưới bbox xe vi phạm."""
        x1, y1, x2, y2 = bbox
        label = f"  {plate_text}  "
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.7
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        box_x1 = x1
        box_y1 = y2 + 4
        box_x2 = x1 + tw + 4
        box_y2 = y2 + th + baseline + 10

        # Nền đỏ đậm
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 200), -1)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (box_x1 + 2, box_y2 - baseline - 2),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _get_vehicle_color(self, event: DetectionEvent, state: Optional[TrackState],
                           is_violating: bool) -> Tuple[Tuple[int, int, int], int]:
        if is_violating:
            return (0, 0, 255), 3
        elif state and state.is_inside and state.first_stop_time:
            if state.is_blocked:
                return (255, 165, 0), 2
            else:
                return (0, 165, 255), 2
        else:
            return event.class_label.color, 2

    def _draw_info_text(self, frame: np.ndarray, event: DetectionEvent,
                        state: Optional[TrackState], is_violating: bool,
                        dwell_time: float, color: Tuple[int, int, int],
                        position: Tuple[int, int], plate_text: str = ""):
        x1, y1 = position
        lines = self._prepare_info_lines(event, state, is_violating, dwell_time, plate_text)

        max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                         for line in lines])
        total_height = len(lines) * 20 + 5

        text_bg_y1 = y1 - total_height - 5
        if text_bg_y1 < 0:
            text_bg_y1 = y1 + int(event.bbox.height) + 5

        cv2.rectangle(frame, (x1, text_bg_y1), (x1 + max_width + 10, text_bg_y1 + total_height),
                      color, -1)

        for i, line in enumerate(lines):
            y_pos = text_bg_y1 + 20 + i * 20
            cv2.putText(frame, line, (x1 + 5, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _prepare_info_lines(self, event: DetectionEvent, state: Optional[TrackState],
                            is_violating: bool, dwell_time: float,
                            plate_text: str = "") -> List[str]:
        lines = [f"ID:{event.track_id}|{event.class_label.display_name}"]

        if state and state.last_velocity > 0:
            lines.append(f"V:{state.last_velocity:.1f}px/s")
        else:
            lines.append("V:0px/s")

        if state and state.is_inside and state.first_stop_time:
            if is_violating:
                lines.append(f"VIOLATION|{dwell_time:.1f}s")
            elif state.is_blocked:
                lines.append(f"BLOCKED|{dwell_time:.1f}s")
            else:
                lines.append(f"STOPPING|{dwell_time:.1f}s")

        # Biển số trong info box (nếu có)
        if plate_text:
            lines.append(f"BSX:{plate_text}")

        return lines

    def draw_debug_info(self, frame: np.ndarray, frame_count: int, fps: float,
                        num_vehicles: int, num_violations: int,
                        velocity_thresh: float, violation_time: float,
                        processing_time: float) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        info_lines = [
            f"Frame: {frame_count}",
            f"FPS: {fps:.1f}",
            f"Vehicles: {num_vehicles}",
            f"Violations: {num_violations}",
            f"Vel Thresh: {velocity_thresh:.1f} px/s",
            f"Violation Time: {violation_time:.1f}s",
            f"Proc Time: {processing_time:.1f}ms"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return frame

    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        if not self.show_legend:
            return frame

        height, width = frame.shape[:2]
        legend_x = width - 180
        legend_y = 10

        cv2.rectangle(frame, (legend_x - 5, legend_y - 5), (width - 10, legend_y + 95),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x - 5, legend_y - 5), (width - 10, legend_y + 95),
                      (255, 255, 255), 1)

        legends = [
            ("Violation", (0, 0, 255)),
            ("Stopping", (0, 165, 255)),
            ("Blocked", (255, 165, 0)),
            ("Moving", (255, 0, 0))
        ]

        for i, (text, color) in enumerate(legends):
            y_pos = legend_y + 15 + i * 20
            cv2.rectangle(frame, (legend_x, y_pos - 10), (legend_x + 15, y_pos), color, -1)
            cv2.putText(frame, text, (legend_x + 20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_title(self, frame: np.ndarray,
                   title: str = "JUNCTION VIOLATION DETECTION SYSTEM") -> np.ndarray:
        height, width = frame.shape[:2]
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x = (width - text_size[0]) // 2
        cv2.putText(frame, title, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        return frame

    def resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height > self.max_height or width > self.max_width:
            scale = min(self.max_width / width, self.max_height / height)
            return cv2.resize(frame,
                              (int(width * scale), int(height * scale)),
                              interpolation=cv2.INTER_AREA)
        elif self.resize_scale < 1.0:
            return cv2.resize(frame,
                              (int(width * self.resize_scale), int(height * self.resize_scale)),
                              interpolation=cv2.INTER_AREA)
        return frame