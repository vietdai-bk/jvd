"""Main pipeline for junction violation detection"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

from jvd.core.models import DetectionEvent, ViolationRecord, TrackState, VehicleClass, BoundingBox
from jvd.core.tracker import VehicleTrackerManager
from jvd.core.analyzer import ViolationAnalyzer
from jvd.utils.fps_counter import FPSCounter


class JunctionDetectorPipeline:
    """Main pipeline orchestrating detection, tracking, and violation analysis"""

    def __init__(self, config: dict):
        self.config = config
        self.model = YOLO(config['model']['path'])

        self.tracker_mgr = VehicleTrackerManager(
            history_len=config['tracking']['history_len'],
            max_stale_frames=config['tracking']['max_stale_frames'],
            smoothing_alpha=config['tracking']['smoothing_alpha']
        )
        self.analyzer = ViolationAnalyzer(
            roi_normalized=config['roi']['points'],
            config=config['violation']
        )
        self.frame_id = 0
        self.fps_counter = FPSCounter(update_interval=config['display']['update_interval_fps'])

    def process_frame(self, frame: np.ndarray, timestamp_sec: Optional[float] = None) -> Tuple[
        List[DetectionEvent], List[ViolationRecord], Dict[int, TrackState], float
    ]:
        self.frame_id += 1
        timestamp = timestamp_sec if timestamp_sec is not None else time.time()
        height, width = frame.shape[:2]

        current_fps = self.fps_counter.update()

        raw_events = self._run_detection(frame, timestamp)
        events = self.tracker_mgr.update(raw_events, self.frame_id)
        violations, states = self.analyzer.analyze(events, self.tracker_mgr, width, height)

        if self.frame_id % 60 == 0:
            self.tracker_mgr.cleanup_stale(self.frame_id)

        return events, violations, states, current_fps

    def _run_detection(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Run YOLO detection and tracking on frame"""
        model_config = self.config['model']

        conf_thresh = model_config.get('conf_threshold', 0.3)

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=model_config['classes'],
            verbose=False,
            conf=conf_thresh,
            iou=model_config.get('iou_threshold', 0.5)
        )

        events = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            has_ids = boxes.id is not None

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                track_id = int(boxes.id[i]) if has_ids else None

                events.append(DetectionEvent(
                    frame_id=self.frame_id,
                    timestamp=timestamp,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    track_id=track_id,
                    confidence=confidence,
                    class_label=VehicleClass.from_coco_id(class_id)
                ))

        return events

    def get_roi_points_pixel(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Get ROI points in pixel coordinates"""
        roi_points = self.config['roi']['points']
        return np.array([[int(x * frame_width), int(y * frame_height)]
                        for x, y in roi_points], dtype=np.int32)
