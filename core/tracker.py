"""Vehicle tracking and movement analysis"""

from collections import deque
from typing import List, Dict, Tuple, Optional
from dataclasses import replace

from .models import DetectionEvent, BoundingBox


class VehicleTrackerManager:
    """Manages vehicle tracking history and movement analysis"""
    
    def __init__(self, history_len: int = 30, max_stale_frames: int = 120, 
                 smoothing_alpha: float = 0.15):
        self.history_len = history_len
        self.max_stale_frames = max_stale_frames
        self.smoothing_alpha = smoothing_alpha
        self._histories: Dict[int, deque] = {}
        self._last_seen: Dict[int, int] = {}
        self._last_boxes: Dict[int, BoundingBox] = {}
        self._velocity_cache: Dict[int, float] = {}
        
    def update(self, events: List[DetectionEvent], frame_id: int) -> List[DetectionEvent]:
        """Update tracker with new detections and return smoothed events"""
        smoothed_events = []
        
        for ev in events:
            if ev.track_id is None:
                smoothed_events.append(ev)
                continue

            tid = ev.track_id
            
            # Get current velocity for adaptive smoothing
            current_velocity = self._get_current_velocity(tid)
            self._velocity_cache[tid] = current_velocity
            
            # Apply adaptive smoothing based on velocity
            if tid in self._last_boxes:
                adaptive_alpha = self._get_adaptive_alpha(current_velocity)
                ev = self._smooth_bbox(ev, self._last_boxes[tid], adaptive_alpha)
            
            self._last_boxes[tid] = ev.bbox
            smoothed_events.append(ev)
            
            # Store position history
            self._update_history(tid, ev)
            self._last_seen[tid] = frame_id

        return smoothed_events
    
    def _get_current_velocity(self, track_id: int) -> float:
        """Get current velocity for a track, returns 0 if not enough data"""
        hist = self._histories.get(track_id)
        if hist is None or len(hist) < 5:
            return 0.0
        
        pts = list(hist)
        lookback = min(3, len(pts) - 1)
        
        curr_x, curr_y, curr_t = pts[-1]
        prev_x, prev_y, prev_t = pts[-lookback] if lookback > 0 else pts[0]
        
        dt = curr_t - prev_t
        if dt < 0.001:
            return 0.0
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        distance = (dx**2 + dy**2)**0.5
        
        return distance / dt
    
    def _get_adaptive_alpha(self, velocity: float) -> float:
        """Get adaptive smoothing alpha based on velocity. Higher alpha = less smoothing"""
        if velocity >= 80.0:  # Fast movement: disable smoothing completely
            return 1.0
        elif velocity >= 40.0:  # Medium movement: high responsiveness
            ratio = (velocity - 40.0) / 40.0
            return 0.7 + ratio * 0.3
        elif velocity > 0:  # Slow movement: moderately responsive
            ratio = velocity / 40.0
            return self.smoothing_alpha + ratio * (0.7 - self.smoothing_alpha)
        else:  # Stopped: use base alpha
            return self.smoothing_alpha
    
    def _smooth_bbox(self, current: DetectionEvent, prev: BoundingBox, alpha: float) -> DetectionEvent:
        """Apply exponential smoothing to bounding box coordinates"""
        smoothed_bbox = BoundingBox(
            x1=alpha * current.bbox.x1 + (1 - alpha) * prev.x1,
            y1=alpha * current.bbox.y1 + (1 - alpha) * prev.y1,
            x2=alpha * current.bbox.x2 + (1 - alpha) * prev.x2,
            y2=alpha * current.bbox.y2 + (1 - alpha) * prev.y2
        )
        return replace(current, bbox=smoothed_bbox)
    
    def _update_history(self, track_id: int, event: DetectionEvent):
        """Update position history for a track"""
        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self.history_len)
        
        bx, by = event.bbox.bottom_center
        self._histories[track_id].append((bx, by, event.timestamp))
    
    def get_movement_stats(self, track_id: int, lookback: int = 15) -> Tuple[float, float, float]:
        """
        Calculate movement statistics for a track
        Returns: (velocity_px_per_sec, distance, time_delta)
        """
        hist = self._histories.get(track_id)
        if hist is None or len(hist) < 5:
            return 0.0, 0.0, 0.0

        pts = list(hist)
        lookback = min(lookback, len(pts) - 1)
        
        curr_x, curr_y, curr_t = pts[-1]
        prev_x, prev_y, prev_t = pts[-lookback]
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        dt = curr_t - prev_t
        
        if dt < 0.001:
            return 0.0, 0.0, 0.0
        
        distance = (dx**2 + dy**2)**0.5
        velocity = distance / dt
        
        return velocity, distance, dt
    
    def get_history(self, track_id: int) -> List[Tuple[float, float, float]]:
        """Get position history for a track"""
        return list(self._histories.get(track_id, []))
    
    def get_velocity(self, track_id: int) -> float:
        """Get cached velocity for a track"""
        return self._velocity_cache.get(track_id, 0.0)
    
    def cleanup_stale(self, current_frame_id: int):
        """Remove stale tracks"""
        stale = [tid for tid, last in self._last_seen.items() 
                if (current_frame_id - last) > self.max_stale_frames]
        for tid in stale:
            self._histories.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._last_boxes.pop(tid, None)
            self._velocity_cache.pop(tid, None)