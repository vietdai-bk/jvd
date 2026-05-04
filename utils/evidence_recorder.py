"""Evidence recorder - lưu ảnh và video vi phạm (3s trước + 10s sau)"""

import cv2
import os
import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Deque, Tuple
from pathlib import Path


class FrameBuffer:
    """Ring buffer lưu N giây frame gần nhất."""

    def __init__(self, fps: float, pre_seconds: float = 3.0):
        self._fps = max(fps, 1.0)
        maxlen = int(self._fps * pre_seconds) + 5
        self._buf: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray):
        with self._lock:
            self._buf.append(frame.copy())

    def snapshot(self) -> list:
        with self._lock:
            return list(self._buf)


class EvidenceRecorder:
    """
    Khi phát hiện vi phạm, lưu:
    - Ảnh snapshot tại thời điểm vi phạm
    - Video = 3s trước (từ buffer) + 10s sau (capture tiếp)
    """

    def __init__(self, output_dir: str, fps: float,
                 pre_seconds: float = 3.0, post_seconds: float = 10.0):
        self.output_dir = Path(output_dir)
        self.fps = max(fps, 1.0)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)

        self.frame_buffer = FrameBuffer(fps, pre_seconds)
        self._active_recorders: dict = {}
        self._lock = threading.Lock()

    def push_frame(self, frame: np.ndarray):
        """Gọi mỗi frame để cập nhật ring buffer và feed post-recorders."""
        self.frame_buffer.push(frame)
        with self._lock:
            finished = []
            for tid, recorder in self._active_recorders.items():
                recorder.feed(frame)
                if recorder.is_done():
                    recorder.finalize()
                    finished.append(tid)
            for tid in finished:
                del self._active_recorders[tid]

    def trigger(self, track_id: int, snapshot_frame: np.ndarray,
                vehicle_label: str = "UNK") -> str:
        """Kích hoạt ghi evidence. Trả về path ảnh đã lưu."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"violation_id{track_id}_{timestamp_str}"

        # Lưu ảnh snapshot
        img_path = self.output_dir / "images" / f"{base_name}.jpg"
        cv2.imwrite(str(img_path), snapshot_frame)

        # Khởi động post-recorder
        vid_path = self.output_dir / "videos" / f"{base_name}.mp4"
        pre_frames = self.frame_buffer.snapshot()
        post_frame_count = int(self.fps * self.post_seconds)

        h, w = snapshot_frame.shape[:2]
        size = (w, h)

        recorder = _PostRecorder(
            path=str(vid_path),
            pre_frames=pre_frames,
            post_frames_needed=post_frame_count,
            fps=self.fps,
            size=size
        )

        with self._lock:
            if track_id not in self._active_recorders:
                self._active_recorders[track_id] = recorder

        return str(img_path)

    def flush_all(self):
        """Finalize tất cả video đang ghi dở khi thoát."""
        with self._lock:
            for recorder in self._active_recorders.values():
                recorder.finalize()
            self._active_recorders.clear()


class _PostRecorder:
    """Ghi pre-frames rồi tiếp tục ghi post-frames."""

    def __init__(self, path: str, pre_frames: list,
                 post_frames_needed: int, fps: float, size: Tuple[int, int]):
        self._path = path
        self._post_needed = post_frames_needed
        self._post_received = 0
        self._done = False
        self._size = size

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)

        if not self._writer.isOpened():
            # Fallback: thử codec khác
            self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

        # Ghi pre-frames
        for f in pre_frames:
            self._write_frame(f)

    def _write_frame(self, frame: np.ndarray):
        """Resize nếu cần rồi ghi."""
        if not self._writer.isOpened():
            return
        fh, fw = frame.shape[:2]
        tw, th = self._size  # target width, height
        if fw != tw or fh != th:
            if tw > 0 and th > 0:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                return
        self._writer.write(frame)

    def feed(self, frame: np.ndarray):
        if self._done:
            return
        self._write_frame(frame)
        self._post_received += 1
        if self._post_received >= self._post_needed:
            self._done = True

    def is_done(self) -> bool:
        return self._done

    def finalize(self):
        if self._writer.isOpened():
            self._writer.release()
        self._done = True