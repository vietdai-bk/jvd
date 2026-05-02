"""Interactive ROI editor - vẽ vùng ROI trực tiếp lên frame video"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Optional


class ROIEditor:
    """
    Cho phép người dùng vẽ vùng ROI bằng cách click chuột lên frame video.
    Nhấn Enter/Space để xác nhận, C để xóa, ESC để hủy (dùng ROI cũ).
    """

    WINDOW = "ROI Editor - Click to add points | Enter=Confirm | C=Clear | ESC=Cancel"

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._points: List[Tuple[int, int]] = []
        self._frame: Optional[np.ndarray] = None
        self._display: Optional[np.ndarray] = None
        self._confirmed = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._points.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Xóa điểm cuối
            if self._points:
                self._points.pop()
                self._redraw()
        elif event == cv2.EVENT_MOUSEMOVE:
            self._redraw(cursor=(x, y))

    def _redraw(self, cursor: Optional[Tuple[int, int]] = None):
        self._display = self._frame.copy()
        h, w = self._display.shape[:2]

        # Hướng dẫn
        instructions = [
            "Left Click: Add point | Right Click: Remove last | Enter/Space: Confirm | C: Clear | ESC: Cancel",
            f"Points: {len(self._points)} (min 3 required)"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(self._display, text, (10, 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(self._display, text, (10, 25 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if not self._points:
            return

        pts = self._points[:]
        if cursor and len(pts) >= 1:
            # Đường preview tới cursor
            cv2.line(self._display, pts[-1], cursor, (200, 200, 200), 1, cv2.LINE_AA)
            if len(pts) >= 2:
                # Đường preview đóng polygon
                cv2.line(self._display, pts[0], cursor, (100, 100, 100), 1, cv2.LINE_AA)

        # Vẽ polygon filled nếu đủ điểm
        if len(pts) >= 3:
            overlay = self._display.copy()
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 255))
            cv2.addWeighted(overlay, 0.25, self._display, 0.75, 0, self._display)
            cv2.polylines(self._display, [pts_arr], True, (0, 220, 220), 2, cv2.LINE_AA)

        # Vẽ các cạnh
        for i in range(len(pts) - 1):
            cv2.line(self._display, pts[i], pts[i + 1], (0, 255, 255), 2, cv2.LINE_AA)

        # Vẽ điểm
        for i, (px, py) in enumerate(pts):
            color = (0, 255, 0) if i == 0 else (0, 200, 255)
            cv2.circle(self._display, (px, py), 6, color, -1)
            cv2.circle(self._display, (px, py), 6, (255, 255, 255), 1)
            cv2.putText(self._display, str(i + 1), (px + 8, py - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run(self, video_path: str, existing_roi: List[List[float]]) -> Optional[List[List[float]]]:
        """
        Mở video, lấy frame đầu để vẽ ROI.
        Returns normalized points [[x,y],...] hoặc None nếu hủy.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Không đọc được frame từ video")

        h, w = frame.shape[:2]

        # Scale để hiển thị không quá to
        max_disp_w, max_disp_h = 1280, 720
        scale = min(max_disp_w / w, max_disp_h / h, 1.0)
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        self._frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # Chuyển ROI cũ sang pixel (theo kích thước display)
        self._points = [
            (int(nx * disp_w), int(ny * disp_h))
            for nx, ny in existing_roi
        ]

        self._redraw()

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, disp_w, disp_h)
        cv2.setMouseCallback(self.WINDOW, self._mouse_callback)

        while True:
            if self._display is not None:
                cv2.imshow(self.WINDOW, self._display)

            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):  # Enter hoặc Space
                if len(self._points) >= 3:
                    self._confirmed = True
                    break
                else:
                    # Flash thông báo
                    tmp = self._display.copy() if self._display is not None else self._frame.copy()
                    cv2.putText(tmp, "Need at least 3 points!", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(self.WINDOW, tmp)
                    cv2.waitKey(800)

            elif key == ord('c') or key == ord('C'):
                self._points = []
                self._redraw()

            elif key == 27:  # ESC
                break

        cv2.destroyWindow(self.WINDOW)

        if not self._confirmed:
            return None

        # Normalize về [0,1]
        return [[round(px / disp_w, 4), round(py / disp_h, 4)]
                for px, py in self._points]

    @staticmethod
    def save_roi_to_config(config_path: Path, roi_points: List[List[float]]):
        """Ghi ROI mới vào config.yaml, giữ nguyên các key khác"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['roi']['points'] = roi_points
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
