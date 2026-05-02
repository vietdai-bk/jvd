"""
CSV logger cho vi phạm.
Mỗi vi phạm ghi 1 dòng: thời gian, loại xe, biển số, tọa độ GPS (mock), dwell time.
"""

import csv
import os
import time
from pathlib import Path
from typing import Optional, Tuple
import random


# Mock GPS: trả về tọa độ cố định ± nhiễu nhỏ để giả lập camera cố định
_BASE_GPS = (10.7769, 106.7009)  # Mặc định: TP.HCM — override trong config


def _mock_gps(base_lat: float, base_lon: float) -> Tuple[float, float]:
    """Giả lập GPS với nhiễu nhỏ ±0.0001 độ (~10m)."""
    lat = base_lat + random.uniform(-0.0001, 0.0001)
    lon = base_lon + random.uniform(-0.0001, 0.0001)
    return round(lat, 6), round(lon, 6)


CSV_HEADER = [
    "timestamp",
    "track_id",
    "vehicle_type",
    "plate_number",
    "dwell_time_sec",
    "violation_type",
    "gps_lat",
    "gps_lon",
    "location_name",
    "snapshot_path",
    "video_path",
]


class ViolationLogger:
    """
    Ghi log vi phạm ra file CSV.
    Tự động tạo file nếu chưa có, tự append nếu đã có.
    """

    def __init__(self, config: dict):
        log_cfg = config.get('logging', {})
        output_dir = Path(log_cfg.get('csv_dir', 'logs'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Tên file theo ngày để dễ quản lý
        date_str = time.strftime("%Y%m%d")
        self._csv_path = output_dir / f"violations_{date_str}.csv"

        # GPS mock config
        gps_cfg = config.get('location', {})
        self._base_lat = gps_cfg.get('lat', _BASE_GPS[0])
        self._base_lon = gps_cfg.get('lon', _BASE_GPS[1])
        self._location_name = gps_cfg.get('name', 'Unknown Junction')

        # Tạo file + header nếu chưa có
        if not self._csv_path.exists():
            with open(self._csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

        print(f"[ViolationLogger] Log CSV: {self._csv_path.resolve()}")

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    def log(self,
            track_id: int,
            vehicle_type: str,
            plate_number: str,
            dwell_time_sec: float,
            violation_type: str = "junction_stop_violation",
            snapshot_path: str = "",
            video_path: str = "") -> None:
        """Ghi 1 dòng vi phạm vào CSV."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lat, lon = _mock_gps(self._base_lat, self._base_lon)
        plate_display = plate_number if plate_number else "UNKNOWN"

        row = [
            ts,
            track_id,
            vehicle_type,
            plate_display,
            f"{dwell_time_sec:.1f}",
            violation_type,
            lat,
            lon,
            self._location_name,
            snapshot_path,
            video_path,
        ]

        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)