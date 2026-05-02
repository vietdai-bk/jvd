"""Data models for the junction violation detection system"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional


class VehicleClass(Enum):
    """Vehicle classes supported by the system"""
    CAR = auto()
    MOTORBIKE = auto()
    TRUCK = auto()
    BUS = auto()
    UNKNOWN = auto()

    @classmethod
    def from_coco_id(cls, coco_id: int):
        mapping = {2: cls.CAR, 3: cls.MOTORBIKE, 5: cls.BUS, 7: cls.TRUCK}
        return mapping.get(coco_id, cls.UNKNOWN)

    @property
    def color(self) -> Tuple[int, int, int]:
        colors = {
            VehicleClass.CAR: (255, 100, 100),
            VehicleClass.MOTORBIKE: (100, 255, 100),
            VehicleClass.TRUCK: (100, 100, 255),
            VehicleClass.BUS: (255, 255, 100),
            VehicleClass.UNKNOWN: (200, 200, 200)
        }
        return colors.get(self, (200, 200, 200))

    @property
    def display_name(self) -> str:
        names = {
            VehicleClass.CAR: "CAR",
            VehicleClass.MOTORBIKE: "MOTO",
            VehicleClass.TRUCK: "TRUCK",
            VehicleClass.BUS: "BUS",
            VehicleClass.UNKNOWN: "UNK"
        }
        return names.get(self, "UNK")


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_int(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass(frozen=True)
class DetectionEvent:
    frame_id: int
    timestamp: float
    bbox: BoundingBox
    track_id: Optional[int] = None
    class_label: VehicleClass = VehicleClass.UNKNOWN
    confidence: float = 0.0


@dataclass(frozen=True)
class ViolationRecord:
    event: DetectionEvent
    dwell_time_seconds: float
    violation_type: str = "junction_stop_violation"


@dataclass
class TrackState:
    first_stop_time: Optional[float] = None
    is_blocked: bool = False
    violation_triggered: bool = False
    is_inside: bool = False
    stop_buffer: int = 0
    last_seen_frame: int = 0
    last_velocity: float = 0.0
    last_dwell_time: float = 0.0
    violation_saved: bool = False