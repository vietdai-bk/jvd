"""
Core module containing main detection and tracking logic
"""

# Import models first (no dependencies)
from jvd.core.models import (
    VehicleClass,
    BoundingBox,
    DetectionEvent,
    ViolationRecord,
    TrackState
)

# Then import other modules
from jvd.core.tracker import VehicleTrackerManager
from jvd.core.analyzer import ViolationAnalyzer
from jvd.core.pipeline import JunctionDetectorPipeline

__all__ = [
    'VehicleClass',
    'BoundingBox',
    'DetectionEvent',
    'ViolationRecord',
    'TrackState',
    'VehicleTrackerManager',
    'ViolationAnalyzer',
    'JunctionDetectorPipeline'
]