"""FPS counter utility"""

import time


class FPSCounter:
    """Simple FPS counter for performance monitoring"""
    
    def __init__(self, update_interval: int = 30):
        self.update_interval = update_interval
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0
    
    def update(self) -> float:
        """Update FPS counter and return current FPS"""
        self.frame_count += 1
        
        if self.frame_count % self.update_interval == 0:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.current_fps = self.update_interval / elapsed
            self.start_time = time.time()
        
        return self.current_fps
    
    @property
    def fps(self) -> float:
        """Get current FPS"""
        return self.current_fps