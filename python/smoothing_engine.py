import numpy as np
import time

class AdaptiveSmoother:
    """
    Advanced smoothing engine for V2.
    - Adaptive alpha based on cursor velocity.
    - Jitter suppression for static hands.
    - Small dead-zone for fine stability.
    """
    
    def __init__(self, screen_w=1920, screen_h=1080):
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        
        self.min_alpha = 0.05  # Slow movement (heavy smoothing)
        self.max_alpha = 0.8   # Fast movement (low latency)
        self.velocity_threshold = 100.0 # Pixels/sec
        
        self.dead_zone = 2 # Pixels
        self.last_time = time.time()

    def smooth(self, tx, ty):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: return self.curr_x, self.curr_y
        
        # 1. Calculate Velocity
        dist = np.sqrt((tx - self.prev_x)**2 + (ty - self.prev_y)**2)
        velocity = dist / dt
        
        # 2. Adaptive Alpha
        # Higher velocity -> Higher alpha (less smoothing, more responsive)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * \
                np.clip(velocity / self.velocity_threshold, 0, 1)
        
        # 3. Apply Dead-zone
        if dist < self.dead_zone:
            # Don't move if it's just tiny jitter
            self.last_time = now
            return int(self.curr_x), int(self.curr_y)
            
        # 4. Exponential Smoothing
        self.curr_x = self.curr_x + alpha * (tx - self.curr_x)
        self.curr_y = self.curr_y + alpha * (ty - self.curr_y)
        
        self.prev_x, self.prev_y = tx, ty
        self.last_time = now
        
        return int(self.curr_x), int(self.curr_y)

    def reset(self, x, y):
        self.prev_x, self.prev_y = x, y
        self.curr_x, self.curr_y = x, y
        self.last_time = time.time()
