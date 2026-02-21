import math
import time

class Smoother:
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        # 1Euro Filter parameters
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = 0
        self.dy_prev = 0
        self.last_time = None

    def _low_pass_filter(self, x, x_prev, alpha):
        return alpha * x + (1 - alpha) * x_prev

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def smooth(self, x, y):
        current_time = time.time()
        if self.x_prev is None:
            self.x_prev, self.y_prev = x, y
            self.last_time = current_time
            return int(x), int(y)

        dt = current_time - self.last_time
        if dt <= 0: return int(self.x_prev), int(self.y_prev)
        self.last_time = current_time

        # Filter the derivative (speed) to estimate jitter
        dx = (x - self.x_prev) / dt
        dy = (y - self.y_prev) / dt
        
        alpha_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = self._low_pass_filter(dx, self.dx_prev, alpha_d)
        self.dy_prev = self._low_pass_filter(dy, self.dy_prev, alpha_d)
        
        # Calculate adaptive cutoff frequency based on speed
        speed = math.sqrt(self.dx_prev**2 + self.dy_prev**2)
        cutoff = self.min_cutoff + self.beta * speed
        
        # Filter the signal
        alpha = self._alpha(cutoff, dt)
        self.x_prev = self._low_pass_filter(x, self.x_prev, alpha)
        self.y_prev = self._low_pass_filter(y, self.y_prev, alpha)

        return int(self.x_prev), int(self.y_prev)