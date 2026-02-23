import time

class GestureState:
    IDLE = "idle"
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    RELEASED = "released"
    COOLDOWN = "cooldown"

class GestureStateMachine:
    """Manages lifecycle of a single gesture type."""
    
    def __init__(self, name, confirm_frames=6, cooldown_ms=300):
        self.name = name
        self.state = GestureState.IDLE
        self.confirm_frames = confirm_frames
        self.cooldown_ms = cooldown_ms
        
        self.frame_counter = 0
        self.last_state_change = time.time()
        
    def update(self, is_detected):
        now = time.time()
        
        if self.state == GestureState.IDLE:
            if is_detected:
                self._change_state(GestureState.DETECTED)
                self.frame_counter = 1
                
        elif self.state == GestureState.DETECTED:
            if is_detected:
                self.frame_counter += 1
                if self.frame_counter >= self.confirm_frames:
                    self._change_state(GestureState.CONFIRMED)
            else:
                self._change_state(GestureState.IDLE)
                
        elif self.state == GestureState.CONFIRMED:
            # Immediate transition to active for responsiveness after confirmation
            self._change_state(GestureState.ACTIVE)
            
        elif self.state == GestureState.ACTIVE:
            if not is_detected:
                self._change_state(GestureState.RELEASED)
                
        elif self.state == GestureState.RELEASED:
            self._change_state(GestureState.COOLDOWN)
            
        elif self.state == GestureState.COOLDOWN:
            if now - self.last_state_change > (self.cooldown_ms / 1000.0):
                self._change_state(GestureState.IDLE)
                
        return self.state

    def _change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.last_state_change = time.time()
            if new_state == GestureState.IDLE:
                self.frame_counter = 0

class GlobalStateMachine:
    """Orchestrates multiple gesture states and ensures only one 'ACTIVE' gesture."""
    
    def __init__(self, gesture_names):
        self.machines = {name: GestureStateMachine(name) for name in gesture_names}
        self.active_gesture = None
        
    def update_all(self, detections):
        """
        detections: dict {name: bool}
        returns: (active_name, state)
        """
        results = {}
        for name, machine in self.machines.items():
            # If another gesture is already ACTIVE, others are forced to IDLE/COOLDOWN
            # unless it's a higher priority, but resolver handles that.
            # Here we just update each machine.
            detect_val = detections.get(name, False)
            state = machine.update(detect_val)
            results[name] = state
            
        # Determine the current globally active gesture
        # (Conflict resolver will be used after this in the engine)
        active = [n for n, s in results.items() if s == GestureState.ACTIVE]
        
        return results
