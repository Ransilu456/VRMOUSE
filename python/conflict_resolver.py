class ConflictResolver:
    """Resolves conflicts between multiple active gestures based on priority."""
    
    def __init__(self):
        # Higher index = Higher priority
        self.priorities = [
            "move",
            "left_click",
            "right_click",
            "double_click",
            "scroll",
            "drag"
        ]
        
    def resolve(self, active_gestures):
        """
        active_gestures: list of strings (gesture names currently in ACTIVE state)
        returns: single gesture name or "none"
        """
        if not active_gestures:
            return "none"
            
        # Find the gesture with the highest priority in the list
        highest_prio_idx = -1
        winner = "none"
        
        for gesture in active_gestures:
            if gesture in self.priorities:
                idx = self.priorities.index(gesture)
                if idx > highest_prio_idx:
                    highest_prio_idx = idx
                    winner = gesture
                    
        return winner
