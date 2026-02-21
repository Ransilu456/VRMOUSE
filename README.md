# Virtual Mouse Pro (MediaPipe AI Version)

This version uses MediaPipe for high-fidelity hand tracking and a dedicated AI model for custom gesture recognition. It features a modern HUD with multiple preview modes and a integrated AI gesture trainer.

## Key Features

- **MediaPipe Hand Tracking**: Super-stable 21-point hand landmark tracking.
- **Custom Gestures**: Train your own gestures using the built-in Studio and Trainer.
- **Specialized Previews**:
    - **Application Mode**: Standard view for mouse control.
    - **Mask Mode**: Visualizes the hand contour and mask used for processing.
    - **Training Mode**: Shows the skeleton and normalized feature vectors (wrist-relative).
- **Modern HUD**: Semi-transparent glass panels, ripple effects on clicks, and real-time status updates.
- **C++ Mouse Control**: Low-latency mouse interaction via a custom-built DLL.

## Setup

1.  **Install Python Dependencies:**
    ```bash
    pip install opencv-python mediapipe numpy pandas scikit-learn joblib
    ```

2.  **Build the Mouse Control DLL:**
    Run the provided PowerShell script:
    ```powershell
    .\build_dll.ps1
    ```

## Usage

### 1. Main Application
Run the control software:
```bash
python python/main.py
```
- **[T]**: Toggle Mouse Control.
- **[M]**: Cycle through Preview Modes (App -> Mask -> Training).
- **[ESC]**: Quit.

### 2. Custom Gesture Training
1.  **Collect Data**: Run `python python/data_collector.py`. Select an action (1-6) and hold 'R' to record your custom hand shape.
2.  **Train Model**: Run `python python/ai_trainer.py`. Press 'T' to train the model. You will see a **Live Preview** box to test your gesture immediately after training.

## Controls (Default Heuristic)

If no AI model is found, the system falls back to:
- **Index Finger Up**: Move cursor.
- **Pinch (Index + Thumb)**: Left Click.
- **V-Sign (Index + Middle)**: Right Click.
- **Fingers Spread**: No action.
- **Pinch + Middle Move**: Scroll.
