# VRMOUSE V2 -- Production Grade Virtual Mouse System

Generated: 2026-02-23 10:13:51

------------------------------------------------------------------------

# 🎯 Project Objective

Redesign the current VRMOUSE system to:

-   Fix inaccurate gesture recognition
-   Eliminate gesture conflicts
-   Add a dedicated black & white hand mask window
-   Improve smoothing and cursor stability
-   Implement gesture state machine
-   Improve AI feature engineering
-   Ensure low latency (\<40ms)
-   Modularize architecture for scalability

------------------------------------------------------------------------

# 🧠 SYSTEM ARCHITECTURE (V2)

## Processing Pipeline

Webcam Frame\
→ Preprocessing\
→ MediaPipe Hand Detection\
→ Landmark Normalization\
→ Feature Extraction\
→ Gesture State Machine\
→ Conflict Resolver\
→ Adaptive Smoothing\
→ C++ Mouse Controller\
→ OS Events

Parallel Output:\
→ Mask Rendering Window

------------------------------------------------------------------------

# 🖥️ NEW FEATURE: BLACK & WHITE HAND MASK WINDOW

## Requirements

Create a second OpenCV window showing:

-   Black background
-   White segmented hand only
-   Optional landmark overlay (Debug mode)

## Implementation Options

1.  MediaPipe Selfie Segmentation
2.  HSV Skin thresholding
3.  Contour extraction using landmarks

## Display Modes

-   Raw Camera
-   Mask Only
-   Debug Mode (Mask + Landmarks + Gesture Name)

------------------------------------------------------------------------

# 🧩 GESTURE STATE MACHINE (MANDATORY)

Current system triggers gestures continuously without lifecycle control.

## Implement Proper Gesture Lifecycle

IDLE → DETECTED → CONFIRMED → ACTIVE → RELEASED → COOLDOWN → IDLE

## Rules

-   Gesture must remain stable for 5--8 frames before CONFIRMED
-   Only ONE ACTIVE gesture allowed at a time
-   Add cooldown (200--400ms) after RELEASED
-   Prevent re-triggering during cooldown

------------------------------------------------------------------------

# 🎛 GESTURE PRIORITY SYSTEM

Prevent conflicts using priority order:

1.  Drag (Highest)
2.  Scroll
3.  Click
4.  Move (Lowest)

Higher priority blocks lower priority gestures.

------------------------------------------------------------------------

# 🖱️ REQUIRED GESTURES

  Gesture        Logic                   Action
  -------------- ----------------------- --------------
  Move           Index finger extended   Cursor Move
  Left Click     Thumb + Index pinch     Left Click
  Right Click    Thumb + Middle pinch    Right Click
  Drag           Pinch + Hold \>300ms    Hold Mouse
  Scroll         Two fingers vertical    Scroll
  Double Click   Two fast pinches        Double Click

------------------------------------------------------------------------

# 🧮 AI IMPROVEMENTS

## Feature Engineering Upgrade

Do NOT use raw landmarks directly.

Extract:

-   Finger joint angles
-   Finger extension states (binary)
-   Relative distances
-   Palm orientation vector
-   Finger tip velocity
-   Gesture stability duration

Minimum 300 samples per gesture.

## Model Options

Preferred:

-   XGBoost
-   LightGBM
-   MLP Neural Network

Temporal gestures → Consider LSTM.

------------------------------------------------------------------------

# 🎯 CURSOR SMOOTHING IMPROVEMENTS

Current: Basic smoothing

Upgrade to:

-   Adaptive smoothing (based on velocity)
-   Dead-zone threshold
-   Velocity-based acceleration
-   Frame prediction
-   Jitter suppression filter

------------------------------------------------------------------------

# 🚀 PERFORMANCE OPTIMIZATION

-   Process every 2nd frame (optional adaptive)
-   Run gesture detection in separate thread
-   Use NumPy vectorized operations
-   Limit FPS to 60
-   Graceful camera reconnect
-   Exception-safe MediaPipe calls

------------------------------------------------------------------------

# 🪟 WINDOW SYSTEM

Application must support:

1.  Main Camera Window
2.  Hand Mask Window
3.  Debug HUD (FPS + State + Active Gesture)

------------------------------------------------------------------------

# 📁 PROPOSED FOLDER STRUCTURE

VRMOUSE_V2/ │ ├── python/ │ ├── main.py │ ├── gesture_engine.py │ ├──
state_machine.py │ ├── conflict_resolver.py │ ├── mask_renderer.py │ ├──
smoothing_engine.py │ ├── feature_extractor.py │ ├── ai/ │ │ ├──
trainer.py │ │ ├── model.pkl │ ├── cpp/ │ ├── mouse_control.cpp │ ├──
config.json └── README.md

------------------------------------------------------------------------

# 🧪 TESTING REQUIREMENTS

## Accuracy Testing

-   50 trials per gesture
-   Confusion matrix logging
-   False positive rate measurement
-   Latency measurement

## Smoothness Testing

-   Cursor deviation error
-   Jitter variance
-   FPS stability monitoring

------------------------------------------------------------------------

# 🏁 EXPECTED FINAL RESULT

-   95%+ gesture accuracy
-   Zero gesture conflicts
-   Stable cursor movement
-   Smooth drag behavior
-   Clean mask window
-   Latency under 40ms

------------------------------------------------------------------------

# 📌 OPTIONAL ADVANCED FEATURES

-   Multi-hand support
-   Gesture recording UI
-   Sensitivity slider
-   Real-time model retraining
-   Auto FPS adjustment

------------------------------------------------------------------------

# 🔥 FINAL INSTRUCTION TO ANTIGRAVITY

This is a full architectural redesign.\
Focus on modular design, stability, performance optimization, and
gesture reliability.\
Eliminate heuristic conflicts and implement strict state-based gesture
control.\
System must be production-grade, extensible, and low latency.
