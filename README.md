# Virtual Mouse (MediaPipe-Free Version)

This version uses pure OpenCV for hand tracking and a C++ DLL for mouse control, eliminating complex dependencies and build errors.

## Prerequisites

- Python 3.x
- G++ Compiler (MinGW/MSYS2)

## Setup

1.  **Install Python Dependencies:**
    ```bash
    pip install opencv-python numpy
    ```

2.  **Build the Mouse Control DLL:**
    Run the provided PowerShell script to compile the C++ code into a DLL:
    ```powershell
    .\build_dll.ps1
    ```

## Usage

Run the application:
```bash
python python/main.py
```

### Gestures (Color/Contour Based)

- **Move Cursor**: Move your hand inside the blue box. The center of your hand will control the cursor.
- **Left Click**: Spread your fingers to create **one gap** (like a 'V' sign).
- **Right Click**: Spread your fingers to create **two or more gaps**.

## Controls

- Press **ESC** to exit.
