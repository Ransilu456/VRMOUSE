import ctypes
import os
import time

dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mouse_control.dll'))
print(f"Testing DLL at: {dll_path}")

try:
    mouse_dll = ctypes.CDLL(dll_path)
    mouse_dll.move_mouse.argtypes = [ctypes.c_int, ctypes.c_int]
    
    print("Moving mouse to (500, 500) in 2 seconds...")
    time.sleep(2)
    mouse_dll.move_mouse(500, 500)
    print("Move call completed.")
    
    print("Clicking in 1 second...")
    time.sleep(1)
    mouse_dll.click_mouse()
    print("Click call completed.")

except Exception as e:
    print(f"Error: {e}")
