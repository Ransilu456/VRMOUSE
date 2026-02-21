#ifdef _WIN32
#include <windows.h>
#define EXPORT extern "C" __declspec(dllexport)
#else
#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>
#define EXPORT extern "C"
#endif

// Helper to send input on Windows
#ifdef _WIN32
void send_mouse_input(DWORD flags, DWORD data = 0) {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = flags;
    input.mi.mouseData = data;
    SendInput(1, &input, sizeof(INPUT));
}
#endif

EXPORT void move_mouse(int x, int y) {
#ifdef _WIN32
    SetCursorPos(x, y);
#else
    Display *d = XOpenDisplay(0);
    XWarpPointer(d, None, DefaultRootWindow(d), 0,0,0,0,x,y);
    XFlush(d);
    XCloseDisplay(d);
#endif
}

EXPORT void click_mouse() {
#ifdef _WIN32
    send_mouse_input(MOUSEEVENTF_LEFTDOWN);
    send_mouse_input(MOUSEEVENTF_LEFTUP);
#else
    Display *d = XOpenDisplay(0);
    XTestFakeButtonEvent(d, 1, True, CurrentTime);
    XTestFakeButtonEvent(d, 1, False, CurrentTime);
    XFlush(d);
    XCloseDisplay(d);
#endif
}

EXPORT void right_click_mouse() {
#ifdef _WIN32
    send_mouse_input(MOUSEEVENTF_RIGHTDOWN);
    send_mouse_input(MOUSEEVENTF_RIGHTUP);
#else
    Display *d = XOpenDisplay(0);
    XTestFakeButtonEvent(d, 3, True, CurrentTime);
    XTestFakeButtonEvent(d, 3, False, CurrentTime);
    XFlush(d);
    XCloseDisplay(d);
#endif
}

EXPORT void double_click_mouse() {
    click_mouse();
    click_mouse();
}

EXPORT void scroll_up() {
#ifdef _WIN32
    send_mouse_input(MOUSEEVENTF_WHEEL, WHEEL_DELTA);
#else
    Display *d = XOpenDisplay(0);
    XTestFakeButtonEvent(d, 4, True, CurrentTime);
    XTestFakeButtonEvent(d, 4, False, CurrentTime);
    XFlush(d);
    XCloseDisplay(d);
#endif
}

EXPORT void scroll_down() {
#ifdef _WIN32
    send_mouse_input(MOUSEEVENTF_WHEEL, (DWORD)-WHEEL_DELTA);
#else
    Display *d = XOpenDisplay(0);
    XTestFakeButtonEvent(d, 5, True, CurrentTime);
    XTestFakeButtonEvent(d, 5, False, CurrentTime);
    XFlush(d);
    XCloseDisplay(d);
#endif
}
