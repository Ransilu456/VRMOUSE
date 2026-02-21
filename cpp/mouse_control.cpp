#ifdef _WIN32
#include <windows.h>
#define EXPORT extern "C" __declspec(dllexport)
#else
#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>
#define EXPORT extern "C"
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
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
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
    mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
    mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);
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
    mouse_event(MOUSEEVENTF_WHEEL, 0, 0, 120, 0);
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
    mouse_event(MOUSEEVENTF_WHEEL, 0, 0, (DWORD)-120, 0);
#else
    Display *d = XOpenDisplay(0);
    XTestFakeButtonEvent(d, 5, True, CurrentTime);
    XTestFakeButtonEvent(d, 5, False, CurrentTime);
    XFlush(d);
    XCloseDisplay(d);
#endif
}
