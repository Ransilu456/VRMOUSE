g++ -O3 -shared -o mouse_control.dll cpp/mouse_control.cpp -static-libgcc -static-libstdc++
if ($?) {
    Write-Host "Build Successful: mouse_control.dll created." -ForegroundColor Green
} else {
    Write-Host "Build Failed." -ForegroundColor Red
}
