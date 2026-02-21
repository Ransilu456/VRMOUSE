import os
import sys
import sysconfig
import subprocess
import pybind11

def build():
    print("Starting build with g++...")
    
    # 1. Gather paths
    python_include = os.path.abspath(sysconfig.get_path('include'))
    pybind11_include = os.path.abspath(pybind11.get_include())
    python_libs_dir = os.path.abspath(os.path.join(sys.base_prefix, 'libs'))
    
    # e.g., python312
    python_lib_name = f"python{sys.version_info.major}{sys.version_info.minor}"
    python_lib_path = os.path.join(python_libs_dir, f"{python_lib_name}.lib")
    
    source_file = os.path.abspath("cpp/mouse_control.cpp")
    
    # Standard suffix for Python 3.12 is .cp312-win_amd64.pyd
    # But mouse_control.pyd should also work if architecture matches.
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.pyd'
    output_file = os.path.abspath(f"mouse_control{ext_suffix}")
    
    print(f"Python Include: {python_include}")
    print(f"Library Path: {python_lib_path}")
    
    # 2. Construct the g++ command
    command = [
        "g++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++11",
        "-DNDEBUG",
        f"-I{python_include}",
        f"-I{pybind11_include}",
        source_file,
        "-o", output_file,
        f"-L{python_libs_dir}",
        f"-l{python_lib_name}",
        "-static-libgcc",
        "-static-libstdc++",
        "-Wl,--add-stdcall-alias"
    ]
    
    print(f"Executing: {' '.join(command)}")
    
    # 3. Run the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Build successful! Created: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Build failed!")
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    build()
