from setuptools import setup, Extension
import pybind11

functions_module = Extension(
    'mouse_control',
    sources=['cpp/mouse_control.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++'
)

setup(
    name='mouse_control',
    version='0.1',
    ext_modules=[functions_module],
)
