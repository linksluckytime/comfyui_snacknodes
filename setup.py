from setuptools import setup, find_packages

setup(
    name="comfyui_snacknodes",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "comfyui>=1.0.0"
    ],
) 