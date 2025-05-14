from pathlib import Path
from setuptools import find_packages, setup


def read_requirements():
    return Path("requirements.txt").read_text().splitlines()


setup(
    name="mlpt",
    version="0.1.0",
    description="PPE Detection Pipeline (YOLO + DVC)",
    author="HASKII",
    python_requires=">=3.9",
    packages=find_packages(),          # захватывает mlpt и все подпакеты
    include_package_data=True,
    install_requires=read_requirements(),
)
