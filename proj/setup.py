#!/usr/bin/env python3
"""Install the Python package when this ROS package is built with catkin."""

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup(**generate_distutils_setup(
    packages=["project_on_ad10_yolo"],
    package_dir={"": "."},
))
