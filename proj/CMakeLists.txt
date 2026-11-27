cmake_minimum_required(VERSION 3.0.2)
project(ad10_yolo_fusion)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  project_on_ad10_Yolo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
