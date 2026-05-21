cmake_minimum_required(VERSION 3.0.2)
project(standalone_ros_rl_feedthrough)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs geometry_msgs agiros_msgs)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/rl_feedthrough_node.py
  scripts/rl_feedthrough_rosbridge_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(FILES scripts/rl_feedthrough_core.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
