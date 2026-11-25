#!/home/nvidia/auto/yolo11/venv/bin/python
# -*- coding: utf-8 -*-
"""六相机 LiDAR 与 YOLO 融合节点的启动入口。"""

import rospy

from project_on_ad10_yolo import ProjectBboxNode


if __name__ == "__main__":
    try:
        ProjectBboxNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
