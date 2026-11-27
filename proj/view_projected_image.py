#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""订阅并显示 project_bbox_node 发布的融合图像。"""

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ProjectedImageViewer:
    def __init__(self):
        rospy.init_node("projected_image_viewer")

        self.image_topic = rospy.get_param(
            "~image_topic", "/fusion/projected_image"
        )
        self.window_name = rospy.get_param(
            "~window_name", "LiDAR Camera Projection"
        )
        self.bridge = CvBridge()
        self.window_created = False

        # 队列只保留最新图像，避免显示速度较慢时不断积压旧帧。
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.on_shutdown(self._close_window)
        rospy.loginfo("图像查看节点已启动，订阅话题：%s", self.image_topic)

    def _image_callback(self, msg):
        """将 ROS Image 转成 OpenCV 图像并在窗口中刷新显示。"""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(3.0, "图像转换失败：%s", exc)
            return

        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True

        cv2.imshow(self.window_name, image)
        # waitKey 负责刷新 OpenCV 窗口事件；按 q 或 Esc 关闭查看节点。
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rospy.signal_shutdown("用户关闭图像窗口")

    def _close_window(self):
        """ROS 节点退出时销毁 OpenCV 窗口。"""
        if self.window_created:
            cv2.destroyAllWindows()
            self.window_created = False


if __name__ == "__main__":
    try:
        ProjectedImageViewer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
