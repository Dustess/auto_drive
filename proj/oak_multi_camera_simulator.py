#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将真实 OAK RGB 图像转发到多个模拟相机话题。"""

import rospy
from sensor_msgs.msg import Image


class OakMultiCameraSimulator:
    def __init__(self):
        rospy.init_node("oak_multi_camera_simulator")

        source_topic = str(rospy.get_param("~source_topic", "/oak/rgb/image_raw"))
        topic_prefix = str(rospy.get_param("~output_topic_prefix", source_topic))
        camera_count = int(rospy.get_param("~camera_count", 5))
        if not 1 <= camera_count <= 5:
            raise rospy.ROSInitException("~camera_count 必须在 1 到 5 之间")

        self.source_topic = rospy.resolve_name(source_topic)
        self.output_topics = [
            rospy.resolve_name("%s_%d" % (topic_prefix.rstrip("/"), index))
            for index in range(1, camera_count + 1)
        ]
        if self.source_topic in self.output_topics:
            raise rospy.ROSInitException("输出话题不能与输入话题相同，否则会形成转发循环")

        self.publishers = [
            rospy.Publisher(topic, Image, queue_size=1)
            for topic in self.output_topics
        ]
        self.subscriber = rospy.Subscriber(
            self.source_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=16 * 1024 * 1024,
        )

        rospy.loginfo(
            "OAK 多相机图像模拟节点已启动：%s -> %s",
            self.source_topic,
            ", ".join(self.output_topics),
        )

    def _image_callback(self, image_msg):
        # 直接转发 ROS Image 消息，保留原始图像数据和 Header（含时间戳、frame_id）。
        for publisher in self.publishers:
            publisher.publish(image_msg)


if __name__ == "__main__":
    try:
        OakMultiCameraSimulator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
