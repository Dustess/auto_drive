#!/home/nvidia/auto/yolo11/venv/bin/python
# -*- coding: utf-8 -*-

"""使用 Ultralytics YOLO 对 OAK RGB 图像进行实时检测并发布检测框和标注图像。"""

import os
import threading
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class OakYoloDetectionNode:
    def __init__(self):
        rospy.init_node("oak_yolo_detection_node")

        self.image_topic = rospy.get_param("~image_topic", "/oak/rgb/image_raw")
        self.detection_topic = rospy.get_param("~detection_topic", "/yolo/detections")
        self.annotated_image_topic = rospy.get_param(
            "~annotated_image_topic", "/yolo/image"
        )
        self.model_path = os.path.expanduser(
            rospy.get_param("~model_path", "yolo11n.pt")
        )
        self.image_size = int(rospy.get_param("~image_size", 640))
        self.confidence = float(rospy.get_param("~confidence", 0.25))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.device = str(rospy.get_param("~device", "0"))
        self.max_rate = float(rospy.get_param("~max_rate", 0.0))

        if not os.path.isfile(self.model_path):
            raise rospy.ROSInitException(
                "YOLO 模型文件不存在: %s；请设置私有参数 ~model_path" % self.model_path
            )
        if self.image_size <= 0:
            raise rospy.ROSInitException("~image_size 必须大于 0")
        if self.max_rate < 0.0:
            raise rospy.ROSInitException("~max_rate 不能小于 0")

        # Ultralytics 使用当前进程的 Python 环境加载，需从 YOLO venv 启动本节点。
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise rospy.ROSInitException(
                "当前 Python 环境未安装 ultralytics: %s；请使用配置好的 YOLO venv 启动节点"
                % exc
            )

        rospy.loginfo("正在加载 YOLO 模型: %s", self.model_path)
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.last_inference_start = 0.0
        self.last_log_time = time.monotonic()
        self.frame_count = 0

        self.detection_pub = rospy.Publisher(
            self.detection_topic, Detection2DArray, queue_size=1
        )
        self.image_pub = rospy.Publisher(self.annotated_image_topic, Image, queue_size=1)
        # queue_size=1 丢弃等待处理的旧帧，优先检测最新图像，降低实时显示延迟。
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=8 * 1024 * 1024,
        )

        rospy.loginfo(
            "OAK YOLO 节点已启动 | image=%s detections=%s annotated_image=%s "
            "imgsz=%d conf=%.2f iou=%.2f device=%s max_rate=%.2fHz",
            self.image_topic,
            self.detection_topic,
            self.annotated_image_topic,
            self.image_size,
            self.confidence,
            self.iou,
            self.device,
            self.max_rate,
        )

    @staticmethod
    def _set_hypothesis(hypothesis_with_pose, class_id, class_name, score):
        """兼容 vision_msgs 不同发行版中的 ObjectHypothesis 字段名称。"""
        hypothesis = getattr(hypothesis_with_pose, "hypothesis", hypothesis_with_pose)
        if hasattr(hypothesis, "class_id"):
            hypothesis.class_id = class_name
        elif hasattr(hypothesis, "id"):
            id_type = hypothesis._slot_types[hypothesis.__slots__.index("id")]
            hypothesis.id = class_name if id_type == "string" else int(class_id)
        if hasattr(hypothesis, "score"):
            hypothesis.score = float(score)

    def _make_detection(self, header, class_id, class_name, score, xyxy):
        detection = Detection2D()
        detection.header = header
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        detection.bbox.center.x = 0.5 * (x1 + x2)
        detection.bbox.center.y = 0.5 * (y1 + y2)
        if hasattr(detection.bbox.center, "theta"):
            detection.bbox.center.theta = 0.0
        detection.bbox.size_x = max(0.0, x2 - x1)
        detection.bbox.size_y = max(0.0, y2 - y1)
        if hasattr(detection, "id"):
            detection.id = ""

        # vision_msgs 的 results 是 ObjectHypothesisWithPose[]。
        result = ObjectHypothesisWithPose()
        self._set_hypothesis(result, class_id, class_name, score)
        detection.results.append(result)
        return detection

    def _image_callback(self, image_msg):
        # 可选限频；默认不主动限速，处理能力不足时由 ROS 队列丢弃旧图像。
        with self.lock:
            now = time.monotonic()
            if self.max_rate > 0.0 and self.last_inference_start > 0.0:
                min_period = 1.0 / self.max_rate
                if now - self.last_inference_start < min_period:
                    return
            self.last_inference_start = now

        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(5.0, "图像转换失败: %s", exc)
            return

        start = time.monotonic()
        try:
            results = self.model.predict(
                source=image,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            rospy.logerr_throttle(5.0, "YOLO 推理失败: %s", exc)
            return

        if not results:
            return

        result = results[0]
        detection_array = Detection2DArray()
        detection_array.header = image_msg.header
        boxes = result.boxes
        if boxes is not None:
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
            scores = boxes.conf.detach().cpu().numpy()
            coordinates = boxes.xyxy.detach().cpu().numpy()
            names = result.names
            for class_id, score, xyxy in zip(class_ids, scores, coordinates):
                if isinstance(names, dict):
                    class_name = str(names.get(int(class_id), class_id))
                else:
                    class_name = str(names[int(class_id)])
                detection_array.detections.append(
                    self._make_detection(
                        image_msg.header, int(class_id), class_name, float(score), xyxy
                    )
                )

        try:
            # Ultralytics plot 返回带检测框、类别名和置信度的 BGR 图像。
            annotated = result.plot()
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = image_msg.header
        except (CvBridgeError, cv2.error) as exc:
            rospy.logerr_throttle(5.0, "标注图像生成失败: %s", exc)
            return

        self.detection_pub.publish(detection_array)
        self.image_pub.publish(annotated_msg)

        self.frame_count += 1
        elapsed = time.monotonic() - start
        log_now = time.monotonic()
        if log_now - self.last_log_time >= 5.0:
            rospy.loginfo(
                "YOLO 运行中 | 检测数=%d 推理与发布耗时=%.1fms 已处理帧数=%d",
                len(detection_array.detections),
                elapsed * 1000.0,
                self.frame_count,
            )
            self.last_log_time = log_now


if __name__ == "__main__":
    try:
        OakYoloDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
