"""YOLO 推理、独立 LiDAR 投影与可视化发布三个任务阶段。"""

import copy
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class YoloFusionMixin:
    @staticmethod
    def _make_2d_detection(header, source, object_id, class_id, class_name, score, xyxy):
        """创建统一的 2D 检测消息；detection.id 标明来源和目标编号。"""
        detection = Detection2D()
        detection.header = copy.deepcopy(header)
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        detection.bbox.center.x = 0.5 * (x1 + x2)
        detection.bbox.center.y = 0.5 * (y1 + y2)
        if hasattr(detection.bbox.center, "theta"):
            detection.bbox.center.theta = 0.0
        detection.bbox.size_x = max(0.0, x2 - x1)
        detection.bbox.size_y = max(0.0, y2 - y1)
        if hasattr(detection, "id"):
            detection.id = "%s:%s" % (source, object_id)

        result = ObjectHypothesisWithPose()
        hypothesis = getattr(result, "hypothesis", result)
        if hasattr(hypothesis, "class_id"):
            hypothesis.class_id = str(class_name)
        elif hasattr(hypothesis, "id"):
            id_type = hypothesis._slot_types[hypothesis.__slots__.index("id")]
            hypothesis.id = str(class_name) if id_type == "string" else int(class_id)
        if hasattr(hypothesis, "score"):
            hypothesis.score = float(score)
        detection.results.append(result)
        return detection

    def _run_yolo(self, image):
        """仅将数据返回给 CPU ；不向其他线程传递模型 Results/GPU tensor。"""
        predict_started = time.perf_counter()
        results = self.yolo_model.predict(
            source=image, imgsz=self.yolo_image_size, conf=self.yolo_confidence,
            iou=self.yolo_iou, device=self.yolo_device, verbose=False,
        )
        predict_ms = (time.perf_counter() - predict_started) * 1000.0
        extract_started = time.perf_counter()
        detections = []
        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
                scores = boxes.conf.detach().cpu().numpy()
                coordinates = boxes.xyxy.detach().cpu().numpy()##转换到CPU
                names = result.names
                for class_id, score, xyxy in zip(class_ids, scores, coordinates):
                    class_name = (str(names.get(int(class_id), class_id))
                                  if isinstance(names, dict) else str(names[int(class_id)]))
                    detections.append((int(class_id), class_name, float(score), xyxy.copy()))
        return detections, {"yolo_predict_ms": predict_ms,
                            "yolo_extract_ms": (time.perf_counter() - extract_started) * 1000.0}

    def _output_header(self, frame):
        header = copy.deepcopy(frame["image"].header)
        if self.output_stamp_with_system_time:
            header.stamp = rospy.Time.now()
        return header

    def _observe_image_age(self, index, name, frame):
        age_ms = (rospy.Time.now().to_sec() - frame["image_stamp"]) * 1000.0
        if age_ms < 0:
            self.metrics.increment(index, "negative_" + name + "_samples")
        else:
            self.metrics.observe(index, name, age_ms)

    def _detection_array(self, header, camera_index, yolo_boxes, projected_boxes=()):
        detection_array = Detection2DArray()
        detection_array.header = copy.deepcopy(header)
        camera_name = self.camera_names[camera_index]
        for index, (class_id, class_name, score, xyxy) in enumerate(yolo_boxes):
            detection_array.detections.append(self._make_2d_detection(
                header, "%s:yolo" % camera_name, index, class_id, class_name, score, xyxy))
        for marker_id, xyxy in projected_boxes:
            detection_array.detections.append(self._make_2d_detection(
                header, "%s:lidar_cluster" % camera_name, marker_id, -1,
                "lidar_cluster", 1.0, xyxy))
        return detection_array

    def _process_frame(self, frame, bbox_msg, bbox_stamp):
        """单模型推理阶段。只发布纯 YOLO 检测并交接图像，不等待投影或绘图。"""
        index = frame["camera_index"]
        if not self._validate_timestamps(frame, bbox_stamp):
            self.metrics.increment(index, "timestamp_dropped")
            return
        started = time.perf_counter()
        try:
            # CvBridge 可能返回消息内存的视图；工作任务拥有唯一可写副本。
            image = self.bridge.imgmsg_to_cv2(frame["image"], desired_encoding="bgr8").copy()
        except CvBridgeError as exc:
            self.metrics.increment(index, "decode_errors")
            rospy.logerr_throttle(3.0, "%s 图像转换失败: %s", self.camera_names[index], exc)
            return
        self.metrics.observe(index, "image_decode_ms", (time.perf_counter() - started) * 1000.0)
        try:
            yolo_boxes, timings = self._run_yolo(image)
        except Exception as exc:
            self.metrics.increment(index, "yolo_errors")
            rospy.logerr_throttle(5.0, "YOLO 推理失败，本帧继续输出图像/可用投影: %s", exc)
            yolo_boxes, timings = [], {}
        else:
            self.metrics.increment(index, "yolo")
            self._observe_image_age(index, "yolo_image_age_ms", frame)
            started = time.perf_counter()
            # queue_size=1 异步 ROS 发布；失败不应中止融合图像流水线。
            if not self._task_expired({"frame": frame}, "yolo_publish"):
                try:
                    detection_array = self._detection_array(self._output_header(frame), index, yolo_boxes)
                    self.yolo_detections_pubs[index].publish(detection_array)
                    self.metrics.increment(index, "yolo_published")
                except Exception as exc:
                    self.metrics.increment(index, "yolo_publish_errors")
                    rospy.logerr_throttle(3.0, "纯 YOLO 检测发布失败: %s", exc)
            self.metrics.observe(index, "yolo_publish_ms", (time.perf_counter() - started) * 1000.0)
        for name, value in timings.items():
            self.metrics.observe(index, name, value)
        task = {"frame": frame, "image": image, "yolo_boxes": yolo_boxes,
                "bbox": bbox_msg, "projected_boxes": []}
        if index in self.projection_camera_indices and frame.get("cloud") is not None:
            self._submit_task(self.projection_buffer, task, "projection")
        else:
            self._submit_task(self.visualization_buffer, task, "visualization")

    def _process_projection(self, task):
        """独立单线程拥有投影坐标 LRU；图像任务与匹配点云一同传递。"""
        frame, image = task["frame"], task["image"]
        index = frame["camera_index"]
        timings = {}
        started = time.perf_counter()
        try:
            height, width = image.shape[:2]
            camera_matrix, distortion, from_topic = self._camera_model_for_image(index, width, height)
            if not from_topic:
                rospy.logwarn_throttle(10.0, "%s 使用 JSON 内参", self.camera_names[index])
            cloud_frame = frame["cloud"].header.frame_id
            if self.expected_lidar_frame and cloud_frame != self.expected_lidar_frame:
                rospy.logwarn_throttle(5.0, "点云 frame_id='%s'，标定期望='%s'",
                                       cloud_frame, self.expected_lidar_frame)
            self._draw_cloud(image, index, frame["cloud"], camera_matrix, distortion, timings=timings)
            if task["bbox"] is not None:
                bbox_started = time.perf_counter()
                task["projected_boxes"] = self._draw_bboxes(
                    image, index, task["bbox"], camera_matrix, distortion)
                timings["bbox_projection_ms"] = (time.perf_counter() - bbox_started) * 1000.0
            self.metrics.increment(index, "projection")
        except Exception as exc:
            self.metrics.increment(index, "projection_errors")
            rospy.logerr_throttle(3.0, "%s 投影中断，继续发布本帧（可能保留部分投影）: %s",
                                 self.camera_names[index], exc)
        finally:
            self.metrics.observe(index, "lidar_projection_ms", (time.perf_counter() - started) * 1000.0)
            for name, value in timings.items():
                if name == "projection_cache_hit":
                    self.metrics.increment(index, "projection_cache_hits" if value else "projection_cache_misses")
                else:
                    self.metrics.observe(index, name, value)
            self._submit_task(self.visualization_buffer, task, "visualization")

    @staticmethod
    def _draw_yolo_boxes(image, detections):
        """CPU 绘图替代 Results.plot，确保下游不再触碰推理模型及 CUDA。"""
        height, width = image.shape[:2]
        for _class_id, name, score, xyxy in detections:
            x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
            x1, x2 = np.clip([x1, x2], 0, width - 1)
            y1, y2 = np.clip([y1, y2], 0, height - 1)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
            cv2.putText(image, "%s %.2f" % (name, score), (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    def _process_visualization(self, task):
        frame, image = task["frame"], task["image"]
        index = frame["camera_index"]
        if frame["frame_version"] <= self.last_output_versions[index]:
            self.metrics.increment(index, "visualization_stale_dropped")
            return
        started = time.perf_counter()
        self._draw_yolo_boxes(image, task["yolo_boxes"])
        self.metrics.observe(index, "plot_ms", (time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        output = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.metrics.observe(index, "cv2_to_imgmsg_ms", (time.perf_counter() - started) * 1000.0)
        if self._task_expired(task, "visualization"):
            return
        output_header = self._output_header(frame)
        output.header = copy.deepcopy(output_header)
        detection_array = self._detection_array(
            output_header, index, task["yolo_boxes"], task["projected_boxes"])
        started = time.perf_counter()
        self.image_pubs[index].publish(output)
        self.detections_pubs[index].publish(detection_array)
        completed = time.perf_counter()
        self.last_output_versions[index] = frame["frame_version"]
        self.metrics.observe(index, "publish_ms", (completed - started) * 1000.0)
        self.metrics.increment(index, "output")
        self.metrics.observe(index, "end_to_end_ms", (completed - frame["received_monotonic"]) * 1000.0)
        # 使用原始采集 stamp，即使用户选择输出 stamp=系统时间也不改变延迟统计。
        self._observe_image_age(index, "image_age_ms", frame)
        self.metrics.observe(index, "output_frame_version", frame["frame_version"])
