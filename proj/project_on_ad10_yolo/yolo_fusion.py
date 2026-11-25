"""YOLO 推理、检测消息构造和单帧融合处理。"""

import copy
import time

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
        """在原始相机图像上运行 YOLO，返回带 YOLO 框的图像和框坐标。"""
        results = self.yolo_model.predict(
            source=image,
            imgsz=self.yolo_image_size,
            conf=self.yolo_confidence,
            iou=self.yolo_iou,
            device=self.yolo_device,
            verbose=False,
        )
        if not results:
            return image, []

        result = results[0]
        boxes = result.boxes
        detections = []
        if boxes is not None:
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
            scores = boxes.conf.detach().cpu().numpy()
            coordinates = boxes.xyxy.detach().cpu().numpy()
            names = result.names
            for class_id, score, xyxy in zip(class_ids, scores, coordinates):
                class_name = (
                    str(names.get(int(class_id), class_id))
                    if isinstance(names, dict)
                    else str(names[int(class_id)])
                )
                detections.append((int(class_id), class_name, float(score), xyxy))
        return result.plot(), detections

    def _process_frame(self, frame, bbox_msg, bbox_stamp):
        """完成单帧处理：时间检查、图像解码、点云/框绘制、结果发布及耗时信息输出。"""
        process_started = time.perf_counter()
        stage_started = process_started
        if not self._validate_timestamps(frame, bbox_stamp):
            return
        timestamp_validation_ms = (time.perf_counter() - stage_started) * 1000.0

        camera_index = frame["camera_index"]
        camera_name = self.camera_names[camera_index]
        project_lidar = (
            camera_index in self.projection_camera_indices
            and frame.get("cloud") is not None
        )
        stage_started = time.perf_counter()
        try:
            image = self.bridge.imgmsg_to_cv2(frame["image"], desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(3.0, "%s 图像转换失败: %s", camera_name, exc)
            return
        image_decode_ms = (time.perf_counter() - stage_started) * 1000.0

        height, width = image.shape[:2]
        camera_matrix = None
        distortion = None
        calibration_lookup_ms = 0.0
        if project_lidar:
            stage_started = time.perf_counter()
            camera_matrix, distortion, from_topic = self._camera_model_for_image(
                camera_index, width, height
            )
            calibration_lookup_ms = (time.perf_counter() - stage_started) * 1000.0
            if not from_topic:
                rospy.logwarn_throttle(
                    10.0,
                    "%s 尚未收到有效 CameraInfo，使用对应 intrinsics JSON 参数",
                    camera_name,
                )

            cloud_frame = frame["cloud"].header.frame_id
            if self.expected_lidar_frame and cloud_frame != self.expected_lidar_frame:
                rospy.logwarn_throttle(
                    5.0,
                    "点云 frame_id='%s'，标定源坐标系期望='%s'；确认两者坐标轴定义一致",
                    cloud_frame,
                    self.expected_lidar_frame,
                )

        # 所有相机都绘制 YOLO 框；首尾相机再叠加 LiDAR 点云和聚类框投影。
        stage_started = time.perf_counter()
        try:
            image, yolo_boxes = self._run_yolo(image)
        except Exception as exc:
            rospy.logerr_throttle(5.0, "YOLO 推理失败，本帧仍继续处理: %s", exc)
            yolo_boxes = []
        yolo_inference_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        detection_array = Detection2DArray()
        output_header = copy.deepcopy(frame["image"].header)
        now = rospy.Time.now()
        if self.output_stamp_with_system_time:
            output_header.stamp = now
        detection_array.header = copy.deepcopy(output_header)
        for index, (class_id, class_name, score, xyxy) in enumerate(yolo_boxes):
            detection_array.detections.append(
                self._make_2d_detection(
                    output_header,
                    "%s:yolo" % camera_name,
                    index,
                    class_id,
                    class_name,
                    score,
                    xyxy,
                )
            )
        yolo_detection_msg_ms = (time.perf_counter() - stage_started) * 1000.0

        total_points = 0
        projected_points = 0
        projected_boxes = []
        lidar_projection_ms = 0.0
        bbox_projection_ms = 0.0
        if project_lidar:
            stage_started = time.perf_counter()
            total_points, projected_points = self._draw_cloud(
                image, camera_index, frame["cloud"], camera_matrix, distortion
            )
            lidar_projection_ms = (time.perf_counter() - stage_started) * 1000.0
            if bbox_msg is not None:
                stage_started = time.perf_counter()
                projected_boxes = self._draw_bboxes(
                    image, camera_index, bbox_msg, camera_matrix, distortion
                )
                for marker_id, xyxy in projected_boxes:
                    detection_array.detections.append(
                        self._make_2d_detection(
                            output_header,
                            "%s:lidar_cluster" % camera_name,
                            marker_id,
                            -1,
                            "lidar_cluster",
                            1.0,
                            xyxy,
                        )
                    )
                bbox_projection_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        output = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        output.header = copy.deepcopy(output_header)
        self.image_pubs[camera_index].publish(output)
        self.detections_pubs[camera_index].publish(detection_array)
        publish_ms = (time.perf_counter() - stage_started) * 1000.0
        total_process_ms = (time.perf_counter() - process_started) * 1000.0
        end_to_end_ms = (
            (time.perf_counter() - frame["received_monotonic"]) * 1000.0
            if "received_monotonic" in frame
            else total_process_ms
        )

        cloud_stamp = frame.get("cloud_stamp")
        bbox_delta = (
            abs(bbox_stamp - cloud_stamp)
            if bbox_stamp is not None and cloud_stamp is not None
            else 0.0
        )
        image_cloud_delta = (
            "%.3f s" % abs(frame["image_stamp"] - cloud_stamp)
            if cloud_stamp is not None
            else "n/a"
        )
        cloud_age = (
            "%.3f s" % (now.to_sec() - cloud_stamp)
            if cloud_stamp is not None
            else "n/a"
        )
        timing_now = time.perf_counter()
        if (
            timing_now - self.last_timing_log[camera_index]
            >= self.timing_log_interval_sec
        ):
            self.last_timing_log[camera_index] = timing_now
            rospy.loginfo(
                "%s 图像处理完成 | mode=%s lidar_projection=%s points=%d projected=%d "
                "lidar_boxes=%d yolo_boxes=%d "
                "image-cloud_dt=%s bbox_dt=%.3f s cloud_reused=%s image_age=%.3f s cloud_age=%s "
                "time_ms[callback=%.2f bbox_match=%.2f queue_wait=%.2f validate=%.2f "
                "decode=%.2f calibration=%.2f yolo=%.2f yolo_msg=%.2f lidar=%.2f "
                "bbox=%.2f publish=%.2f process=%.2f end_to_end=%.2f] output_stamp=%s",
                camera_name,
                self.mode,
                "yes" if project_lidar else "no",
                total_points,
                projected_points,
                len(projected_boxes),
                len(yolo_boxes),
                image_cloud_delta,
                bbox_delta,
                "yes" if frame.get("cloud_reused", False) else "no",
                now.to_sec() - frame["image_stamp"],
                cloud_age,
                frame.get("input_callback_ms", 0.0),
                frame.get("bbox_match_ms", 0.0),
                frame.get("queue_wait_ms", 0.0),
                timestamp_validation_ms,
                image_decode_ms,
                calibration_lookup_ms,
                yolo_inference_ms,
                yolo_detection_msg_ms,
                lidar_projection_ms,
                bbox_projection_ms,
                publish_ms,
                total_process_ms,
                end_to_end_ms,
                "AD10_now" if self.output_stamp_with_system_time else "camera_image_stamp",
            )
