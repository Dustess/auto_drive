#!/home/nvidia/auto/yolo11/venv/bin/python
# -*- coding: utf-8 -*-

"""运行 OAK 图像检测，并将 LiDAR 点云/三维框投影到检测图像。

cloud_only：投影聚类点云（默认 /lidar/clusters）。
cloud_bbox：投影非地面点云（默认 /lidar/nonground_cloud）和 /lidar/bboxes 中的 Marker 框。
外参方向为 p_camera = R_camera_from_lidar * p_lidar + t。
图像输出和 /fusion/detections_2d 共用时间戳；检测数组同时包含 YOLO 框与 LiDAR 框投影。
"""

# itertools 用于按步长读取点云，避免一次处理过多点；其余模块分别用于配置、几何、同步和图像处理。
import itertools
import json
import math
import os
import copy
import threading
from collections import deque
import rospkg
import cv2
import message_filters
import numpy as np
import rospy
import tf.transformations
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class ProjectBboxNode:
    def __init__(self):
        # 初始化 ROS 节点，并读取运行模式；cloud_only 不等待 bbox，cloud_bbox 需要三路数据。
        rospy.init_node("project_bbox_node")

        self.mode = str(rospy.get_param("~mode", "cloud_bbox")).strip().lower()
        if self.mode not in ("cloud_only", "cloud_bbox"):
            raise rospy.ROSInitException("~mode must be 'cloud_only' or 'cloud_bbox'")

        # 以 AD10 系统时间为统一时钟；拒绝使用仿真时间或 rosbag 回放时间。
        if rospy.get_param("/use_sim_time", False):
            raise rospy.ROSInitException(
                "This node requires AD10 system time; set /use_sim_time to false"
            )

        # 输入输出话题及标定文件路径均可通过 launch 参数覆盖。
        self.image_topic = rospy.get_param("~image_topic", "/oak/rgb/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/oak/rgb/camera_info")
        default_cloud_topic = (
            "/lidar/clusters" if self.mode == "cloud_only" else "/lidar/nonground_cloud"
        )
        self.cloud_topic = rospy.get_param("~cloud_topic", default_cloud_topic)
        self.bbox_topic = rospy.get_param("~bbox_topic", "/lidar/bboxes")
        self.output_topic = rospy.get_param("~output_image_topic", "/fusion/projected_image")
        self.output_detections_topic = rospy.get_param(
            "~output_detections_topic", "/fusion/detections_2d"
        )


##相对路径读取外参内参文件
        package_path = rospkg.RosPack().get_path("lidar_camera_projection")

        self.intrinsics_json = rospy.get_param(
            "~intrinsics_json",
            os.path.join(package_path, "config", "camera_intrinsics.json"),
        )


        self.extrinsic_json = rospy.get_param(
            "~extrinsic_json",
            os.path.join(package_path, "config", "extrinsic_airy_to_oak_rgb.json"),
        )



        # self.intrinsics_json = rospy.get_param(
        #     "~intrinsics_json",
        #     "/home/robot/auto/lidar_cam/results/camera_intrinsics.json",
        # )
        # self.extrinsic_json = rospy.get_param(
        #     "~extrinsic_json",
        #     "/home/robot/auto/lidar_cam/results/extrinsic_airy_to_oak_rgb.json",
        # )
        # 同步、时钟偏差、点数和绘图参数。
        # lidar_process.py 当前屏蔽了旋转，仅给点云加 lidar_offset；此处只配置该平移量，
        # 必须与 lidar_process.py 实际采用的 ~lidar_offset 一致。
        self.sync_slop_sec = float(rospy.get_param("~sync_slop_sec", 0.70))##默认0.1
        self.bbox_slop_sec = float(rospy.get_param("~bbox_slop_sec", 0.70))##默认0.2
        self.max_clock_skew_sec = float(rospy.get_param("~max_clock_skew_sec", 2.0))
        self.max_points = max(1, int(rospy.get_param("~max_points", 50000)))
        self.min_depth_m = float(rospy.get_param("~min_depth_m", 0.15))
        self.max_depth_m = float(rospy.get_param("~max_depth_m", 20.0))##去除很远和很近的点
        self.point_radius_px = max(1, int(rospy.get_param("~point_radius_px", 2)))
        self.yolo_model_path = os.path.expanduser(
            rospy.get_param("~yolo_model_path", "/home/nvidia/auto/test/yolo11n.pt")
        )
        self.yolo_image_size = int(rospy.get_param("~yolo_image_size", 640))
        self.yolo_confidence = float(rospy.get_param("~yolo_confidence", 0.25))
        self.yolo_iou = float(rospy.get_param("~yolo_iou", 0.45))
        self.yolo_device = str(rospy.get_param("~yolo_device", "0"))
        # 默认沿用相机采集时间，使图像与 Detection2DArray 可用于后续按帧关联。
        self.output_stamp_with_system_time = bool(
            rospy.get_param("~output_stamp_with_system_time", False)
        )
        self.expected_lidar_frame = str(rospy.get_param("~expected_lidar_frame", ""))
        self.undo_lidar_preprocessing = bool(
            rospy.get_param("~undo_lidar_preprocessing", True)
        )
        # 撤销 lidar_process 发布前添加的平移；默认值对应它的 ~lidar_offset 默认值。
        self.cloud_frame_translation = np.asarray(
            rospy.get_param("~cloud_frame_translation", [1.53, -0.7, -0.35]),
            dtype=np.float64,
        ).reshape(3)

        # deque 保存尚未配对的图像/点云及近期 bbox；锁保护回调间共享缓存和相机参数。
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.pending_frames = deque(maxlen=8)
        self.bbox_cache = deque(maxlen=24)

        # 先读取 JSON 标定作为可用默认值，再建立 ROS 发布、订阅和图像/点云近似同步器。
        self._load_calibration()

        if not os.path.isfile(self.yolo_model_path):
            raise rospy.ROSInitException(
                "YOLO 模型文件不存在: %s；请设置 ~yolo_model_path" % self.yolo_model_path
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise rospy.ROSInitException(
                "当前 Python 环境未安装 ultralytics；请使用 AD10 的 YOLO venv 启动: %s"
                % exc
            )
        rospy.loginfo("正在加载 YOLO 模型: %s", self.yolo_model_path)
        self.yolo_model = YOLO(self.yolo_model_path)
        rospy.loginfo(
            "YOLO 参数 | imgsz=%d conf=%.2f iou=%.2f device=%s",
            self.yolo_image_size,
            self.yolo_confidence,
            self.yolo_iou,
            self.yolo_device,
        )

        self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.detections_pub = rospy.Publisher(
            self.output_detections_topic, Detection2DArray, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1
        )

        image_sub = message_filters.Subscriber(self.image_topic, Image)
        cloud_sub = message_filters.Subscriber(self.cloud_topic, PointCloud2)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, cloud_sub], queue_size=10, slop=self.sync_slop_sec
        )




        
        self.sync.registerCallback(self._image_cloud_callback)

        self.bbox_sub = None
        if self.mode == "cloud_bbox":
            self.bbox_sub = rospy.Subscriber(
                self.bbox_topic, MarkerArray, self._bbox_callback, queue_size=10
            )

        rospy.loginfo(
            "投影融合节点启动 | mode=%s image=%s CameraInfo=%s cloud=%s bbox=%s "
            "output_image=%s output_detections=%s",
            self.mode,
            self.image_topic,
            self.camera_info_topic,
            self.cloud_topic,
            self.bbox_topic if self.mode == "cloud_bbox" else "(disabled)",
            self.output_topic,
            self.output_detections_topic,
        )
        rospy.loginfo(
            "时间参数(秒) | image-cloud同步=%.3f bbox同步=%.3f 输入时钟最大偏差=%.3f",
            self.sync_slop_sec,
            self.bbox_slop_sec,
            self.max_clock_skew_sec,
        )
        rospy.loginfo(
            "标定已加载 | K=%s | LiDAR->相机平移(m)=%s | JSON图像尺寸=%s",
            np.array2string(self.json_camera_matrix, precision=2),
            np.array2string(self.translation_camera_from_lidar, precision=4),
            self.calibration_image_size,
        )
        rospy.loginfo(
            "点云平移逆补偿 | enabled=%s | lidar_process lidar_offset(m)=%s",
            self.undo_lidar_preprocessing,
            np.array2string(self.cloud_frame_translation, precision=4),
        )


        print("投影节点初始化完成")

    def _load_calibration(self):
        """加载相机内参、畸变参数以及 Airy LiDAR 到 OAK RGB 相机的外参。"""
        try:
            with open(self.intrinsics_json, "r", encoding="utf-8") as stream:
                intrinsics = json.load(stream)
            with open(self.extrinsic_json, "r", encoding="utf-8") as stream:
                extrinsics = json.load(stream)
        except (OSError, ValueError) as exc:
            raise rospy.ROSInitException("Cannot load calibration JSON: %s" % exc)

        # 保存 JSON 内参作为 CameraInfo 尚未到达时的回退值。
        self.json_camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
        self.json_distortion = np.asarray(intrinsics.get("distortion", []), dtype=np.float64)
        size = intrinsics.get("image_size_wh", [])
        self.calibration_image_size = tuple(size[:2]) if len(size) >= 2 else None

        # 兼容标定文件的两种外参表达：R+t 分开存储，或 4x4 齐次变换矩阵。
        if "rotation_camera_from_airy" in extrinsics:
            self.rotation_camera_from_lidar = np.asarray(
                extrinsics["rotation_camera_from_airy"], dtype=np.float64
            )
            self.translation_camera_from_lidar = np.asarray(
                extrinsics["translation_camera_from_airy_m"], dtype=np.float64
            ).reshape(3)
        elif "T_camera_from_airy" in extrinsics:
            transform = np.asarray(extrinsics["T_camera_from_airy"], dtype=np.float64)
            self.rotation_camera_from_lidar = transform[:3, :3]
            self.translation_camera_from_lidar = transform[:3, 3]
        else:
            raise rospy.ROSInitException(
                "Extrinsic JSON needs rotation_camera_from_airy + "
                "translation_camera_from_airy_m, or T_camera_from_airy"
            )

        if self.json_camera_matrix.shape != (3, 3):
            raise rospy.ROSInitException("camera_matrix must be a 3x3 matrix")
        if self.rotation_camera_from_lidar.shape != (3, 3):
            raise rospy.ROSInitException("LiDAR-to-camera rotation must be 3x3")
        if not np.isfinite(self.rotation_camera_from_lidar).all() or not np.isfinite(
            self.translation_camera_from_lidar
        ).all():
            raise rospy.ROSInitException("Extrinsic calibration contains non-finite values")

        self.camera_matrix = self.json_camera_matrix.copy()
        self.distortion = self.json_distortion.copy()
        self.camera_info_received = False

    def _camera_info_callback(self, msg):
        """接收相机驱动发布的实时内参；有效时优先于 JSON 内参。"""
        matrix = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
        if msg.width <= 0 or msg.height <= 0 or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            rospy.logwarn_throttle(5.0, "CameraInfo 无效，继续使用 intrinsics JSON")
            return
        with self.lock:
            self.camera_matrix = matrix
            self.distortion = np.asarray(msg.D, dtype=np.float64)
            self.camera_info_size = (int(msg.width), int(msg.height))
            self.camera_info_received = True
        camera_info_stamp = self._stamp_sec(msg)
        if camera_info_stamp is None:
            rospy.logwarn_throttle(30.0, "CameraInfo 时间戳为零；按静态标定参数使用")
        else:
            age = rospy.Time.now().to_sec() - camera_info_stamp
            if abs(age) > self.max_clock_skew_sec:
                rospy.logwarn_throttle(
                    10.0,
                    "CameraInfo 时间戳与 AD10 系统时钟偏差 %.3f s；请核对相机驱动时钟",
                    age,
                )

    @staticmethod
    def _stamp_sec(msg):
        """取 ROS 消息 Header 时间戳；缺失或为零时返回 None。"""
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return None
        value = stamp.to_sec()
        return value if value > 0.0 else None

    @staticmethod
    def _marker_array_stamp(msg, receive_time):
        """MarkerArray 本身没有 Header，使用其中 marker 时间戳的中位数。"""
        stamps = [
            marker.header.stamp.to_sec()
            for marker in msg.markers
            if marker.header.stamp.to_sec() > 0.0
        ]
        # MarkerArray 顶层没有 Header。聚类节点遇到空点云时会发布空 MarkerArray，
        # 因此数组为空时使用 AD10 接收时间作为该消息的时间戳。
        return float(np.median(stamps)) if stamps else receive_time

    def _image_cloud_callback(self, image_msg, cloud_msg):
        """接收近似同步的图像和点云；bbox 模式下先缓存，等待匹配的 MarkerArray。"""

        print("开始订阅相机和雷达")
        image_stamp = self._stamp_sec(image_msg)
        cloud_stamp = self._stamp_sec(cloud_msg)
        if image_stamp is None or cloud_stamp is None:
            rospy.logwarn_throttle(3.0, "图像或点云 Header.stamp 为零，跳过该帧")
            return

        frame = {
            "image": image_msg,
            "cloud": cloud_msg,
            "image_stamp": image_stamp,
            "cloud_stamp": cloud_stamp,
            "pair_stamp": 0.5 * (image_stamp + cloud_stamp),
        }

        rospy.loginfo("已从相机和雷达获取配对的frame，时间间隔设置为：%.3f",self.sync_slop_sec)



        if self.mode == "cloud_only":
            self._process_frame(frame, None, None)
            return

        with self.lock:
            self.pending_frames.append(frame)
        self._match_pending_frames()

    def _bbox_callback(self, msg):
        """缓存聚类框并尝试与等待中的图像/点云帧按时间戳配对。"""
        receive_time = rospy.Time.now().to_sec()
        stamp = self._marker_array_stamp(msg, receive_time)
        with self.lock:
            self.bbox_cache.append((stamp, msg, receive_time))
        self._match_pending_frames()

    def _match_pending_frames(self):
        """在时间容差内寻找最近 bbox；过期或无法配对的帧会被丢弃，避免队列阻塞。"""
        matches = []
        with self.lock:
            while self.pending_frames and self.bbox_cache:
                frame = self.pending_frames[0]
                best_index = min(
                    range(len(self.bbox_cache)),
                    key=lambda index: abs(self.bbox_cache[index][0] - frame["pair_stamp"]),
                )
                bbox_stamp, bbox_msg, receive_time = self.bbox_cache[best_index]
                delta = abs(bbox_stamp - frame["pair_stamp"])
                if delta > self.bbox_slop_sec:
                    # 最接近的 bbox 已早于当前等待帧的容差范围，丢弃这个过期 bbox。
                    if bbox_stamp < frame["pair_stamp"] - self.bbox_slop_sec:
                        del self.bbox_cache[best_index]
                        continue
                    # 最近的 bbox 已晚于当前等待帧的容差范围；当前帧无法配对，
                    # 丢弃它以免阻塞后续帧的匹配。
                    self.pending_frames.popleft()
                    rospy.logwarn_throttle(
                        3.0,
                        "未找到时间匹配的 bbox MarkerArray，丢弃一帧等待中的图像/点云",
                    )
                    continue
                self.pending_frames.popleft()
                del self.bbox_cache[best_index]
                matches.append((frame, bbox_msg, bbox_stamp))

        for frame, bbox_msg, bbox_stamp in matches:
            self._process_frame(frame, bbox_msg, bbox_stamp)

    def _validate_timestamps(self, frame, bbox_stamp=None):
        """校验图像/点云同步差以及所有输入相对 AD10 当前 ROS 时钟的偏差。"""
        now = rospy.Time.now().to_sec()
        stamps = {
            "image": frame["image_stamp"],
            "pointcloud": frame["cloud_stamp"],
        }
        if bbox_stamp is not None:
            stamps["bbox"] = bbox_stamp

        pair_delta = abs(frame["image_stamp"] - frame["cloud_stamp"])
        if pair_delta > self.sync_slop_sec + 1e-3:
            rospy.logwarn_throttle(
                3.0,
                "图像/点云时间差 %.3f s 超过同步阈值 %.3f s，丢弃帧",
                pair_delta,
                self.sync_slop_sec,
            )
            return False

        bad = [(name, stamp, now - stamp) for name, stamp in stamps.items()
               if abs(now - stamp) > self.max_clock_skew_sec]
        if bad:
            name, stamp, skew = bad[0]
            rospy.logwarn_throttle(
                3.0,
                "%s 时间戳与 AD10 ROS 系统时钟偏差 %.3f s (stamp=%.6f now=%.6f)，丢弃帧；"
                "检查相机/雷达是否使用 AD10 时钟，或调大 ~max_clock_skew_sec",
                name,
                skew,
                stamp,
                now,
            )
            return False
        return True

    def _camera_model_for_image(self, width, height):
        """取得当前相机模型，并按实际图像分辨率缩放相机矩阵的焦距和主点。"""
        with self.lock:
            matrix = self.camera_matrix.copy()
            distortion = self.distortion.copy()
            from_topic = self.camera_info_received
            camera_info_size = getattr(self, "camera_info_size", None)

        reference_size = camera_info_size if from_topic else self.calibration_image_size
        if reference_size:
            calib_width, calib_height = reference_size
            if calib_width > 0 and calib_height > 0:
                matrix[0, :] *= float(width) / calib_width
                matrix[1, :] *= float(height) / calib_height
        return matrix, distortion, from_topic

    def _read_lidar_points(self, cloud_msg):
        """从 PointCloud2 读取 xyz；通过 stride 均匀抽取以限制每帧最大点数。"""
        total = int(cloud_msg.width) * int(cloud_msg.height)
        stride = max(1, int(math.ceil(float(total) / self.max_points)))
        points = list(
            itertools.islice(
                point_cloud2.read_points(
                    cloud_msg,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                ),
                0,
                None,
                stride,
            )
        )
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def _lidar_to_camera(self, points_lidar):
        """撤销 lidar_process 的平移（可选），再应用标定的 LiDAR 到相机外参。"""
        points_calibration_frame = points_lidar
        if self.undo_lidar_preprocessing:
            # 当前 lidar_process.py 的 debug_flag=1：不旋转，只执行
            # p_cloud = p_lidar + lidar_offset，所以逆操作是减去同一个 offset。
            points_calibration_frame = points_lidar - self.cloud_frame_translation
        return (
            points_calibration_frame @ self.rotation_camera_from_lidar.T
            + self.translation_camera_from_lidar
        )

    @staticmethod
    def _project_camera_points(points_camera, camera_matrix, distortion):
        """用 OpenCV 相机模型将相机坐标 (米) 投影到图像像素坐标。"""
        projected, _ = cv2.projectPoints(
            points_camera.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            camera_matrix,
            distortion,
        )
        return projected.reshape(-1, 2)

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

    def _draw_cloud(self, image, cloud_msg, camera_matrix, distortion):
        """筛选相机前方且深度有效的点，投影后按距离着色并绘制到图像。"""
        points_lidar = self._read_lidar_points(cloud_msg)
        if points_lidar.size == 0:
            return 0, 0
        points_camera = self._lidar_to_camera(points_lidar)
        depth = points_camera[:, 2]
        valid = (
            np.isfinite(points_camera).all(axis=1)
            & (depth >= self.min_depth_m)
            & (depth <= self.max_depth_m)
        )
        points_camera = points_camera[valid]
        depth = depth[valid]
        if points_camera.size == 0:
            return len(points_lidar), 0

        pixels = self._project_camera_points(points_camera, camera_matrix, distortion)
        height, width = image.shape[:2]
        in_image = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        pixels = pixels[in_image].astype(np.int32)
        depth = depth[in_image]
        if pixels.size == 0:
            return len(points_lidar), 0

        # 先绘制远处的点，使近处的点最后绘制并保持可见。
        order = np.argsort(depth)[::-1]
        color_values = np.clip(depth / self.max_depth_m * 255.0, 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(color_values.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
        for index in order:
            u, v = pixels[index]
            cv2.circle(
                image,
                (int(u), int(v)),
                self.point_radius_px,
                tuple(int(value) for value in colors[index]),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        return len(points_lidar), int(pixels.shape[0])

    @staticmethod
    def _marker_corners_lidar(marker):
        """根据 CUBE Marker 的尺寸、姿态和中心位置，计算 LiDAR 坐标系下的八个角点。"""
        sx, sy, sz = marker.scale.x, marker.scale.y, marker.scale.z
        if min(sx, sy, sz) <= 0.0:
            return np.empty((0, 3), dtype=np.float64)
        half = np.asarray([sx, sy, sz], dtype=np.float64) * 0.5
        corners = np.asarray(
            [
                [-half[0], -half[1], -half[2]],
                [ half[0], -half[1], -half[2]],
                [ half[0],  half[1], -half[2]],
                [-half[0],  half[1], -half[2]],
                [-half[0], -half[1],  half[2]],
                [ half[0], -half[1],  half[2]],
                [ half[0],  half[1],  half[2]],
                [-half[0],  half[1],  half[2]],
            ],
            dtype=np.float64,
        )
        q = marker.pose.orientation
        quat = np.asarray([q.x, q.y, q.z, q.w], dtype=np.float64)
        if np.linalg.norm(quat) < 1e-9:
            quat = np.asarray([0.0, 0.0, 0.0, 1.0])
        rotation = tf.transformations.quaternion_matrix(quat)[:3, :3]
        center = np.asarray(
            [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z],
            dtype=np.float64,
        )
        return corners @ rotation.T + center

    def _draw_bboxes(self, image, marker_array, camera_matrix, distortion):
        """将 MarkerArray 中的立方体框投影为绿色线框，并标出聚类 ID。"""
        height, width = image.shape[:2]
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        projected_boxes = []
        for marker in marker_array.markers:
            if marker.action not in (Marker.ADD, Marker.MODIFY) or marker.type != Marker.CUBE:
                continue
            corners_lidar = self._marker_corners_lidar(marker)
            if corners_lidar.shape != (8, 3):
                continue
            corners_camera = self._lidar_to_camera(corners_lidar)
            pixels = self._project_camera_points(corners_camera, camera_matrix, distortion)

            visible = (
                (corners_camera[:, 2] > self.min_depth_m)
                & np.isfinite(pixels).all(axis=1)
            )
            visible_pixels = pixels[visible]
            if visible_pixels.size:
                x_min = float(np.min(visible_pixels[:, 0]))
                y_min = float(np.min(visible_pixels[:, 1]))
                x_max = float(np.max(visible_pixels[:, 0]))
                y_max = float(np.max(visible_pixels[:, 1]))
                if x_max >= 0 and y_max >= 0 and x_min < width and y_min < height:
                    x1 = int(np.clip(math.floor(x_min), 0, width - 1))
                    y1 = int(np.clip(math.floor(y_min), 0, height - 1))
                    x2 = int(np.clip(math.ceil(x_max), 0, width - 1))
                    y2 = int(np.clip(math.ceil(y_max), 0, height - 1))
                    if x2 > x1 and y2 > y1:
                        projected_boxes.append((int(marker.id), (x1, y1, x2, y2)))

            for start, end in edges:
                p0 = corners_camera[start]
                p1 = corners_camera[end]
                if p0[2] <= self.min_depth_m or p1[2] <= self.min_depth_m:
                    continue
                u0, v0 = pixels[start]
                u1, v1 = pixels[end]
                if not np.isfinite([u0, v0, u1, v1]).all():
                    continue
                cv2.line(
                    image,
                    (int(round(u0)), int(round(v0))),
                    (int(round(u1)), int(round(v1))),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if np.any(visible):
                center_px = np.mean(pixels[visible], axis=0)
                if np.isfinite(center_px).all():
                    u = int(np.clip(round(center_px[0]), 0, width - 1))
                    v = int(np.clip(round(center_px[1]), 0, height - 1))
                    cv2.putText(
                        image,
                        "cluster %d" % marker.id,
                        (u, v),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
        return projected_boxes

    def _process_frame(self, frame, bbox_msg, bbox_stamp):
        """完成单帧处理：时间检查、图像解码、点云/框绘制、结果发布及耗时信息输出。"""
        if not self._validate_timestamps(frame, bbox_stamp):
            return

        try:
            image = self.bridge.imgmsg_to_cv2(frame["image"], desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(3.0, "图像转换失败: %s", exc)
            return

        height, width = image.shape[:2]
        camera_matrix, distortion, from_topic = self._camera_model_for_image(width, height)
        if not from_topic:
            rospy.logwarn_throttle(
                10.0,
                "尚未收到有效 CameraInfo，使用 intrinsics JSON 中的相机内参/畸变参数",
            )

        cloud_frame = frame["cloud"].header.frame_id
        if self.expected_lidar_frame and cloud_frame != self.expected_lidar_frame:
            rospy.logwarn_throttle(
                5.0,
                "点云 frame_id='%s'，标定源坐标系期望='%s'；确认两者坐标轴定义一致",
                cloud_frame,
                self.expected_lidar_frame,
            )

        # 先在相机原图上运行 YOLO 并绘制 2D 检测框；随后叠加 LiDAR 点云和 3D 框投影。
        try:
            image, yolo_boxes = self._run_yolo(image)
        except Exception as exc:
            rospy.logerr_throttle(5.0, "YOLO 推理失败，本帧仍发布 LiDAR 投影: %s", exc)
            yolo_boxes = []

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
                    "yolo",
                    index,
                    class_id,
                    class_name,
                    score,
                    xyxy,
                )
            )

        total_points, projected_points = self._draw_cloud(
            image, frame["cloud"], camera_matrix, distortion
        )
        projected_boxes = []
        if bbox_msg is not None:
            projected_boxes = self._draw_bboxes(
                image, bbox_msg, camera_matrix, distortion
            )
            for marker_id, xyxy in projected_boxes:
                detection_array.detections.append(
                    self._make_2d_detection(
                        output_header,
                        "lidar_cluster",
                        marker_id,
                        -1,
                        "lidar_cluster",
                        1.0,
                        xyxy,
                    )
                )

        output = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        output.header = copy.deepcopy(output_header)
        self.image_pub.publish(output)
        self.detections_pub.publish(detection_array)
        rospy.loginfo("投影图像已发布在:%s",self.output_topic)
        rospy.loginfo("YOLO 与 LiDAR 2D 框已发布在:%s", self.output_detections_topic)

        bbox_delta = (
            abs(bbox_stamp - frame["pair_stamp"]) if bbox_stamp is not None else 0.0
        )
        rospy.loginfo_throttle(
            2.0,
            "投影完成 | mode=%s points=%d projected=%d lidar_boxes=%d yolo_boxes=%d "
            "image-cloud_dt=%.3f s "
            "bbox_dt=%.3f s image_age=%.3f s cloud_age=%.3f s output_stamp=%s",
            self.mode,
            total_points,
            projected_points,
            len(projected_boxes),
            len(yolo_boxes),
            abs(frame["image_stamp"] - frame["cloud_stamp"]),
            bbox_delta,
            now.to_sec() - frame["image_stamp"],
            now.to_sec() - frame["cloud_stamp"],
            "AD10_now" if self.output_stamp_with_system_time else "camera_image_stamp",
        )


if __name__ == "__main__":
    # 独立运行节点并保持 ROS 回调循环；收到 ROS 中断后正常退出。
    try:
        ProjectBboxNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
