"""六相机 LiDAR 与 YOLO 融合 ROS 节点实现。

cloud_only：投影聚类点云（默认 /lidar/clusters）。
cloud_bbox：投影非地面点云（默认 /lidar/nonground_cloud）及时间匹配的 Marker 框。
外参方向为 p_camera = R_camera_from_lidar * p_lidar + t。
六路相机都发布 YOLO 图像和检测数组；默认首尾相机叠加 LiDAR 投影。
检测 ID 带 camera_N 前缀以区分来源。
"""

import json
import os
import threading
import time
from collections import OrderedDict, deque

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray

from .calibration import CalibrationMixin
from .projection import LidarProjectionMixin
from .monitoring import PerformanceMonitor, ResourceMonitor
from .scheduling import LatestFrameBuffer
from .yolo_fusion import YoloFusionMixin


class ProjectBboxNode(CalibrationMixin, LidarProjectionMixin, YoloFusionMixin):
    def __init__(self):
        startup_started = time.perf_counter()
        # 初始化 ROS 节点并读取运行模式；cloud_bbox 模式会尝试叠加时间匹配的 bbox。
        rospy.init_node("project_bbox_node")

        self.mode = str(rospy.get_param("~mode", "cloud_bbox")).strip().lower()
        if self.mode not in ("cloud_only", "cloud_bbox"):
            raise rospy.ROSInitException("~mode must be 'cloud_only' or 'cloud_bbox'")

        # 以 AD10 系统时间为统一时钟；拒绝使用仿真时间或 rosbag 回放时间。
        if rospy.get_param("/use_sim_time", False):
            raise rospy.ROSInitException(
                "This node requires AD10 system time; set /use_sim_time to false"
            )

        # 六路图像都运行 YOLO；默认话题对应真实 OAK 加五路模拟话题。
        self.camera_count = 6
        self.camera_names = ["camera_%d" % (index + 1) for index in range(self.camera_count)]
        projection_cameras = rospy.get_param("~projection_cameras", [1, 6])##从yaml加载
        if (not isinstance(projection_cameras, list)
                or any(type(index) is not int or not 1 <= index <= 6
                       for index in projection_cameras)):
            raise rospy.ROSInitException("~projection_cameras 必须为 1..6 的列表，可用 [] 关闭投影")
        self.projection_camera_indices = {index - 1 for index in projection_cameras}
        base_image_topic = rospy.get_param("~image_topic", "/oak/rgb/image_raw")
        default_image_topics = [base_image_topic] + [
            "%s_%d" % (base_image_topic.rstrip("/"), index)
            for index in range(1, self.camera_count)
        ]
        self.image_topics = self._six_values(
            "~camera_image_topics",
            rospy.get_param("~camera_image_topics", default_image_topics),
        )

        # 默认模拟图像共用真实相机的 CameraInfo；真实多相机时可分别配置六个话题。
        base_camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/oak/rgb/camera_info"
        )
        self.camera_info_topics = self._six_values(
            "~camera_info_topics",
            rospy.get_param(
                "~camera_info_topics", [base_camera_info_topic] * self.camera_count
            ),
        )

        default_cloud_topic = (
            "/lidar/clusters" if self.mode == "cloud_only" else "/lidar/nonground_cloud"
        )
        self.cloud_topic = rospy.get_param("~cloud_topic", default_cloud_topic)
        self.bbox_topic = rospy.get_param("~bbox_topic", "/lidar/bboxes")
        base_output_topic = rospy.get_param(
            "~output_image_topic", "/fusion/projected_image"
        )
        base_detections_topic = rospy.get_param(
            "~output_detections_topic", "/fusion/detections_2d"
        )
        default_output_topics = [base_output_topic] + [
            "%s_%d" % (base_output_topic.rstrip("/"), index)
            for index in range(2, self.camera_count + 1)
        ]
        default_detection_topics = [base_detections_topic] + [
            "%s_%d" % (base_detections_topic.rstrip("/"), index)
            for index in range(2, self.camera_count + 1)
        ]
        self.output_topics = self._six_values(
            "~output_image_topics",
            rospy.get_param("~output_image_topics", default_output_topics),
        )
        self.output_detections_topics = self._six_values(
            "~output_detections_topics",
            rospy.get_param("~output_detections_topics", default_detection_topics),
        )

        self.output_yolo_detections_topics = self._six_values(
            "~output_yolo_detections_topics",
            rospy.get_param("~output_yolo_detections_topics",
                            [topic + "/yolo" for topic in self.output_detections_topics]),
        )
        # 纯 YOLO 与融合结果必须使用不同话题，避免同一帧出现两种语义。
        detection_topics = self.output_detections_topics + self.output_yolo_detections_topics
        if len({rospy.resolve_name(topic) for topic in detection_topics}) != len(detection_topics):
            raise rospy.ROSInitException("融合/纯 YOLO 检测话题必须各路独立且互不重复")

        # 标定文件放在脚本旁的 config 目录；支持旧的单文件参数作为六路共享回退。
        self.config_dir = os.path.expanduser(
            rospy.get_param(
                "~config_dir",
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config",
                ),
            )
        )
        default_intrinsics = [
            os.path.join(self.config_dir, "camera_%d_intrinsics.json" % (index + 1))
            for index in range(self.camera_count)
        ]
        default_extrinsics = [
            os.path.join(self.config_dir, "camera_%d_extrinsics.json" % (index + 1))
            for index in range(self.camera_count)
        ]
        legacy_intrinsics = rospy.get_param("~intrinsics_json", None)
        legacy_extrinsics = rospy.get_param("~extrinsic_json", None)
        self.intrinsics_jsons = self._six_values(
            "~intrinsics_jsons",
            rospy.get_param(
                "~intrinsics_jsons",
                [legacy_intrinsics] * self.camera_count
                if legacy_intrinsics
                else default_intrinsics,
            ),
        )
        self.extrinsics_jsons = self._six_values(
            "~extrinsics_jsons",
            rospy.get_param(
                "~extrinsics_jsons",
                [legacy_extrinsics] * self.camera_count
                if legacy_extrinsics
                else default_extrinsics,
            ),
        )
        # 取图像时间之前最近的点云，并限制其最大年龄。兼容旧 launch 中的
        # ~sync_slop_sec 参数；新配置应使用语义更准确的 ~max_lidar_age_sec。
        # lidar_process.py 当前屏蔽了旋转，仅给点云加 lidar_offset；此处只配置该平移量，
        # 必须与 lidar_process.py 实际采用的 ~lidar_offset 一致。
        legacy_sync_slop = float(rospy.get_param("~sync_slop_sec", 0.30))
        self.max_lidar_age_sec = max(
            0.0,
            float(rospy.get_param("~max_lidar_age_sec", legacy_sync_slop)),
        )
        self.bbox_slop_sec = max(0.0, float(rospy.get_param("~bbox_slop_sec", 0.10)))
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

        # 缓存近期带时间戳的 LiDAR 数据；图像回调取最近过去帧，不等待 LiDAR。
        # 三阶段各用每路单槽缓存；耗时操作在锁外执行。
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.lidar_cache = deque(
            maxlen=max(1, int(rospy.get_param("~lidar_cache_size", 12)))
        )
        self.bbox_cache = deque(
            maxlen=max(1, int(rospy.get_param("~bbox_cache_size", 24)))
        )
        self.last_cloud_stamp_by_camera = [None] * self.camera_count
        self.projected_cloud_cache = OrderedDict()
        self.projected_cloud_cache_size = max(
            1, int(rospy.get_param("~projected_cloud_cache_size", 16))
        )
        self.input_buffer = LatestFrameBuffer(self.camera_count)
        self.projection_buffer = LatestFrameBuffer(self.camera_count)
        self.visualization_buffer = LatestFrameBuffer(self.camera_count)
        self.frame_versions = [0] * self.camera_count
        self.frame_version_lock = threading.Lock()
        self.last_output_versions = [-1] * self.camera_count
        self.stop_worker = threading.Event()
        self.max_output_age_sec = max(0.0, float(rospy.get_param("~max_output_age_sec", 0.5)))
        self.monitor_interval_sec = max(0.1, float(rospy.get_param("~monitor_interval_sec", 1.0)))
        self.monitor_log_interval_sec = max(0.0, float(rospy.get_param(
            "~monitor_log_interval_sec", rospy.get_param("~timing_log_interval_sec", 2.0))))
        self.monitor_topic = rospy.get_param("~monitor_topic", "/fusion/performance")
        self.metrics = PerformanceMonitor(self.camera_names, float(rospy.get_param(
            "~monitor_window_sec", 5.0)))
        self.resources = ResourceMonitor(
            str(rospy.get_param("~gpu_load_path", "")),
            float(rospy.get_param("~gpu_load_scale", 0.1)),
        )
        if rospy.has_param("~inference_queue_size"):
            rospy.logwarn("~inference_queue_size 已停用：现在每路仅保留最新待处理帧")

        # 先读取 JSON 标定作为可用默认值，再建立 ROS 发布、订阅和时间缓存。
        calibration_started = time.perf_counter()
        self.camera_models = self._load_calibrations()
        self.calibration_load_ms = (time.perf_counter() - calibration_started) * 1000.0
        rospy.loginfo("模块计时 | 六路标定加载=%.2f ms", self.calibration_load_ms)

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
        model_load_started = time.perf_counter()
        self.yolo_model = YOLO(self.yolo_model_path)
        self.yolo_model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
        rospy.loginfo("模块计时 | YOLO 模型加载=%.2f ms", self.yolo_model_load_ms)
        rospy.loginfo(
            "YOLO 参数 | imgsz=%d conf=%.2f iou=%.2f device=%s",
            self.yolo_image_size,
            self.yolo_confidence,
            self.yolo_iou,
            self.yolo_device,
        )

        ros_setup_started = time.perf_counter()
        self.image_pubs = [
            rospy.Publisher(topic, Image, queue_size=1) for topic in self.output_topics
        ]
        self.detections_pubs = [
            rospy.Publisher(topic, Detection2DArray, queue_size=1)
            for topic in self.output_detections_topics
        ]
        self.yolo_detections_pubs = [
            rospy.Publisher(topic, Detection2DArray, queue_size=1)
            for topic in self.output_yolo_detections_topics
        ]
        self.monitor_pub = rospy.Publisher(self.monitor_topic, String, queue_size=1)
        self.camera_info_subs = [
            rospy.Subscriber(
                topic,
                CameraInfo,
                lambda msg, camera_index=index: self._camera_info_callback(
                    camera_index, msg
                ),
                queue_size=1,
            )
            for index, topic in enumerate(self.camera_info_topics)
        ]

        self.cloud_sub = rospy.Subscriber(
            self.cloud_topic, PointCloud2, self._cloud_callback, queue_size=10
        )
        self.image_subs = []
        # 六路图像都直接触发处理；首尾两路从 LiDAR 缓存中取最近过去帧。
        for camera_index, topic in enumerate(self.image_topics):
            image_sub = rospy.Subscriber(
                topic,
                Image,
                lambda image_msg, index=camera_index: self._image_callback(
                    index, image_msg
                ),
                queue_size=1,
            )
            self.image_subs.append(image_sub)

        self.bbox_sub = None
        if self.mode == "cloud_bbox":
            self.bbox_sub = rospy.Subscriber(
                self.bbox_topic, MarkerArray, self._bbox_callback, queue_size=10
            )
        self.ros_setup_ms = (time.perf_counter() - ros_setup_started) * 1000.0
        rospy.loginfo("模块计时 | ROS 发布/订阅初始化=%.2f ms", self.ros_setup_ms)

        self.worker_threads = [
            threading.Thread(target=target, name=name, daemon=True)
            for name, target in (
                ("yolo_inference_worker", self._inference_worker),
                ("lidar_projection_worker", self._projection_worker),
                ("visualization_worker", self._visualization_worker),
                ("performance_monitor", self._monitor_worker),
            )
        ]
        self.worker_thread = self.worker_threads[0]  # 兼容旧调试代码。
        rospy.on_shutdown(self._shutdown_worker)
        for thread in self.worker_threads:
            thread.start()
        rospy.loginfo("流水线已启动 | 投影相机=%s 监控=%s 纯YOLO检测=%s",
                      projection_cameras, self.monitor_topic,
                      ", ".join(self.output_yolo_detections_topics))

        rospy.loginfo(
            "六相机投影融合节点启动 | mode=%s images=%s CameraInfo=%s cloud=%s bbox=%s "
            "output_images=%s output_detections=%s",
            self.mode,
            ", ".join(self.image_topics),
            ", ".join(self.camera_info_topics),
            self.cloud_topic,
            self.bbox_topic if self.mode == "cloud_bbox" else "(disabled)",
            ", ".join(self.output_topics),
            ", ".join(self.output_detections_topics),
        )
        rospy.loginfo(
            "时间参数(秒) | 图像可复用LiDAR最大年龄=%.3f cloud-bbox匹配=%.3f 输入时钟最大偏差=%.3f",
            self.max_lidar_age_sec,
            self.bbox_slop_sec,
            self.max_clock_skew_sec,
        )
        for index, (camera_name, model) in enumerate(
            zip(self.camera_names, self.camera_models)
        ):
            rospy.loginfo(
                "%s 标定 | K=%s | LiDAR->相机平移(m)=%s | JSON图像尺寸=%s | K文件=%s | 外参文件=%s",
                camera_name,
                np.array2string(model["json_camera_matrix"], precision=2),
                np.array2string(model["translation_camera_from_lidar"], precision=4),
                model["calibration_image_size"],
                self.intrinsics_jsons[index],
                self.extrinsics_jsons[index],
            )
        rospy.loginfo(
            "点云平移逆补偿 | enabled=%s | lidar_process lidar_offset(m)=%s",
            self.undo_lidar_preprocessing,
            np.array2string(self.cloud_frame_translation, precision=4),
        )


        print("投影节点初始化完成")
        rospy.loginfo(
            "模块计时 | 节点初始化总耗时=%.2f ms",
            (time.perf_counter() - startup_started) * 1000.0,
        )

    def _cloud_callback(self, cloud_msg):
        """将带时间戳的点云存入有界缓存，供后续每张图像独立查找。"""
        callback_started = time.perf_counter()
        cloud_stamp = self._stamp_sec(cloud_msg)
        if cloud_stamp is None:
            rospy.logwarn_throttle(
                3.0,
                "LiDAR 点云 Header.stamp 为零，不能按图像时间复用该帧",
            )
            return

        with self.lock:
            self._insert_timestamped(self.lidar_cache, (cloud_stamp, cloud_msg))
            cache_size = len(self.lidar_cache)
        rospy.logdebug_throttle(
            5.0,
            "LiDAR 缓存收到点云 | stamp=%.6f cache=%d callback=%.2f ms",
            cloud_stamp,
            cache_size,
            (time.perf_counter() - callback_started) * 1000.0,
        )

    @staticmethod
    def _insert_timestamped(cache, item):
        """按 Header 时间戳有序插入有界 deque，兼容少量乱序消息。"""
        entries = list(cache)
        stamp = item[0]
        index = 0
        while index < len(entries) and entries[index][0] <= stamp:
            index += 1
        entries.insert(index, item)
        cache.clear()
        cache.extend(entries[-cache.maxlen :])

    def _select_lidar_for_image(self, camera_index, image_stamp):
        """选择不晚于图像的最新点云，并选择对应时间附近的 bbox。"""
        with self.lock:
            selected_cloud = next(
                (
                    (stamp, msg)
                    for stamp, msg in reversed(self.lidar_cache)
                    if stamp <= image_stamp
                ),
                None,
            )
            if selected_cloud is None:
                return None, None, None, None, False

            cloud_stamp, cloud_msg = selected_cloud
            cloud_age = image_stamp - cloud_stamp
            if cloud_age > self.max_lidar_age_sec:
                return None, None, None, None, False

            bbox_msg = None
            bbox_stamp = None
            if self.mode == "cloud_bbox" and self.bbox_cache:
                bbox_candidates = [
                    (stamp, msg)
                    for stamp, msg, _receive_time in self.bbox_cache
                    if stamp <= image_stamp
                    and abs(stamp - cloud_stamp) <= self.bbox_slop_sec
                ]
                if bbox_candidates:
                    bbox_stamp, bbox_msg = min(
                        bbox_candidates,
                        key=lambda item: abs(item[0] - cloud_stamp),
                    )

            reused = self.last_cloud_stamp_by_camera[camera_index] == cloud_stamp
            self.last_cloud_stamp_by_camera[camera_index] = cloud_stamp
            return cloud_msg, cloud_stamp, bbox_msg, bbox_stamp, reused

    def _image_callback(self, camera_index, image_msg):
        """保存每路最新图像；启用投影的相机按时间戳附加可用 LiDAR 引用。"""
        callback_started = time.perf_counter()
        if self.stop_worker.is_set():
            return
        # 在匹配 LiDAR 之前确定到达顺序，慢回调不能取得更大的版本覆盖新帧。
        frame_version = self._reserve_frame_version(camera_index)
        self.metrics.increment(camera_index, "input")
        image_stamp = self._stamp_sec(image_msg)
        if image_stamp is None:
            self.metrics.increment(camera_index, "invalid_stamp")
            rospy.logwarn_throttle(
                3.0,
                "%s 图像 Header.stamp 为零，跳过该帧",
                self.camera_names[camera_index],
            )
            return

        cloud_msg = None
        cloud_stamp = None
        bbox_msg = None
        bbox_stamp = None
        cloud_reused = False
        bbox_lookup_ms = 0.0
        if camera_index in self.projection_camera_indices:
            lookup_started = time.perf_counter()
            (
                cloud_msg,
                cloud_stamp,
                bbox_msg,
                bbox_stamp,
                cloud_reused,
            ) = self._select_lidar_for_image(camera_index, image_stamp)
            bbox_lookup_ms = (time.perf_counter() - lookup_started) * 1000.0
            if cloud_msg is None:
                self.metrics.increment(camera_index, "lidar_unavailable")
                rospy.logwarn_throttle(
                    5.0,
                    "%s 没有可用的近期历史点云，本帧继续发布 YOLO 结果但跳过 LiDAR 投影",
                    self.camera_names[camera_index],
                )
            elif self.mode == "cloud_bbox" and bbox_msg is None:
                rospy.logwarn_throttle(
                    5.0,
                    "%s 没有找到与当前点云时间匹配的 bbox；本帧仍投影点云并发布 YOLO 结果",
                    self.camera_names[camera_index],
                )

        frame = {
            "camera_index": camera_index,
            "frame_version": frame_version,
            "image": image_msg,
            "cloud": cloud_msg,
            "image_stamp": image_stamp,
            "cloud_stamp": cloud_stamp,
            "bbox_stamp": bbox_stamp,
            "cloud_reused": cloud_reused,
            "bbox_match_ms": bbox_lookup_ms,
            "received_monotonic": callback_started,
            "input_callback_ms": (time.perf_counter() - callback_started) * 1000.0,
        }
        self._enqueue_frame(frame, bbox_msg, bbox_stamp)

    def _bbox_callback(self, msg):
        """缓存聚类框；图像回调会按所选点云的时间戳查找最近匹配框。"""
        callback_started = time.perf_counter()
        receive_time = rospy.Time.now().to_sec()
        stamp = self._marker_array_stamp(msg, receive_time)
        with self.lock:
            self._insert_timestamped(
                self.bbox_cache, (stamp, msg, receive_time)
            )
            cache_size = len(self.bbox_cache)
        rospy.logdebug_throttle(
            5.0,
            "模块计时 | bbox 回调=%.2f ms markers=%d cache=%d",
            (time.perf_counter() - callback_started) * 1000.0,
            len(msg.markers),
            cache_size,
        )

    def _reserve_frame_version(self, camera_index):
        with self.frame_version_lock:
            self.frame_versions[camera_index] += 1
            return self.frame_versions[camera_index]

    def _enqueue_frame(self, frame, bbox_msg, bbox_stamp):
        frame["bbox"] = bbox_msg
        frame["bbox_stamp"] = bbox_stamp
        index = frame["camera_index"]
        if "frame_version" not in frame:
            frame["frame_version"] = self._reserve_frame_version(index)
        for name in ("input_callback_ms", "bbox_match_ms"):
            self.metrics.observe(index, name, frame.get(name))
        self._submit_task(self.input_buffer, {"frame": frame}, "input")

    def _submit_task(self, buffer, task, stage):
        if self.stop_worker.is_set():
            return
        frame = task["frame"]
        task[stage + "_enqueued_monotonic"] = time.perf_counter()
        previous = buffer.put(frame["camera_index"], task, frame["frame_version"])
        if previous is not None:
            reason = "_stale_dropped" if previous is task else "_overwritten"
            self.metrics.increment(frame["camera_index"], stage + reason)

    def _task_expired(self, task, stage):
        if self.stop_worker.is_set():
            return True
        frame = task["frame"]
        age = time.perf_counter() - frame["received_monotonic"]
        if self.max_output_age_sec and age > self.max_output_age_sec:
            self.metrics.increment(frame["camera_index"], stage + "_expired")
            return True
        return False

    def _run_stage(self, buffer, stage, process):
        while not self.stop_worker.is_set() and not rospy.is_shutdown():
            task = buffer.get(timeout=0.1)
            if task is None:
                continue
            frame = task["frame"]
            index = frame["camera_index"]
            wait_ms = (time.perf_counter() - task[stage + "_enqueued_monotonic"]) * 1000.0
            self.metrics.observe(index, stage + "_queue_wait_ms", wait_ms)
            if stage == "input":
                self.metrics.observe(index, "queue_wait_ms", wait_ms)
            if self._task_expired(task, stage):
                continue
            try:
                process(task)
            except Exception as exc:
                self.metrics.increment(index, stage + "_errors")
                rospy.logerr_throttle(3.0, "%s %s 阶段失败: %s",
                                     self.camera_names[index], stage, exc)

    def _inference_worker(self):
        self._run_stage(self.input_buffer, "input", lambda task: self._process_frame(
            task["frame"], task["frame"].get("bbox"), task["frame"].get("bbox_stamp")))

    def _projection_worker(self):
        self._run_stage(self.projection_buffer, "projection", self._process_projection)

    def _visualization_worker(self):
        self._run_stage(self.visualization_buffer, "visualization", self._process_visualization)

    def _monitor_worker(self):
        last_log = 0.0
        while not self.stop_worker.is_set() and not rospy.is_shutdown():
            try:
                resource_sample = self.resources.sample()
                snapshot = self.metrics.snapshot()
                snapshot["schema_version"] = 1
                snapshot["stamp"] = rospy.Time.now().to_sec()
                snapshot["resources"] = resource_sample
                snapshot["pending"] = {name: buffer.snapshot() for name, buffer in (
                    ("input", self.input_buffer), ("projection", self.projection_buffer),
                    ("visualization", self.visualization_buffer))}
                self.monitor_pub.publish(String(data=json.dumps(snapshot, allow_nan=False)))
                now = time.perf_counter()
                if self.monitor_log_interval_sec and now - last_log >= self.monitor_log_interval_sec:
                    last_log = now
                    self._log_monitor(snapshot)
            except Exception as exc:
                rospy.logerr_throttle(5.0, "性能监控采样失败: %s", exc)
            if self.stop_worker.wait(self.monitor_interval_sec):
                break

    @staticmethod
    def _metric_text(value):
        return "n/a" if value is None else "%.1f" % value

    def _log_monitor(self, snapshot):
        for camera in snapshot["cameras"]:
            fps = camera["fps"]
            ages = camera["observations"].get("image_age_ms", {})
            cache = camera["projection_cache"]["hit_rate"]
            drops = {key: value for key, value in camera["counters"]["window"].items()
                     if any(word in key for word in ("overwritten", "dropped", "expired", "errors"))}
            rospy.loginfo("监控 %s | FPS in/yolo/out=%.1f/%.1f/%.1f "
                          "image_age_ms P50/P95=%s/%s cache_hit=%s%% drops/errors=%s",
                          camera["name"], fps["input"], fps["yolo"], fps["output"],
                          self._metric_text(ages.get("p50")), self._metric_text(ages.get("p95")),
                          self._metric_text(cache * 100.0 if cache is not None else None), drops)
        resources = snapshot["resources"]
        rospy.loginfo("资源 | CPU=%s%% process_CPU=%s%% GPU=%s%% RAM=%s%% total_YOLO_FPS=%.1f",
                      self._metric_text(resources["cpu_percent"]),
                      self._metric_text(resources["process_cpu_percent"]),
                      self._metric_text(resources["gpu"]["percent"]),
                      self._metric_text(resources["ram"].get("used_percent")),
                      snapshot["total_yolo_fps"])

    def _shutdown_worker(self):
        self.stop_worker.set()
        for subscriber in getattr(self, "image_subs", []):
            subscriber.unregister()
        for buffer in (self.input_buffer, self.projection_buffer, self.visualization_buffer):
            buffer.close()
        deadline = time.perf_counter() + 3.0
        for thread in self.worker_threads:
            if thread is threading.current_thread() or thread.ident is None:
                continue
            thread.join(timeout=max(0.0, deadline - time.perf_counter()))
            if thread.is_alive():
                rospy.logwarn("关闭时线程尚未返回（推理/原生库调用不可强制中断）: %s", thread.name)
