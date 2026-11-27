"""相机标定加载、CameraInfo 更新和时间戳校验。"""

import json

import numpy as np
import rospy


class CalibrationMixin:
    @staticmethod
    def _six_values(param_name, values):
        """校验相机配置参数为恰好六项。"""
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise rospy.ROSInitException("%s 必须包含恰好六项" % param_name)
        return [str(value) for value in values]

    def _load_calibrations(self):
        """逐相机加载 K/D 和 Airy LiDAR 到相机坐标系的 R/t。"""
        camera_models = []
        for index, (intrinsics_path, extrinsics_path) in enumerate(
            zip(self.intrinsics_jsons, self.extrinsics_jsons)
        ):
            try:
                with open(intrinsics_path, "r", encoding="utf-8") as stream:
                    intrinsics = json.load(stream)
                with open(extrinsics_path, "r", encoding="utf-8") as stream:
                    extrinsics = json.load(stream)
            except (OSError, ValueError) as exc:
                raise rospy.ROSInitException(
                    "%s 标定文件读取失败 (K=%s, 外参=%s): %s"
                    % (self.camera_names[index], intrinsics_path, extrinsics_path, exc)
                )

            camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(intrinsics.get("distortion", []), dtype=np.float64)
            size = intrinsics.get("image_size_wh", [])
            calibration_image_size = tuple(size[:2]) if len(size) >= 2 else None

            # 兼容原标定 JSON 的 R+t 与 4x4 齐次矩阵两种表达。
            if "rotation_camera_from_airy" in extrinsics:
                rotation = np.asarray(
                    extrinsics["rotation_camera_from_airy"], dtype=np.float64
                )
                translation = np.asarray(
                    extrinsics["translation_camera_from_airy_m"], dtype=np.float64
                ).reshape(3)
            elif "T_camera_from_airy" in extrinsics:
                transform = np.asarray(extrinsics["T_camera_from_airy"], dtype=np.float64)
                if transform.shape != (4, 4):
                    raise rospy.ROSInitException(
                        "%s T_camera_from_airy 必须是 4x4"
                        % self.camera_names[index]
                    )
                rotation = transform[:3, :3]
                translation = transform[:3, 3]
            else:
                raise rospy.ROSInitException(
                    "%s 外参 JSON 缺少 R/t 或 T_camera_from_airy"
                    % self.camera_names[index]
                )

            if camera_matrix.shape != (3, 3):
                raise rospy.ROSInitException(
                    "%s camera_matrix 必须是 3x3" % self.camera_names[index]
                )
            if rotation.shape != (3, 3):
                raise rospy.ROSInitException(
                    "%s LiDAR->camera rotation 必须是 3x3" % self.camera_names[index]
                )
            if not np.isfinite(camera_matrix).all() or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                raise rospy.ROSInitException(
                    "%s 标定文件包含非有限数值" % self.camera_names[index]
                )

            camera_models.append(
                {
                    "json_camera_matrix": camera_matrix,
                    "json_distortion": distortion,
                    "calibration_image_size": calibration_image_size,
                    "rotation_camera_from_lidar": rotation,
                    "translation_camera_from_lidar": translation,
                    "camera_matrix": camera_matrix.copy(),
                    "distortion": distortion.copy(),
                    "camera_info_received": False,
                }
            )
        return camera_models

    def _camera_info_callback(self, camera_index, msg):
        """接收该相机的实时内参；有效时优先于对应 JSON 内参。"""
        matrix = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
        if msg.width <= 0 or msg.height <= 0 or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            rospy.logwarn_throttle(
                5.0,
                "%s CameraInfo 无效，继续使用 intrinsics JSON",
                self.camera_names[camera_index],
            )
            return
        with self.lock:
            model = self.camera_models[camera_index]
            model["camera_matrix"] = matrix
            model["distortion"] = np.asarray(msg.D, dtype=np.float64)
            model["camera_info_size"] = (int(msg.width), int(msg.height))
            model["camera_info_received"] = True
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

    def _validate_timestamps(self, frame, bbox_stamp=None):
        """校验图像时间戳，以及存在 LiDAR 时的同步差和时钟偏差。"""
        now = rospy.Time.now().to_sec()
        stamps = {"image": frame["image_stamp"]}
        cloud_stamp = frame.get("cloud_stamp")
        if cloud_stamp is not None:
            stamps["pointcloud"] = cloud_stamp
        if bbox_stamp is not None:
            stamps["bbox"] = bbox_stamp

        if cloud_stamp is not None:
            pair_delta = abs(frame["image_stamp"] - cloud_stamp)
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

    def _camera_model_for_image(self, camera_index, width, height):
        """取得当前相机模型，并按实际图像分辨率缩放相机矩阵的焦距和主点。"""
        with self.lock:
            model = self.camera_models[camera_index]
            matrix = model["camera_matrix"].copy()
            distortion = model["distortion"].copy()
            from_topic = model["camera_info_received"]
            camera_info_size = model.get("camera_info_size")

        reference_size = (
            camera_info_size if from_topic else model["calibration_image_size"]
        )
        if reference_size:
            calib_width, calib_height = reference_size
            if calib_width > 0 and calib_height > 0:
                matrix[0, :] *= float(width) / calib_width
                matrix[1, :] *= float(height) / calib_height
        return matrix, distortion, from_topic
