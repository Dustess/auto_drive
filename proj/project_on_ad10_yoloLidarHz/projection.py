"""LiDAR 点云与聚类框的相机投影和绘制函数。"""

import itertools
import math

import cv2
import numpy as np
import rospy
import tf.transformations
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker


class LidarProjectionMixin:
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

    def _lidar_to_camera(self, camera_index, points_lidar):
        """撤销 lidar_process 的平移（可选），再应用标定的 LiDAR 到相机外参。"""
        points_calibration_frame = points_lidar
        if self.undo_lidar_preprocessing:
            # 当前 lidar_process.py 的 debug_flag=1：不旋转，只执行
            # p_cloud = p_lidar + lidar_offset，所以逆操作是减去同一个 offset。
            points_calibration_frame = points_lidar - self.cloud_frame_translation
        model = self.camera_models[camera_index]
        return (
            points_calibration_frame @ model["rotation_camera_from_lidar"].T
            + model["translation_camera_from_lidar"]
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

    def _draw_cloud(self, image, camera_index, cloud_msg, camera_matrix, distortion):
        """筛选相机前方且深度有效的点，投影后按距离着色并绘制到图像。"""
        points_lidar = self._read_lidar_points(cloud_msg)
        if points_lidar.size == 0:
            return 0, 0
        points_camera = self._lidar_to_camera(camera_index, points_lidar)
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

    def _draw_bboxes(
        self, image, camera_index, marker_array, camera_matrix, distortion
    ):
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
            corners_camera = self._lidar_to_camera(camera_index, corners_lidar)
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
