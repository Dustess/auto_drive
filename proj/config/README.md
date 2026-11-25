# 六路相机标定文件

节点默认按 `camera_1` 到 `camera_6` 加载以下文件对：

- `camera_N_intrinsics.json`
- `camera_N_extrinsics.json`

当前六对文件分别从 `lidar_cam/results/camera_intrinsics.json` 和
`lidar_cam/results/extrinsic_airy_to_oak_rgb.json` 复制，格式和字段保持不变。
接入真实多相机后，应逐路替换为该相机的内参，以及同一 LiDAR 坐标系到该相机坐标系的外参。

默认图像话题为 `/oak/rgb/image_raw`、`/oak/rgb/image_raw_1` 至
`/oak/rgb/image_raw_5`。可通过 ROS 私有参数 `~camera_image_topics`、
`~camera_info_topics`、`~output_image_topics` 和 `~output_detections_topics`
分别覆盖六路话题列表。
