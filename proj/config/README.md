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

## 分阶段运行配置

`project_on_ad10_Yolo.py` 仍是入口，`project_on_ad10_yolo/` 现在采用：

```text
六路回调 → 每路最新帧槽 → 单 YOLO Worker → 纯 YOLO 检测话题
                              │
                              ├─ 有匹配点云 → 独立 LiDAR Worker ─┐
                              └─ 无点云/未启用投影 ──────────────┤
                                                              ↓
                                                  独立绘图/发布 Worker
                                                              ↓
                                                  原图像与融合检测话题
```

三个阶段各自使用每路容量 1 的缓存和公平轮询。新帧覆盖该阶段同路尚未处理的旧帧；正在处理的任务不会被中断。每个任务带相机编号、回调入口分配的递增版本、原图像时间戳及已匹配的 LiDAR/BBox 引用。迟到的旧投影不能覆盖新输出。图像解码后复制为任务独占数组，原 ROS 图像不被绘图修改。单投影线程独占投影坐标缓存。

YOLO 推理线程不做 `result.plot()`、点云绘制或融合图像编码；下游使用 CPU 检测坐标绘制橙色 YOLO 框，外观与原 Ultralytics 绘图略有变化。Camera 1/6 的融合图像频率仍受投影速度限制，但纯 YOLO 检测及其他相机不必等待它。线程共享 CPU/GIL，实际收益需要 AD10 实测。本阶段未加入 Overlay 缓存、多进程或 TensorRT Batch。

已有图像与融合 `Detection2DArray` 话题名称、类型及默认原始 Header 保持不变。新增纯 YOLO `Detection2DArray`：

| 相机 | 默认纯 YOLO 话题 |
| --- | --- |
| Camera 1 | `/fusion/detections_2d/yolo` |
| Camera 2–6 | `/fusion/detections_2d_2/yolo` 至 `/fusion/detections_2d_6/yolo` |

可用 `~output_yolo_detections_topics` 配置恰好六项，不能与融合检测话题重名。纯 YOLO 检测不会附加 LiDAR 框，也不受融合输出丢帧影响；推理失败不发布伪造的纯 YOLO 空结果。成功推理但没有目标时正常发布空数组。投影异常仍将当前任务交给可视化，可能保留异常发生前已画出的部分点云；异常会单独计数。

在 AD10 启动好 ROS master、相机和 LiDAR 节点后，使用原有 ROS + YOLO venv 环境：

```bash
cd /home/nvidia/auto
rosparam load proj/config/yolo_runtime.yaml /project_bbox_node
./yolo11/venv/bin/python proj/project_on_ad10_Yolo.py _yolo_model_path:=/home/nvidia/auto/test/yolo11n.pt
```

若要用 `rosrun`，本目录已包含 `package.xml` 和 `CMakeLists.txt`，包名为 `ad10_yolo_fusion`。保持仓库位于 `/home/nvidia/auto` 时，在**启动节点的终端**执行：

```bash
source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH=/home/nvidia/auto:$ROS_PACKAGE_PATH
rospack find ad10_yolo_fusion
rosparam load /home/nvidia/auto/proj/config/yolo_runtime.yaml /project_bbox_node
rosrun ad10_yolo_fusion project_on_ad10_Yolo.py _yolo_model_path:=/home/nvidia/auto/test/yolo11n.pt
```

`roscore` 必须已运行，且 `rospack find` 应返回 `/home/nvidia/auto/proj`。入口脚本的 shebang 固定为 `/home/nvidia/auto/yolo11/venv/bin/python`；若虚拟环境不在这里，应修改入口 shebang 或使用上面的显式 Python 启动方式。`rosrun` 只负责定位并执行脚本，不会自动安装 `vision_msgs`、`ultralytics` 等依赖，也不会启动相机或 LiDAR 话题。若使用 catkin 工作空间，应将 `proj` 作为 `src` 下的包并构建、source 该工作空间，届时不必手动设置 `ROS_PACKAGE_PATH`。

工作目录不同则替换上述路径。样例配置只设置本阶段参数，不覆盖六路标定或输入话题。参数启动时读取，修改后需重启节点。若用 `__name` 改节点名称，`rosparam load` 的命名空间也应修改。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `~projection_cameras` | `[1, 6]` | 使用 **1 起始**相机编号，`[]` 关闭投影用于基线对比 |
| `~max_output_age_sec` | `0.5` | 收到图像至提交输出的任务期限，各阶段取帧及发布前检查；0 关闭期限检查 |
| `~monitor_topic` | `/fusion/performance` | `std_msgs/String`，内容为 JSON |
| `~monitor_interval_sec` | `1.0` | 独立监控线程两次采样之间的等待时间 |
| `~monitor_window_sec` | `5.0` | FPS、延迟分布、命中率的滚动窗口 |
| `~monitor_log_interval_sec` | `2.0` | 六路简要日志间隔；0 关闭日志 |
| `~gpu_load_path` | `""` | 自动探测，也可指定板端可读 GPU load 路径 |
| `~gpu_load_scale` | `0.1` | sysfs 原始数值到百分比的倍率，按板端接口调整 |

`~inference_queue_size` 已停用，启动时若仍配置会警告。旧 `~timing_log_interval_sec` 作为未设置 `~monitor_log_interval_sec` 时的回退值。期限检查不会中断 native/CUDA 调用，首次模型热身较慢时可能丢弃最初几帧。缓存容量是**每个阶段每路一帧**，另有在处理的任务，不是全流水线每路总共一帧。

## 实时监控

终端每 2 秒输出六路 `in/yolo/out FPS`、图像年龄 P50/P95、投影缓存命中率及丢帧/异常，另输出 CPU、进程 CPU、GPU、RAM 和总 YOLO FPS。完整指标默认每秒发布：

```bash
rostopic echo /fusion/performance/data
rostopic hz /fusion/detections_2d/yolo
rostopic hz /fusion/projected_image
```

若要直接查看各阶段耗时，可在另一个已 source ROS 环境的终端运行：

```bash
python3 proj/view_performance.py
python3 proj/view_performance.py --camera 1
```

脚本每收到一条监控消息便打印六路 FPS，以及 `yolo_predict_ms`、`yolo_extract_ms`、`cloud_draw_ms`、三个阶段的 `queue_wait_ms` 和 `image_age_ms` 的 P95。`--camera 1` 只看 Camera 1。显示 `-` 表示该路在滚动窗口内没有对应样本，例如 Camera 2–5 不执行点云绘制。

JSON `schema_version=1`，主要字段如下：

| 字段 | 语义 |
| --- | --- |
| `cameras[N].fps.input` | 图像回调到达 FPS，含随后因时间戳无效被丢弃的消息 |
| `cameras[N].fps.yolo` | 成功完成预测与 CPU 检测数据提取的 FPS，包含无目标帧；不等同于融合输出 FPS |
| `cameras[N].fps.output` | 原图像与融合检测均成功提交给 ROS Publisher 的 FPS |
| `cameras[N].fps.projection` | 完成点云及可用 BBox 投影的 FPS |
| `cameras[N].observations.image_age_ms` | 原图像 `header.stamp` 至融合发布提交结束的年龄；含 `count/mean/p50/p95/latest` |
| `cameras[N].observations.yolo_image_age_ms` | 原图像时间戳至 YOLO 数据提取完成的年龄 |
| `cameras[N].observations.end_to_end_ms` | 本节点图像回调至融合发布提交结束的单调时钟耗时 |
| `cameras[N].projection_cache` | 本窗口实际投影坐标缓存 `hits/misses/hit_rate`；未发生查询时 `hit_rate=null` |
| `cameras[N].counters.window/total` | 窗口/启动累计计数，含成功事件、覆盖丢帧、迟到/过期丢弃和异常 |
| `pending.input/projection/visualization` | 每个阶段六路是否存在待处理任务，不含正在执行的任务 |
| `resources.cpu_percent/cpu_per_core_percent` | 全机总 CPU 及逐核使用率，范围 0–100% |
| `resources.process_cpu_percent/threads` | 本进程 CPU（多核可超过 100%）、线程 TID/名称/CPU |
| `resources.ram/rss_bytes` | 全机内存状态、本进程 RSS |
| `resources.gpu` | GPU 使用率、数据源、设备列表或不可用原因；无法读取时 `percent=null` |

资源负载属于整个节点/设备，不是每路相机独占占用。CPU 首次采样没有差分基线时为 `null`。GPU 优先探测 Jetson 的 `ga10b/gv11b/gp10b/gpu` 设备目录（包括 `/sys/devices/platform/17000000.ga10b/load`），以及对应的 devfreq 设备入口；默认数值除以 10 转换为百分比，见 [NVIDIA Jetson Linux 时钟文档](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/SD/Clocks.html)。找不到时再尝试带 0.8 秒超时的 `nvidia-smi`；不要求安装 psutil、jtop 或 root 权限。设备不可读时显示 `n/a`，不会伪装成 0%。`nvidia-smi` 若有多卡，顶层百分比是设备最大值，逐卡值见 `devices`，并非本进程 GPU 独占使用率。

分阶段耗时位于各路 `observations`：`input_queue_wait_ms`（另保留 `queue_wait_ms` 同义项）、`projection_queue_wait_ms`、`visualization_queue_wait_ms`、`input_callback_ms`、`bbox_match_ms`、`image_decode_ms`、`yolo_predict_ms`、`yolo_extract_ms`、`yolo_publish_ms`、`cloud_convert_ms`、`project_calculation_ms`、`cloud_draw_ms`、`bbox_projection_ms`、`lidar_projection_ms`、`plot_ms`、`cv2_to_imgmsg_ms`、`publish_ms`。缓存命中帧的转换/投影计算耗时记 0，绘制仍每帧执行；`total_points/projected_points` 也会记录。`yolo_predict_ms` 是预测接口墙钟耗时，不是 CUDA kernel 计时，CPU 数据提取/同步单独计入 `yolo_extract_ms`。

常见计数器：`input_overwritten/projection_overwritten/visualization_overwritten` 表示被新版本覆盖；`*_stale_dropped` 表示迟到的旧版本；`*_expired` 表示超过任务期限；`lidar_unavailable` 表示回调匹配不到有效历史点云，**不算缓存未命中**；`*_errors` 为各类异常。`yolo_published` 计数表示纯 YOLO 检测实际提交成功。FPS 和分位数只覆盖窗口内事件，每路每项最多保留 4096 个样本；极高输入率或过长窗口超出此上限时会截断。

图像年龄要求相机时间与 AD10 ROS 系统时钟一致；发布耗时只到 ROS 异步发布提交完成，不含网络传输或显示端耗时。输出采用系统时间戳时仍以**原采集时间戳**计算年龄；出现负年龄会计入 `negative_*_samples`，不纳入延迟分布。超过 `~max_clock_skew_sec` 的输入仍按原规则丢弃。无当前窗口样本时延迟为 `null`，避免把停流后的旧数值当成当前状态。

## 验证及板端对比

离线回归使用真实 NumPy/OpenCV，ROS、模型和消息传输使用替身，无需启动 ROS master：

```bash
python3 -m unittest discover -s proj/tests -v
python3 -m py_compile proj/project_on_ad10_Yolo.py proj/project_on_ad10_yolo/*.py
```

在 AD10 用同一模型、分辨率、输入场景和功率模式，先启动 `projection_cameras: []`，再启动 `[1, 6]`。待模型热身后分别观察六路 YOLO FPS、输出 FPS、P95 图像年龄、各阶段等待与绘图耗时、丢帧和资源负载。此改动的离线测试不能证明板端达到 180 FPS；硬件推理吞吐、GIL 竞争、ROS 实际传输及 GPU sysfs 路径需板端验证。
