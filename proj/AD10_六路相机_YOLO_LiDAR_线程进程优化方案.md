# AD10 六路相机 YOLO + LiDAR：线程、进程与实时性优化方案

> 适用项目：`Dustess/auto_drive`；目标平台：Miivii APEX AD10 / Jetson Orin、Ubuntu 20.04、ROS Noetic。  
> 目标：六路相机各 30 Hz，YOLO 检测与 Camera 1、Camera 6 的 LiDAR 投影同时运行；优先降低图像积压和端到端延迟，再追求总吞吐量接近 180 FPS。  
> 文档性质：第 1–14 节保留改造前分析及目标方案；本地第一阶段实现进度见第 15 节。尚无 AD10 板端性能实测结论。

## 1. 参考代码

仓库：<https://github.com/Dustess/auto_drive>

| 文件 | 作用 |
| --- | --- |
| [`proj/project_on_ad10_Yolo.py`](https://github.com/Dustess/auto_drive/blob/main/proj/project_on_ad10_Yolo.py) | 启动入口；构造 `ProjectBboxNode`，执行 `rospy.spin()` |
| [`proj/project_on_ad10_yolo/fusion_node.py`](https://github.com/Dustess/auto_drive/blob/main/proj/project_on_ad10_yolo/fusion_node.py) | 六路相机订阅、LiDAR/BBox 缓存、单全局 FIFO 队列、单推理 Worker |
| [`proj/project_on_ad10_yolo/yolo_fusion.py`](https://github.com/Dustess/auto_drive/blob/main/proj/project_on_ad10_yolo/yolo_fusion.py) | `_run_yolo()`、`_process_frame()`、检测消息构造、投影调用、图像绘制及发布 |
| [`proj/project_on_ad10_yolo/projection.py`](https://github.com/Dustess/auto_drive/blob/main/proj/project_on_ad10_yolo/projection.py) | PointCloud2 读取、LiDAR→相机投影、投影坐标缓存、逐点绘图、Marker BBox 投影 |

相关成熟设计可参考：

- [NVIDIA DeepStream `nvstreammux`](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvstreammux.html)：多路视频缓冲、批量组装和超时发送。
- [Ultralytics YOLO Thread-Safe Inference](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/)：模型实例的线程安全限制。
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)：Execution Context、CUDA Stream、动态 Batch 与异步执行（按本机 TRT 8.5 API 核实）。

> 上述参考用于借鉴架构；它们的性能数据不能直接当成 AD10 的测量结果。CSDN 文章可用于寻找工程实践思路，但必须在本机验证。

## 2. 当前实现与实验事实

### 2.1 现有流水线

```text
Camera 1..6 ROS Subscribers
          │
          ├─ Camera 1/6：在图像回调中匹配历史 LiDAR 和 BBox
          │
          ▼
  processing_queue = Queue(maxsize=12)
          │
          ▼
  _inference_worker（仅 1 个显式创建的工作线程）
          │
          ├─ _validate_timestamps()
          ├─ CvBridge: imgmsg_to_cv2()
          ├─ YOLO.predict() + result.plot()
          ├─ Camera 1/6: _draw_cloud() + _draw_bboxes()
          ├─ Detection2DArray 构造
          └─ CvBridge: cv2_to_imgmsg() + ROS publish()
```

六路相机属于同一个 Python ROS 进程。各相机的 `rospy` TCPROS 接收通常在各自连接线程中调用回调；**这不等同于六个独立 YOLO 推理线程**。

### 2.2 已验证的现象

原配置：

```python
self.projection_camera_indices = {0, self.camera_count - 1}
```

实验中改为：

```python
self.projection_camera_indices = {99, 100}
```

六路相机均不再进入投影分支，但 YOLO 仍运行；观察到图像延迟、卡顿明显改善。

**可以得出的结论**：投影相关处理及其资源竞争/串行等待，是当前实时性问题的重要贡献因素。  
**不能直接得出的结论**：瓶颈完全来自坐标投影，或者 GPU YOLO 推理已经足以实现 180 FPS。需要分阶段计时进一步确认。

## 3. 代码级瓶颈

| 位置 | 现状 | 风险 / 优化方向 |
| --- | --- | --- |
| `fusion_node.py` 的 `processing_queue` | 六路共用 FIFO，最多 12 个任务 | 旧帧积压、队头阻塞、各相机处理机会不均 |
| `_inference_worker()` | 单 Worker 串行 `_process_frame()` | Camera 1/6 投影会延迟其他相机的 YOLO |
| `yolo_fusion.py` 的 `_run_yolo()` | `predict()` 后立即 `result.plot()` | GPU 推理和可视化绑在同一个关键路径 |
| `_process_frame()` | YOLO → 投影 → 图像编码 → 同步发布 | 绘图、消息编码和 ROS 发布阻塞下一次 YOLO |
| `projection.py` 的 `_read_lidar_points()` | `sensor_msgs.point_cloud2.read_points` 逐点生成，再转 NumPy | 首次处理新点云时有 Python 遍历与转换开销 |
| `_draw_cloud()` | 已缓存 `(pixels, colors, order)`，但**每次**仍循环 `cv2.circle()` | 同一 LiDAR 帧的绘制开销没有复用；可能是 CPU 热点 |
| `_draw_bboxes()` | 每个 Marker 投影角点并绘制 | 需要独立计时，但不应先假定是主要瓶颈 |
| `projected_cloud_cache` | 已按相机、LiDAR stamp、标定、图像尺寸等缓存像素投影 | 优化时应保留现有缓存；不要误判为“没有缓存” |

**优先定位的热点**：`_draw_cloud()` 中 `for index in order: cv2.circle(...)`，同时分别测量点云读取、首次投影、缓存命中、BBox 绘制、`result.plot()`、图像编码和发布。

## 4. 核心设计原则

1. **优先降低延迟而非保留每帧**：在线感知应允许跳过过期帧，避免越算越旧。
2. **六路相机公平调度**：采用每路容量 1 的 Latest Frame Buffer，而非共用长 FIFO。
3. **YOLO、LiDAR 投影、可视化解耦**：耗时的点云绘制不能阻塞下一次推理。
4. **优先共享一个推理后端**：不要直接创建六个 Ultralytics 模型/六个 CUDA 进程。
5. **只在测量支持时增加并发**：TensorRT Context 数量、Batch 大小、投影 Worker 数量都需要 A/B 实验。
6. **保持时间一致性**：图像、LiDAR、BBox 和检测结果都保存原始时间戳，按 `camera_id + image stamp (+ sequence/version)` 关联，禁止无条件拼接“各自最新”的结果。
7. **避免无谓的数据拷贝**：相机回调尽量只存 ROS Image 引用和元数据，待确定会处理时再解码。

## 5. 推荐目标架构

```text
                  六路 Camera Subscriber（rospy）
                             │
                 Latest Frame Buffer × 6
                    每路容量 1，新覆旧
                             │
                   公平轮询 / 事件唤醒
                             │
                       YOLO Scheduler
                             │
              ┌──────────────┴──────────────┐
              │                             │
         GPU 推理流水线                 LiDAR 投影流水线
              │                             │
      预处理 → TensorRT                 LiDAR 缓存
              │                             │
       结构化 YOLO 结果                Camera 1/6
              │                        时间戳匹配
              │                             │
              │                        投影坐标/Overlay
              │                             │
              └───────────┬─────────────────┘
                          │
             结果关联 / 可视化 / ROS 发布
              （独立工作线程或下游节点）
```

### 5.1 首选落地形态：单进程、多个独立流水线

建议先保持一个 Python ROS 进程：

| 模块 | 建议资源 | 说明 |
| --- | --- | --- |
| 相机接收 | 原有 6 个 Subscriber | 无需另外手动建立 6 个采集线程 |
| 帧缓存 | 每路 1 个槽 + 版本号 | 消除全局 FIFO 积压 |
| YOLO 调度 | 1 个线程 | 公平轮询、有新帧才唤醒 |
| YOLO 推理 | 初始 1 个 Worker | 先保留 Ultralytics 作基线，再替换 TRT |
| LiDAR 回调 | 保留现有轻量回调 | 只存数据和时间戳 |
| LiDAR 投影 | 初始 1 个独立 Worker | 实测后可试 Camera 1/6 各 1 个 |
| 可视化/发布 | 初始 1 个 Worker | 丢弃过期待显示任务；避免反压到 YOLO |

> Python GIL 可能影响纯 Python CPU 运算的并行度；C++/NumPy/OpenCV/推理库是否释放 GIL 取决于实现。线程解耦可消除**逻辑串行阻塞**，不保证消除所有 CPU 竞争。

### 5.2 备选形态：两个独立进程

当投影 CPU 开销或 GIL 竞争明显时，可迁移为：

```text
Process A：camera_yolo_node（Python 调度 + C++ TensorRT）
  ├─ 六路图像订阅与最新帧缓存
  ├─ 统一 YOLO 调度
  ├─ Detection2DArray 发布
  └─ 可选独立可视化

Process B：lidar_projection_node（C++ ROS / OpenCV / PCL）
  ├─ LiDAR/MarkerArray 订阅
  ├─ Camera 1/6 订阅或带时间戳的投影任务输入
  ├─ 点云转换、投影、绘制
  └─ 投影结果发布
```

跨进程传输 ROS1 原始图像会增加序列化、内存复制和消息开销。优先让投影进程直接订阅原始相机和点云话题，避免从推理节点再转发完整图像。第三个 `visualization_node` 仅在发布/绘图负载值得拆分且传输成本可接受时引入。

**不建议默认使用六个 YOLO 进程**：虽然多进程能利用 CPU 多核心并隔离错误，但六套 GPU 模型实例、CUDA 上下文和显存可能抵消收益；多进程不等于 GPU 计算能力增加。

## 6. 具体改动 A：六路 Latest Frame Buffer

### 6.1 替换全局 FIFO

现有：

```python
self.processing_queue = Queue(maxsize=12)
```

建议使用六路逻辑缓存，每路只保留**尚未取走的最新图像**。用锁与 `Condition` 做线程间交接，尽量避免忙轮询。

```python
import threading

self.latest_frames = [None] * self.camera_count
self.frame_versions = [0] * self.camera_count
self.frame_cond = threading.Condition()

def _store_frame(self, camera_index, frame):
    with self.frame_cond:
        self.latest_frames[camera_index] = frame
        self.frame_versions[camera_index] += 1
        self.frame_cond.notify()
```

示意调度伪代码：

```python
def _next_frame(self):
    with self.frame_cond:
        while not self.stop_worker.is_set():
            for offset in range(self.camera_count):
                idx = (self.next_camera + offset) % self.camera_count
                if self.latest_frames[idx] is not None:
                    frame = self.latest_frames[idx]
                    self.latest_frames[idx] = None  # 原子取走
                    self.next_camera = (idx + 1) % self.camera_count
                    return frame
            self.frame_cond.wait(timeout=0.1)
    return None
```

上述代码是结构示例，不是可直接替换的完整补丁。正式实现需处理：初始化 `next_camera`、关闭唤醒、工作线程异常、每路丢帧计数、帧版本号和 ROS 退出条件。不要只依赖 `header.seq`，因为一些驱动的该字段可能不连续或不唯一。

### 6.2 缓存任务的元数据

```text
camera_id
image_msg 或生命周期安全的图像引用
original_image_stamp
frame_version / sequence
received_monotonic
（需要投影时）cloud_stamp、cloud/bbox 引用或可重查缓存的索引
```

图像回调尽可能只做入槽、时间戳检查与必要的轻量匹配。是否把 LiDAR 查找从回调移到投影调度器，需要确保不会因缓存更新而改变本帧本应匹配的点云；稳妥做法是保留已选点云引用及时间戳。

## 7. 具体改动 B：拆解 `_process_frame()`

建议把现有单一 `_process_frame()` 拆成三个阶段及独立任务接口：

```python
def process_yolo(frame):
    # 图像解码 → 预处理 → 模型推理 → 检测数据提取
    # 返回 camera_id、原始 image_stamp、模型检测列表
    ...

def process_projection(frame):
    # 按原始时间戳选择 LiDAR/BBox → 投影/缓存 → 像素坐标或 Overlay
    # 返回 image_stamp、cloud_stamp、projection_result
    ...

def process_visualization(frame, yolo_result, projection_result=None):
    # 独立执行图像绘制、编码和发布
    ...
```

关键约束：

- YOLO 的结构化结果可以先发布，不需要等 LiDAR Overlay；如原有用户订阅的是“YOLO+LiDAR 合并的 Detection2DArray”，则额外保留融合发布接口。
- 若可视化需要同一张图像的 YOLO 和 LiDAR 数据，必须使用 `camera_id + 原始图像时间戳/版本` 匹配；超时后应定义降级策略（如只显示 YOLO，不无限等待投影）。
- 不要在持有帧缓存锁或 LiDAR 缓存锁时执行 `predict()`、`projectPoints()`、`cv2.circle()` 或 ROS `publish()`。
- 原始图像要保持只读或进行明确所有权管理，避免两个线程同时原地绘制同一 `cv::Mat` / NumPy 数组。

## 8. 具体改动 C：LiDAR 投影与绘制优化

### 8.1 保留现有投影坐标缓存

`projection.py` 已经有 `projected_cloud_cache`，按相机、时间戳、分辨率、相机模型参数等生成 Key，缓存 `pixels/colors/order`。**不能把它当作完全缺失的功能重新设计。**

### 8.2 单独测量绘图循环

当前每帧会执行：

```python
for index in order:
    u, v = pixels[index]
    cv2.circle(image, (int(u), int(v)), self.point_radius_px,
               tuple(int(value) for value in colors[index]),
               thickness=-1, lineType=cv2.LINE_AA)
```

首先统计：

```text
lidar_read_ms
project_calculation_ms
projection_cache_hit
cloud_draw_ms
bbox_projection_ms
projected_points
```

将首次投影（缓存未命中）与重复绘图（缓存命中）分开统计。

### 8.3 测试 Overlay 缓存

同一 LiDAR 帧在固定相机标定、图像尺寸条件下，投影结果可以提前绘制为可复用的图层：

```text
LiDAR 10 Hz → 像素投影 → 生成透明 Overlay + mask
                                    │
Camera 1/6 30 Hz ──────────────────┴─→ 按 stamp 匹配并叠加
```

这样缓存命中时不需要重复逐点调用 `cv2.circle()`，而是做一次图层合成。需注意：

- Overlay 只缓存 LiDAR 层，不包含 YOLO 图像，以免把旧图像叠到新图像。
- 透明背景必须用 mask/alpha，不能简单把黑底图做无条件相加。
- 缓存需包含相机 ID、LiDAR stamp、分辨率、标定版本及绘图配置（点半径、颜色策略等），变更时失效。
- 不可忽视移动目标的时间错位；缓存只减少计算，不取代时间同步/运动补偿。
- 按实际分辨率限制 Overlay 缓存大小，避免大幅增加共享内存占用。

### 8.4 可选进一步优化

- 若只需检测融合，评估 **仅绘制 LiDAR 聚类 BBox**，不必在实时主链路绘制全部点云。
- 允许可视化点云降采样，但不应无意降低真正用于检测/融合的点云质量。
- 点云新帧的首次 `read_points` 转换可在 LiDAR Worker 中**一次完成，多相机共享只读数组**。
- 首次投影耗时高时，再迁移 C++/PCL/OpenCV，减少 Python 逐点处理。
- 对比 `LINE_AA`、`LINE_8`、不同点半径与投影点上限对视觉质量和耗时的影响。

## 9. 具体改动 D：YOLO 与可视化分离

当前 `_run_yolo()` 在取出检测列表后立即 `result.plot()`。建议将模型原始检测和结果绘制解耦：

1. 推理 Worker：`YOLO.predict()`、boxes 提取、生成结构化检测结果。
2. 下游可视化 Worker：在图像上绘制 YOLO 框和可用的 LiDAR Overlay。
3. 发布结构化 YOLO Detection2DArray 不必受全图绘制影响。
4. 可视化缓存应采用每路容量 1 或有限容量、主动丢旧帧；不能用无界队列反向拖慢推理。

优先分开记录 `predict_ms`、`plot_ms`、`cv2_to_imgmsg_ms` 和 `publish_ms`。注意：原先 `yolo_inference_ms` 包含 `result.plot()`，不是纯 GPU 推理耗时。

## 10. TensorRT 优化：Batch 与 Context

六路各 30 Hz，总输入需求为：

\[
6 \times 30 = 180\ \mathrm{frames/s}
\]

对应约 `5.56 ms/帧` 的**平均吞吐量预算**，并不是要求每张图像端到端延迟必须少于 5.56 ms。

先用**投影关闭的情况**建立 YOLO 基线，再测试以下配置：

| 实验 | Engine | Context | Stream | Batch | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 1 | 1 | 1 | 1 | 串行 TRT 基线 |
| B | 1 | 1 | 1 | 2 | 微批吞吐量 |
| C | 1 | 1 | 1 | 3 | 微批吞吐量 |
| D | 1 | 1 | 1 | 6 | 满批吞吐量与等待成本 |
| E | 1 | 2 | 2 | 1 | 多 Context 并发 |
| F | 1 | 2 | 2 | 2 | 微批与并发组合 |

实施注意：

- TensorRT Engine 可共享权重；每个并发 Worker 应有自己独立的 Execution Context、Stream 与执行缓冲，不能无保护地并发复用同一个 Context。
- 目前若 Engine 输入固定为 `(1,3,640,640)`，**不能直接改成 Batch=2/3/6**；需重新构建兼容目标 Batch 或动态 Batch Profile 的 Engine。
- 微批组装必须设置超时，防止为了凑齐六帧而增加不合理的图像等待。可以从 `2–5 ms` 作为**测试起点**，非确定的最优参数。
- Context 更多也可能增加显存、CPU 调度和 GPU 争用，不能预设双 Context 更快。
- Python → C++ 本身不保证 GPU 计算速度提高；提升通常来自 TensorRT、FP16、预处理优化、Batch、异步数据传输和任务调度。
- Jetson 上 CPU/GPU 共享物理内存，仍须评估重复缓冲和内存带宽。

## 11. 线程数量与进程划分的推荐决策

| 问题 | 首选方案 | 何时改变 |
| --- | --- | --- |
| 为六个相机各创建 YOLO 线程？ | **否** | 只有独立 Context/模型实例 + 测量证明有收益时才增加推理并发 |
| 为六个相机各创建 YOLO 进程？ | **否** | 特殊隔离要求或跨 GPU 分配时再考虑 |
| 六路缓存是否共用 FIFO？ | **否** | 用每路 1 槽 Latest Frame Buffer |
| YOLO Worker 数量 | **1** | 完成 TRT 基线后测试 2 Context / 2 Streams |
| 投影 Worker 数量 | **1** | CPU 负载与两路延迟测量后再测试 2 |
| 投影是否拆到 C++ 进程？ | **第二阶段** | 线程解耦后仍有明显 Python/GIL/CPU 瓶颈 |
| 可视化是否单独进程？ | **暂不** | 只有绘制/发布很重且 ROS IPC 成本可控时再拆 |

## 12. 性能监测与 A/B 实验

### 12.1 必须记录的指标

| 指标 | 含义 |
| --- | --- |
| 每路输入 FPS | 相机到达频率 |
| 每路实际 YOLO FPS | 每台相机真正完成检测的频率 |
| 总 YOLO FPS | 六路合计吞吐量 |
| `queue_wait_ms` / `buffer_age_ms` | 调度等待时间 |
| `yolo_predict_ms` | 模型预测接口耗时（进一步分解预/推理/后处理） |
| `plot_ms` / `cloud_draw_ms` | 可视化 CPU 时间 |
| `projection_cache_hit` | 投影缓存命中状态 |
| `cloud_convert_ms` / `project_calculation_ms` | 点云转换与首次投影 |
| `publish_ms` | 编码和 ROS 发布开销 |
| `drop_count[camera]` | 每路覆盖丢帧数 |
| `image_age_ms` P50/P95 | 原始采集时间戳至输出完成时间（前提是同步时钟） |
| CPU 各核/线程负载、GPU 使用、RAM | 查找资源竞争 |

当前代码的 `end_to_end_ms` 是从收到 ROS 图像回调到完成处理；**不是**相机曝光时刻到结果的全链路延迟。对真正的图像年龄应使用 `output_time - original_image_stamp`，并确保系统时间基准一致。

AD10 可用监测命令：

```bash
htop -H
pidstat -t -p <PID> 1
sudo tegrastats
rostopic hz /fusion/projected_image
```

`rostopic hz` 测的是所订阅话题的输出频率，不能替代各相机的推理 FPS 和原始采集延迟。CUDA 推理可用 Nsight Systems 做阶段与 kernel 时序分析（注意输出文件路径、采集时段和权限）。

### 12.2 推荐实验矩阵

| 阶段 | YOLO | Camera 1/6 投影 | 核心变化 | 预期验证 |
| --- | --- | --- | --- | --- |
| 0 | 开 | 关 `{99,100}` | 保持原始代码 | 建立基线 |
| 1 | 开 | 关 | 六路 Latest Frame Buffer | 排队时间与 P95 图像年龄是否下降 |
| 2 | 开 | 开 | 投影独立 Worker | Camera 2–5 是否不再被投影阻塞 |
| 3 | 开 | 开 | Overlay 缓存 / 绘图优化 | 投影 CPU 耗时能否明显下降 |
| 4 | 开 | 开 | YOLO 与可视化分离 | YOLO 有效吞吐量是否提高 |
| 5 | TensorRT | 开 | Batch / Context 实验 | 在实时性约束下提高总 FPS |
| 6 | TensorRT | 开 | 按需拆分 C++ 投影进程 | 比较隔离收益与 IPC 成本 |

建议每次固定相机话题、分辨率、模型、CPU/GPU 功率模式、绘图参数和场景，持续运行后统计 P50/P95、各路 FPS 与丢帧。最好使用可重复的录制数据或固定输入进行 A/B 测试。每次只改一个主要变量。

## 13. 最小风险实施次序

- [ ] **P0**：保持 `{99,100}` 作为无投影基线，记录每路 FPS、图像年龄、Worker 队列等待。
- [ ] **P0**：在 `_draw_cloud()` 新增 `projection_cache_hit`、`cloud_draw_ms`；单独测 `result.plot()` 和 `cv2_to_imgmsg()`。
- [ ] **P1**：将全局 FIFO 换成六路容量 1 的最新帧缓存，采用公平轮询和事件唤醒。
- [ ] **P1**：将 `_process_frame()` 拆分为 YOLO、投影和可视化三个可独立执行的阶段。
- [ ] **P1**：让投影在独立 Worker 执行，保留现有 LiDAR 时间匹配逻辑和标定参数。
- [ ] **P2**：测试 Overlay 缓存、点数上限、简化绘图和仅投影 LiDAR BBox。
- [ ] **P2**：只发布结构化检测结果的性能与带可视化图像的性能分开测试。
- [ ] **P3**：把 YOLO 推理后端替换为 TensorRT，先单 Context，再 Batch 和双 Context。
- [ ] **P3**：若 CPU 投影仍显著影响推理，再迁移独立 C++ ROS 进程。

## 14. 结论

**优先处理的不是“六个相机要不要六个 YOLO 线程”，而是“一个 `_process_frame()` 不应串行承担六路 YOLO、两路点云投影、所有绘图及同步发布”。**

本项目最优先的两项工程变更：

1. **`fusion_node.py`：全局 FIFO → 六路最新帧缓存 + 公平调度；YOLO 和投影不再互相阻塞。**
2. **`projection.py` / `yolo_fusion.py`：分别测量和优化逐点 `cv2.circle()`、`result.plot()`、图像编码与发布；复用现有投影坐标缓存并按需加入 Overlay 缓存。**

推荐最终演进路径：

```text
单进程串行现状
    ↓
单进程多流水线（低风险、优先验证）
    ↓
独立 C++ 投影进程（仅在确有 CPU/GIL 瓶颈时）
    ↓
TensorRT Batch/多 Context（实测吞吐量与延迟后决定）
```

**不要把增加线程/进程数量当成优化目标。** 目标应是六路公平、尽量处理最新帧、保持时间戳关联，以及在可接受延迟和资源消耗下逐步接近 180 FPS。

## 15. 第一阶段本地实现（2026-11-26）

已修改 `project_on_ad10_yolo/`，入口仍为 `project_on_ad10_Yolo.py`：

- 三个独立工作阶段：单模型 YOLO → 可选 LiDAR 投影 → 绘图/融合发布。监控另有独立线程。
- 各阶段每路容量 1，公平轮询、覆盖丢旧帧、原始时间戳及递增版本随任务传递；拒绝迟到旧投影。图像回调仍选择最近过去点云与匹配 BBox。
- 原融合图像/检测话题保留，新增各路 `.../yolo` 纯检测话题，不等待投影。CPU 框绘制移到发布阶段。
- 保留并完善投影坐标 LRU 缓存，增加点云转换、计算、逐点绘制、BBox、解码、推理、编码、发布和等待耗时。
- ROS JSON 话题 `/fusion/performance` 与定时日志：每路输入/推理/输出 FPS、图像年龄 P50/P95、缓存命中率、分阶段丢帧、全机/进程/逐核/线程 CPU、GPU 和内存。
- `~projection_cameras: []` 可关闭投影，无需再修改代码为 `{99,100}`。新增配置和操作说明见 [`config/README.md`](config/README.md) 及 [`config/yolo_runtime.yaml`](config/yolo_runtime.yaml)。

这是消除跨帧串行阻塞的流水线实现，同一帧的投影仍接在 YOLO 后完成。融合输出受投影速度及任务期限限制；不把不同帧的各自最新结果拼在一起。本阶段未实现 Overlay 图层缓存、C++ 独立进程、TensorRT 多 Context/Batch。

已加入离线回归覆盖慢投影隔离、并发回调顺序、跨阶段版本拒绝、原图不被修改、点云时间匹配、缓存失效、异常降级、监控窗口及关闭唤醒。测试使用 ROS/模型替身；实际 AD10 六路吞吐、延迟下降幅度及 GPU 采样仍待板端验证。
