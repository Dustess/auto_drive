"""离线回归：真实 NumPy/OpenCV，ROS 消息/传输及模型使用轻量替身。"""

import json
import sys
import tempfile
import threading
import time
import types
import unittest
from collections import OrderedDict, deque
from pathlib import Path
from unittest.mock import patch

import numpy as np


class Stamp:
    def __init__(self, value):
        self.secs = int(value)
        self.nsecs = int(round((value - self.secs) * 1e9))

    def to_sec(self):
        return self.secs + self.nsecs / 1e9


class DetectionArray:
    def __init__(self):
        self.header = None
        self.detections = []


class Detection:
    def __init__(self):
        self.id = ""
        self.results = []
        self.bbox = types.SimpleNamespace(center=types.SimpleNamespace(x=0, y=0, theta=0))


class Publisher:
    def __init__(self, *args, **kwargs):
        self.messages = []
        self.published = threading.Event()

    def publish(self, message):
        self.messages.append(message)
        self.published.set()


class Bridge:
    def imgmsg_to_cv2(self, message, desired_encoding):
        return message.pixels

    def cv2_to_imgmsg(self, image, encoding):
        return types.SimpleNamespace(pixels=image.copy(), encoding=encoding)


def install_ros_stubs():
    def module(name, **attributes):
        value = types.ModuleType(name)
        value.__dict__.update(attributes)
        sys.modules[name] = value
        return value

    rospy = module("rospy", Time=types.SimpleNamespace(now=lambda: Stamp(time.time())),
                   is_shutdown=lambda: False, ROSInitException=RuntimeError,
                   init_node=lambda *a: None, get_param=lambda name, default=None: default,
                   has_param=lambda name: False, resolve_name=lambda name: name,
                   on_shutdown=lambda callback: None, Publisher=Publisher,
                   Subscriber=lambda *a, **kw: types.SimpleNamespace(unregister=lambda: None))
    for name in ("loginfo", "logwarn", "logerr_throttle", "logwarn_throttle", "logdebug_throttle"):
        setattr(rospy, name, lambda *a, **kw: None)
    module("cv_bridge", CvBridge=Bridge, CvBridgeError=ValueError)
    points = module("sensor_msgs.point_cloud2", read_points=lambda msg, **kw: iter(msg.points))
    module("sensor_msgs", point_cloud2=points)
    module("sensor_msgs.msg", CameraInfo=object, Image=object, PointCloud2=object)
    module("std_msgs")
    module("std_msgs.msg", String=lambda **kw: types.SimpleNamespace(**kw))
    module("vision_msgs")
    module("vision_msgs.msg", Detection2D=Detection, Detection2DArray=DetectionArray,
           ObjectHypothesisWithPose=lambda: types.SimpleNamespace(class_id="", score=0))
    module("visualization_msgs")
    module("visualization_msgs.msg", MarkerArray=object,
           Marker=types.SimpleNamespace(ADD=0, MODIFY=0, CUBE=1))
    transforms = module("tf.transformations", quaternion_matrix=lambda q: np.eye(4))
    module("tf", transformations=transforms)
    return rospy


rospy = install_ros_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_on_ad10_yolo.fusion_node import ProjectBboxNode
from project_on_ad10_yolo.monitoring import PerformanceMonitor, ResourceMonitor
from project_on_ad10_yolo.scheduling import LatestFrameBuffer


def make_node():
    node = ProjectBboxNode.__new__(ProjectBboxNode)
    node.camera_count = 6
    node.camera_names = ["camera_%d" % (index + 1) for index in range(6)]
    node.metrics = PerformanceMonitor(node.camera_names)
    node.bridge = Bridge()
    node.lock = threading.Lock()
    node.frame_version_lock = threading.Lock()
    node.frame_versions = [0] * 6
    node.last_output_versions = [-1] * 6
    node.stop_worker = threading.Event()
    node.input_buffer = LatestFrameBuffer(6)
    node.projection_buffer = LatestFrameBuffer(6)
    node.visualization_buffer = LatestFrameBuffer(6)
    node.image_pubs = [Publisher() for _ in range(6)]
    node.detections_pubs = [Publisher() for _ in range(6)]
    node.yolo_detections_pubs = [Publisher() for _ in range(6)]
    node.output_stamp_with_system_time = False
    node.max_output_age_sec = 2.0
    node.max_clock_skew_sec = 10.0
    node.max_lidar_age_sec = 0.3
    node.bbox_slop_sec = 0.1
    node.mode = "cloud_bbox"
    node.projection_camera_indices = {0, 5}
    node.expected_lidar_frame = ""
    node._run_yolo = lambda image: ([(1, "object", 0.9, np.array([1, 2, 14, 15]))],
                                   {"yolo_predict_ms": 0.1})
    node.camera_models = [{"camera_matrix": np.array([[20., 0, 16], [0, 20, 16], [0, 0, 1]]),
                           "distortion": np.zeros(5), "camera_info_received": False,
                           "calibration_image_size": (32, 32),
                           "rotation_camera_from_lidar": np.eye(3),
                           "translation_camera_from_lidar": np.zeros(3)} for _ in range(6)]
    node.projected_cloud_cache = OrderedDict()
    node.projected_cloud_cache_size = 2
    node.undo_lidar_preprocessing = False
    node.cloud_frame_translation = np.zeros(3)
    node.min_depth_m = 0.15
    node.max_depth_m = 20.
    node.max_points = 50000
    node.point_radius_px = 2
    node.lidar_cache = deque(maxlen=12)
    node.bbox_cache = deque(maxlen=24)
    node.last_cloud_stamp_by_camera = [None] * 6
    node.worker_threads = []
    return node


def make_frame(index=0, cloud=True, version=1, stamp=None):
    stamp = time.time() if stamp is None else stamp
    header = types.SimpleNamespace(stamp=Stamp(stamp), frame_id="optical_%d" % index, seq=0)
    lidar = types.SimpleNamespace(header=types.SimpleNamespace(stamp=Stamp(stamp - .05), frame_id="lidar"),
                                  points=[(0., 0., 2.), (.2, .2, 3.)], width=2, height=1)
    return {"camera_index": index, "image": types.SimpleNamespace(header=header, pixels=np.zeros((32, 32, 3), np.uint8)),
            "image_stamp": stamp, "cloud": lidar if cloud else None,
            "cloud_stamp": stamp - .05 if cloud else None, "bbox_stamp": None,
            "frame_version": version, "received_monotonic": time.perf_counter()}


class BufferTests(unittest.TestCase):
    def test_overwrite_fairness_and_stale_result_rejection(self):
        buffer = LatestFrameBuffer(6)
        first, second = object(), object()
        self.assertIsNone(buffer.put(0, first, 1))
        self.assertIs(buffer.put(0, second, 2), first)
        for index in range(1, 6):
            buffer.put(index, index, 1)
        self.assertEqual(sum(buffer.snapshot()), 6)
        self.assertIs(buffer.get(), second)
        buffer.put(0, first, 3)
        self.assertEqual([buffer.get() for _ in range(5)], [1, 2, 3, 4, 5])
        self.assertIs(buffer.get(), first)
        self.assertIs(buffer.put(0, second, 2), second)
        self.assertIsNone(buffer.get(timeout=0))

    def test_close_wakes_waiter(self):
        buffer = LatestFrameBuffer(6)
        finished = threading.Event()
        waiter = threading.Thread(target=lambda: (buffer.get(timeout=5), finished.set()))
        waiter.start()
        buffer.close()
        self.assertTrue(finished.wait(.5))
        waiter.join()


class MonitoringTests(unittest.TestCase):
    def test_rolling_fps_cache_rate_and_latency_expire(self):
        with patch("time.perf_counter", return_value=100.):
            monitor = PerformanceMonitor(["camera_1"], window_sec=5)
            for key in ("input", "input", "yolo", "projection_cache_hits", "projection_cache_misses"):
                monitor.increment(0, key)
            monitor.observe(0, "image_age_ms", 10.)
            monitor.observe(0, "image_age_ms", 30.)
        camera = monitor.snapshot(now=102.)["cameras"][0]
        self.assertEqual(camera["fps"]["input"], 1.)
        self.assertEqual(camera["projection_cache"]["hit_rate"], .5)
        self.assertEqual(camera["observations"]["image_age_ms"]["p50"], 20.)
        self.assertEqual(camera["observations"]["image_age_ms"]["p95"], 29.)
        camera = monitor.snapshot(now=106.)["cameras"][0]
        self.assertEqual(camera["fps"]["input"], 0.)
        self.assertIsNone(camera["projection_cache"]["hit_rate"])
        self.assertIsNone(camera["observations"]["image_age_ms"]["p95"])
        self.assertEqual(camera["counters"]["total"]["input"], 2)

    def test_resource_gpu_scale_and_unavailable_is_null(self):
        with tempfile.NamedTemporaryFile(mode="w") as stream:
            stream.write("450\n")
            stream.flush()
            monitor = ResourceMonitor(stream.name)
            first = monitor.sample()
            self.assertEqual(first["gpu"]["percent"], 45.)
            self.assertIsNone(first["process_cpu_percent"])
            self.assertGreater(first["rss_bytes"], 0)
            self.assertTrue(first["cpu_per_core_percent"])
            json.dumps(monitor.sample(), allow_nan=False)
        monitor = ResourceMonitor("/no/such/gpu/load")
        self.assertIsNone(monitor.sample()["gpu"]["percent"])

    def test_orin_ga10b_auto_detection(self):
        with tempfile.NamedTemporaryFile(mode="w") as stream:
            stream.write("310")
            stream.flush()
            with patch("project_on_ad10_yolo.monitoring.glob.glob", side_effect=lambda pattern:
                       [stream.name] if pattern == "/sys/devices/platform/*ga10b/load" else []):
                gpu = ResourceMonitor()._gpu()
            self.assertEqual(gpu["source"], stream.name)
            self.assertEqual(gpu["percent"], 31.)


class PipelineTests(unittest.TestCase):
    def test_visualization_worker_survives_encoding_failure(self):
        node = make_node()
        failed = threading.Event()
        original_encode = node.bridge.cv2_to_imgmsg

        def encode(image, encoding):
            if not failed.is_set():
                failed.set()
                raise RuntimeError("encoding failure")
            return original_encode(image, encoding)

        node.bridge.cv2_to_imgmsg = encode
        worker = threading.Thread(target=node._visualization_worker)
        node.worker_threads = [worker]
        worker.start()
        try:
            node._process_frame(make_frame(1, cloud=False, version=1), None, None)
            self.assertTrue(failed.wait(1))
            node._process_frame(make_frame(1, cloud=False, version=2), None, None)
            self.assertTrue(node.image_pubs[1].published.wait(1))
            self.assertEqual(node.last_output_versions[1], 2)
            counters = node.metrics.snapshot()["cameras"][1]["counters"]["total"]
            self.assertEqual(counters["visualization_errors"], 1)
            self.assertEqual(counters["output"], 1)
        finally:
            node._shutdown_worker()

    def test_yolo_failure_does_not_count_success_or_publish_false_empty_detection(self):
        node = make_node()
        node._run_yolo = lambda image: (_ for _ in ()).throw(RuntimeError("inference failed"))
        node._process_frame(make_frame(1, cloud=False), None, None)
        node._process_visualization(node.visualization_buffer.get())
        self.assertEqual(node.yolo_detections_pubs[1].messages, [])
        counters = node.metrics.snapshot()["cameras"][1]["counters"]["total"]
        self.assertEqual(counters.get("yolo", 0), 0)
        self.assertEqual(counters["yolo_errors"], 1)
        self.assertEqual(counters["output"], 1)

    def test_slow_old_callback_cannot_overwrite_new_callback(self):
        node = make_node()
        entered, release = threading.Event(), threading.Event()
        old_stamp = time.time() - .1
        new_stamp = time.time()

        def lookup(index, stamp):
            if stamp < new_stamp - .05:
                entered.set()
                release.wait(2)
            return None, None, None, None, False

        node._select_lidar_for_image = lookup
        old = threading.Thread(target=node._image_callback,
                               args=(0, make_frame(stamp=old_stamp)["image"]))
        old.start()
        try:
            self.assertTrue(entered.wait(1))
            node._image_callback(0, make_frame(stamp=new_stamp)["image"])
        finally:
            release.set()
            old.join(2)
        task = node.input_buffer.get()
        self.assertAlmostEqual(task["frame"]["image_stamp"], new_stamp, places=6)
        self.assertEqual(task["frame"]["frame_version"], 2)

    def test_future_image_age_is_flagged_not_reported_as_negative_latency(self):
        node = make_node()
        node._observe_image_age(0, "image_age_ms", make_frame(stamp=time.time() + 1))
        camera = node.metrics.snapshot()["cameras"][0]
        self.assertEqual(camera["counters"]["total"]["negative_image_age_ms_samples"], 1)
        self.assertNotIn("image_age_ms", camera["observations"])

    def test_slow_projection_does_not_block_yolo_or_other_camera_output(self):
        node = make_node()
        entered, release = threading.Event(), threading.Event()
        original_draw = node._draw_cloud

        def slow_draw(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test projection release timed out")
            return original_draw(*args, **kwargs)

        node._draw_cloud = slow_draw
        node.worker_threads = [threading.Thread(target=method) for method in (
            node._inference_worker, node._projection_worker, node._visualization_worker)]
        for thread in node.worker_threads:
            thread.start()
        try:
            frame = make_frame(0)
            node._enqueue_frame(frame, None, None)
            self.assertTrue(entered.wait(1))
            self.assertTrue(node.yolo_detections_pubs[0].published.is_set())
            node._enqueue_frame(make_frame(1, cloud=False), None, None)
            self.assertTrue(node.image_pubs[1].published.wait(1))
            self.assertFalse(node.image_pubs[0].published.is_set())
            release.set()
            self.assertTrue(node.image_pubs[0].published.wait(1))
            self.assertEqual(node.image_pubs[0].messages[0].header.stamp.to_sec(), frame["image"].header.stamp.to_sec())
            self.assertEqual(node.image_pubs[0].messages[0].header.frame_id, "optical_0")
            self.assertFalse(frame["image"].pixels.any(), "input ROS image must remain immutable")
        finally:
            release.set()
            node._shutdown_worker()
        self.assertFalse(any(thread.is_alive() for thread in node.worker_threads))

    def test_projection_failure_degrades_to_yolo_and_preserves_frame_identity(self):
        node = make_node()
        frame = make_frame()
        node._process_frame(frame, None, None)
        task = node.projection_buffer.get()
        node._draw_cloud = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("bad cloud"))
        node._process_projection(task)
        output_task = node.visualization_buffer.get()
        self.assertIs(output_task["frame"], frame)
        node._process_visualization(output_task)
        detection = node.detections_pubs[0].messages[0]
        self.assertEqual(len(detection.detections), 1)
        self.assertEqual(detection.detections[0].id, "camera_1:yolo:0")
        self.assertEqual(node.metrics.snapshot()["cameras"][0]["counters"]["total"]["projection_errors"], 1)

    def test_missing_cloud_and_identical_stamps_do_not_block_new_versions(self):
        node = make_node()
        stamp = time.time()
        for version in (1, 2):
            frame = make_frame(cloud=False, stamp=stamp, version=version)
            node._process_frame(frame, None, None)
            self.assertIsNone(node.projection_buffer.get(timeout=0))
            node._process_visualization(node.visualization_buffer.get())
        self.assertEqual(len(node.image_pubs[0].messages), 2)
        self.assertEqual(node.last_output_versions[0], 2)

    def test_old_projection_result_cannot_replace_new_nonprojection_frame(self):
        node = make_node()
        old = make_frame(version=1)
        node._process_frame(old, None, None)
        task = node.projection_buffer.get()
        node._process_frame(make_frame(cloud=False, version=2), None, None)
        node._process_visualization(node.visualization_buffer.get())
        node._process_projection(task)
        self.assertIsNone(node.visualization_buffer.get(timeout=0))
        self.assertEqual(node.last_output_versions[0], 2)

    def test_expired_output_is_not_published(self):
        node = make_node()
        node._process_frame(make_frame(cloud=False), None, None)
        task = node.visualization_buffer.get()
        task["frame"]["received_monotonic"] -= 10
        node._process_visualization(task)
        self.assertEqual(node.image_pubs[0].messages, [])

    def test_lidar_selection_uses_past_timestamp_and_age_bound(self):
        node = make_node()
        for stamp in (9.9, 10.1, 9.8):
            node._insert_timestamped(node.lidar_cache, (stamp, "cloud%.1f" % stamp))
        node.bbox_cache.extend([(9.85, "bbox_old", 0), (9.95, "bbox_new", 0), (10.05, "future", 0)])
        cloud, stamp, bbox, bbox_stamp, reused = node._select_lidar_for_image(0, 10.)
        self.assertEqual((cloud, stamp), ("cloud9.9", 9.9))
        self.assertLessEqual(bbox_stamp, 10.)
        self.assertFalse(reused)
        self.assertTrue(node._select_lidar_for_image(0, 10.)[-1])
        self.assertIsNone(node._select_lidar_for_image(0, 11.)[0])

    def test_projection_cache_hits_and_invalidation(self):
        node = make_node()
        frame = make_frame()
        matrix, distortion, _ = node._camera_model_for_image(0, 32, 32)
        first, second = {}, {}
        image1 = np.zeros((32, 32, 3), np.uint8)
        image2 = image1.copy()
        self.assertEqual(node._draw_cloud(image1, 0, frame["cloud"], matrix, distortion, first), (2, 2))
        node._draw_cloud(image2, 0, frame["cloud"], matrix, distortion, second)
        self.assertFalse(first["projection_cache_hit"])
        self.assertTrue(second["projection_cache_hit"])
        self.assertEqual(second["cloud_convert_ms"], 0)
        np.testing.assert_array_equal(image1, image2)
        node.max_depth_m = 1.
        third = {}
        self.assertEqual(node._draw_cloud(image2, 0, frame["cloud"], matrix, distortion, third), (2, 0))
        self.assertFalse(third["projection_cache_hit"])

    def test_yolo_extracts_detached_cpu_data_without_plot(self):
        class Tensor:
            def __init__(self, array):
                self.array = np.array(array)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.array

        node = make_node()
        del node._run_yolo
        coordinates = Tensor([[1., 2., 10., 20.]])
        result = types.SimpleNamespace(names={3: "vehicle"}, boxes=types.SimpleNamespace(
            cls=Tensor([3.]), conf=Tensor([.9]), xyxy=coordinates))
        node.yolo_model = types.SimpleNamespace(predict=lambda **kw: [result])
        node.yolo_image_size, node.yolo_confidence, node.yolo_iou, node.yolo_device = 640, .25, .45, "0"
        boxes, timings = node._run_yolo(np.zeros((32, 32, 3), np.uint8))
        self.assertEqual(boxes[0][:3], (3, "vehicle", .9))
        coordinates.array[:] = 0
        self.assertEqual(boxes[0][3][0], 1.)
        self.assertIn("yolo_predict_ms", timings)

    def test_full_constructor_starts_all_stages_and_monitor(self):
        with tempfile.NamedTemporaryFile() as weights:
            params = {"~yolo_model_path": weights.name, "~monitor_log_interval_sec": 0,
                      "~gpu_load_path": "/no/such/gpu"}
            model_module = types.ModuleType("ultralytics")
            model_module.YOLO = lambda path: types.SimpleNamespace(predict=lambda **kw: [])
            with patch.dict(sys.modules, {"ultralytics": model_module}), patch.object(
                    rospy, "get_param", side_effect=lambda name, default=None: params.get(name, default)):
                node = ProjectBboxNode()
                try:
                    self.assertTrue(node.monitor_pub.published.wait(2))
                    data = json.loads(node.monitor_pub.messages[-1].data)
                    self.assertEqual(len(data["cameras"]), 6)
                    self.assertEqual(len(node.worker_threads), 4)
                    node._image_callback(1, make_frame(1, cloud=False)["image"])
                    self.assertTrue(node.image_pubs[1].published.wait(1))
                    self.assertTrue(node.yolo_detections_pubs[1].published.is_set())
                finally:
                    node._shutdown_worker()
                self.assertFalse(any(thread.is_alive() for thread in node.worker_threads))


if __name__ == "__main__":
    unittest.main()
