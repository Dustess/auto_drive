#!/usr/bin/env python3
"""将 /fusion/performance 的 JSON 指标显示为六路相机的实时表格。"""

import argparse
import json

import rospy
from std_msgs.msg import String


def _number(value):
    return "-" if value is None else "%.1f" % value


def _p95(camera, name):
    return _number(camera.get("observations", {}).get(name, {}).get("p95"))


def _show(message, camera_filter):
    try:
        snapshot = json.loads(message.data)
    except (TypeError, ValueError) as exc:
        rospy.logwarn_throttle(5.0, "监控 JSON 无法解析: %s", exc)
        return

    print("\n滚动窗口 %.1f s；耗时单位 ms，显示 P95（- 表示窗口内没有样本）" %
          snapshot.get("window_sec", 0.0))
    print("相机  FPS 输入/YOLO/输出  YOLO预测  结果提取  点云绘制  等待:输入/投影/绘图  图像年龄")
    for camera in snapshot.get("cameras", []):
        if camera_filter and camera["index"] + 1 != camera_filter:
            continue
        fps = camera.get("fps", {})
        fps_text = "/".join(_number(fps.get(name)) for name in ("input", "yolo", "output"))
        wait_text = "/".join(_p95(camera, name) for name in (
            "input_queue_wait_ms", "projection_queue_wait_ms",
            "visualization_queue_wait_ms"))
        print("%-5s %-20s %-9s %-9s %-9s %-22s %s" % (
            camera.get("name", "?"), fps_text,
            _p95(camera, "yolo_predict_ms"),
            _p95(camera, "yolo_extract_ms"),
            _p95(camera, "cloud_draw_ms"),
            wait_text, _p95(camera, "image_age_ms")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, choices=range(1, 7),
                        help="只显示指定相机，编号 1–6")
    parser.add_argument("--topic", default="/fusion/performance",
                        help="监控话题，默认 /fusion/performance")
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("view_fusion_performance", anonymous=True)
    rospy.Subscriber(args.topic, String, lambda msg: _show(msg, args.camera), queue_size=1)
    print("等待监控消息：%s；Ctrl+C 退出" % args.topic)
    rospy.spin()


if __name__ == "__main__":
    main()
