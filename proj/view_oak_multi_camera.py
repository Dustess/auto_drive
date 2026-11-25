#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""用 Qt 窗口分别显示真实 OAK 图像和五路模拟图像。"""

import sys

import rospy
from python_qt_binding.QtCore import QObject, Qt, Signal
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from sensor_msgs.msg import Image


class ImageSignals(QObject):
    # ROS 回调在后台线程运行，通过 Qt 信号在 GUI 线程更新面板。
    image_received = Signal(int, object)


class CameraPanel(QFrame):
    def __init__(self, topic):
        super().__init__()
        self.current_image = None
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.topic_label = QLabel(topic)
        self.topic_label.setAlignment(Qt.AlignCenter)
        self.topic_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.topic_label)

        self.image_label = QLabel("等待图像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 200)
        self.image_label.setStyleSheet("background-color: #111; color: #ddd;")
        layout.addWidget(self.image_label, 1)

    def set_image(self, image):
        self.current_image = image
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self.current_image is None:
            return
        pixmap = QPixmap.fromImage(self.current_image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class MultiCameraWindow(QMainWindow):
    def __init__(self, topics, title):
        super().__init__()
        self.setWindowTitle(title)
        self.panels = []

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        for index, topic in enumerate(topics):
            panel = CameraPanel(topic)
            self.panels.append(panel)
            grid.addWidget(panel, index // 3, index % 3)

        self.setCentralWidget(container)
        self.resize(1440, 900)

    def update_image(self, index, image):
        self.panels[index].set_image(image)

    def closeEvent(self, event):
        if not rospy.is_shutdown():
            rospy.signal_shutdown("用户关闭图像窗口")
        event.accept()


class OakMultiCameraViewer:
    def __init__(self, app):
        default_topics = ["/oak/rgb/image_raw"] + [
            "/oak/rgb/image_raw_%d" % index for index in range(1, 6)
        ]
        configured_topics = rospy.get_param("~image_topics", default_topics)
        if not isinstance(configured_topics, (list, tuple)) or len(configured_topics) != 6:
            raise rospy.ROSInitException(
                "~image_topics 必须包含原始图像和五路模拟图像，共六个话题"
            )

        self.image_topics = [rospy.resolve_name(str(topic)) for topic in configured_topics]
        title = str(rospy.get_param("~window_name", "OAK Multi-Camera Images"))
        self.app = app
        self.window = MultiCameraWindow(self.image_topics, title)
        self.signals = ImageSignals()
        self.signals.image_received.connect(self.window.update_image)

        self.subscribers = [
            rospy.Subscriber(
                topic,
                Image,
                lambda msg, index=index: self._image_callback(index, msg),
                queue_size=1,
                buff_size=2**24,
            )
            for index, topic in enumerate(self.image_topics)
        ]
        rospy.on_shutdown(self.app.quit)
        rospy.loginfo(
            "六路 OAK 图像查看节点已启动，面板顺序为：%s",
            ", ".join(self.image_topics),
        )

    @staticmethod
    def _to_qimage(msg):
        """直接从 ROS Image 缓冲区创建 QImage，不经过 OpenCV。"""
        width = int(msg.width)
        height = int(msg.height)
        step = int(msg.step)
        encoding = msg.encoding.lower()
        data = bytes(msg.data)

        if width <= 0 or height <= 0 or step <= 0:
            raise ValueError("图像尺寸或 step 无效")

        if encoding in ("rgb8", "bgr8"):
            image = QImage(data, width, height, step, QImage.Format_RGB888)
            if encoding == "bgr8":
                image = image.rgbSwapped()
        elif encoding in ("rgba8", "bgra8"):
            image = QImage(data, width, height, step, QImage.Format_RGBA8888)
            if encoding == "bgra8":
                image = image.rgbSwapped()
        elif encoding in ("mono8", "8uc1"):
            image = QImage(data, width, height, step, QImage.Format_Grayscale8)
        elif encoding == "8uc3":
            image = QImage(data, width, height, step, QImage.Format_RGB888).rgbSwapped()
        else:
            raise ValueError("暂不支持图像编码 %s" % msg.encoding)

        if image.isNull():
            raise ValueError("Qt 无法读取图像缓冲区")
        # copy() 让 QImage 自己持有像素数据，避免 ROS 消息释放后缓冲区失效。
        return image.copy()

    def _image_callback(self, index, msg):
        try:
            image = self._to_qimage(msg)
        except (TypeError, ValueError) as exc:
            rospy.logerr_throttle(
                3.0,
                "话题 %s 图像转换失败：%s",
                self.image_topics[index],
                exc,
            )
            return

        self.signals.image_received.emit(index, image)

    def run(self):
        self.window.show()
        return self.app.exec_()


if __name__ == "__main__":
    try:
        rospy.init_node("oak_multi_camera_viewer")
        app = QApplication(rospy.myargv(sys.argv))
        viewer = OakMultiCameraViewer(app)
        viewer.run()
    except rospy.ROSInterruptException:
        pass
