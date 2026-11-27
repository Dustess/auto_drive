"""每路单槽、轮询取帧；只在短临界区内交接任务所有权。"""

import threading
import time


class LatestFrameBuffer:
    def __init__(self, camera_count):
        self._slots = [None] * camera_count
        self._versions = [-1] * camera_count
        self._next_camera = 0
        self._closed = False
        self._condition = threading.Condition()

    def put(self, camera_index, item, version):
        """返回被覆盖项；迟到版本或已关闭时返回传入项本身。"""
        with self._condition:
            if self._closed or version <= self._versions[camera_index]:
                return item
            previous = self._slots[camera_index]
            self._slots[camera_index] = item
            self._versions[camera_index] = version
            self._condition.notify()
            return previous

    def get(self, timeout=0.1):
        deadline = time.perf_counter() + timeout
        with self._condition:
            while not self._closed:
                for offset in range(len(self._slots)):
                    index = (self._next_camera + offset) % len(self._slots)
                    item = self._slots[index]
                    if item is not None:
                        self._slots[index] = None
                        self._next_camera = (index + 1) % len(self._slots)
                        return item
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
        return None

    def snapshot(self):
        with self._condition:
            return [item is not None for item in self._slots]

    def close(self):
        with self._condition:
            self._closed = True
            self._slots = [None] * len(self._slots)
            self._condition.notify_all()
