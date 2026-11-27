"""低开销滚动指标及 Linux/Jetson 资源采样，不依赖 ROS、psutil 或 CUDA。"""

import glob
import math
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict, deque


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "latest": None}
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": percentile(0.5), "p95": percentile(0.95), "latest": values[-1]}


class PerformanceMonitor:
    def __init__(self, camera_names, window_sec=5.0):
        self.camera_names = list(camera_names)
        self.window_sec = max(0.1, float(window_sec))
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        # 每路每项最多 4096 个样本；限制异常高输入率/长窗口时的内存。
        self._events = [defaultdict(lambda: deque(maxlen=4096)) for _ in camera_names]
        self._values = [defaultdict(lambda: deque(maxlen=4096)) for _ in camera_names]
        self._totals = [defaultdict(int) for _ in camera_names]

    def increment(self, camera_index, name, amount=1):
        now = time.perf_counter()
        with self._lock:
            self._events[camera_index][name].append((now, amount))
            self._totals[camera_index][name] += amount

    def observe(self, camera_index, name, value):
        if value is None or not math.isfinite(float(value)):
            return
        now = time.perf_counter()
        with self._lock:
            self._values[camera_index][name].append((now, float(value)))

    def snapshot(self, now=None):
        now = time.perf_counter() if now is None else now
        cutoff = now - self.window_sec
        duration = max(1e-6, min(self.window_sec, now - self.started))
        copied = []
        with self._lock:
            for index, name in enumerate(self.camera_names):
                for series in (self._events[index], self._values[index]):
                    for samples in series.values():
                        while samples and samples[0][0] < cutoff:
                            samples.popleft()
                counters = {key: sum(value for _, value in samples)
                            for key, samples in self._events[index].items()}
                values = {key: [value for _, value in samples]
                          for key, samples in self._values[index].items()}
                copied.append((index, name, counters, dict(self._totals[index]), values))
        cameras = []
        for index, name, counters, totals, values in copied:
            hits = counters.get("projection_cache_hits", 0)
            misses = counters.get("projection_cache_misses", 0)
            cameras.append({
                "index": index, "name": name,
                "fps": {key: counters.get(key, 0) / duration
                        for key in ("input", "yolo", "output", "projection")},
                "counters": {"window": counters, "total": totals},
                "observations": {key: _summary(samples) for key, samples in values.items()},
                "projection_cache": {"hits": hits, "misses": misses,
                                     "hit_rate": hits / (hits + misses) if hits + misses else None},
            })
        return {"timestamp_monotonic": now, "window_sec": self.window_sec,
                "effective_window_sec": duration, "sample_limit_per_metric": 4096,
                "total_yolo_fps": sum(item["fps"]["yolo"] for item in cameras),
                "cameras": cameras}


class ResourceMonitor:
    def __init__(self, gpu_load_path="", gpu_load_scale=0.1):
        self._previous_cpu = {}
        self._previous_tasks = {}
        self._previous_time = None
        self._ticks = os.sysconf("SC_CLK_TCK")
        self._page_size = os.sysconf("SC_PAGE_SIZE")
        self._pid = os.getpid()
        self._gpu_path = gpu_load_path
        self._gpu_scale = float(gpu_load_scale)
        self._smi = shutil.which("nvidia-smi")

    @staticmethod
    def _cpu_times():
        result = {}
        with open("/proc/stat", "r") as stream:
            for line in stream:
                fields = line.split()
                if fields and fields[0].startswith("cpu"):
                    ticks = [int(value) for value in fields[1:9]]
                    result[fields[0]] = (sum(ticks), ticks[3] + ticks[4])
        return result

    @staticmethod
    def _task_stat(path):
        with open(path, "r") as stream:
            raw = stream.read()
        closing = raw.rfind(")")
        fields = raw[closing + 2:].split()  # starts at field 3 (state)
        return raw[raw.find("(") + 1:closing], int(fields[11]) + int(fields[12]), int(fields[21])

    def _gpu(self):
        paths = [self._gpu_path] if self._gpu_path else []
        if not paths:
            # Orin (JetPack 5) 使用 17000000.ga10b，而非含 gpu 的目录名。
            # NVIDIA Jetson Linux r35.5 Clocks 文档：load / 10 为百分比。
            for device in ("*ga10b", "*gv11b", "*gp10b", "*gpu*"):
                for parent in ("/sys/devices/platform", "/sys/devices",
                               "/sys/devices/platform/bus@0", "/sys/devices/platform/host1x"):
                    paths.extend(sorted(glob.glob(parent + "/" + device + "/load")))
                paths.extend(sorted(glob.glob("/sys/class/devfreq/" + device + "/device/load")))
        errors = []
        for path in paths:
            try:
                with open(path, "r") as stream:
                    percent = float(stream.read().strip()) * self._gpu_scale
                if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
                    raise ValueError("GPU load out of range; check gpu_load_scale")
                return {"percent": percent, "source": path, "devices": [], "error": None}
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        # 显式路径错误必须可见，不能悄悄切换到另一 GPU。
        if self._smi and not self._gpu_path:
            try:
                output = subprocess.run(
                    [self._smi, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=0.8, check=True,
                ).stdout
                devices = []
                for line in output.strip().splitlines():
                    index, utilization, used, total = [part.strip() for part in line.split(",")]
                    devices.append({"index": int(index), "percent": float(utilization),
                                    "memory_used_mib": float(used), "memory_total_mib": float(total)})
                if devices:
                    return {"percent": max(item["percent"] for item in devices),
                            "source": "nvidia-smi (max across devices)",
                            "devices": devices, "error": None}
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors.append(str(exc))
        return {"percent": None, "source": None, "devices": [],
                "error": "; ".join(errors) or "No readable GPU load source"}

    def sample(self):
        gpu = self._gpu()
        now = time.perf_counter()
        elapsed = now - self._previous_time if self._previous_time is not None else None
        result = {"cpu_percent": None, "cpu_per_core_percent": {},
                  "process_cpu_percent": None, "threads": [], "ram": {}, "rss_bytes": None,
                  "gpu": gpu, "errors": []}
        try:
            current_cpu = self._cpu_times()
            for name, (total, idle) in current_cpu.items():
                previous = self._previous_cpu.get(name)
                percent = None
                if previous and total > previous[0]:
                    percent = max(0.0, min(100.0, 100.0 * (
                        1.0 - (idle - previous[1]) / (total - previous[0]))))
                if name == "cpu":
                    result["cpu_percent"] = percent
                else:
                    result["cpu_per_core_percent"][name] = percent
            self._previous_cpu = current_cpu
        except (OSError, ValueError, IndexError) as exc:
            result["errors"].append(str(exc))
        tasks = {}
        python_threads = {thread.native_id: thread.name for thread in threading.enumerate()}
        paths = [("process", "/proc/%d/stat" % self._pid)]
        paths.extend((path.split("/")[-2], path)
                     for path in glob.glob("/proc/%d/task/*/stat" % self._pid))
        for key, path in paths:
            try:
                name, ticks, rss = self._task_stat(path)
                tasks[key] = ticks
                previous = self._previous_tasks.get(key)
                percent = (max(0, ticks - previous) / self._ticks / elapsed * 100.0
                           if previous is not None and elapsed and elapsed > 0 else None)
                if key == "process":
                    result["process_cpu_percent"] = percent
                    result["rss_bytes"] = rss * self._page_size
                else:
                    result["threads"].append({"tid": int(key),
                                              "name": python_threads.get(int(key), name),
                                              "cpu_percent": percent})
            except (OSError, ValueError, IndexError):
                continue  # 线程可能在枚举后退出。
        self._previous_tasks = tasks
        self._previous_time = now
        try:
            with open("/proc/meminfo", "r") as stream:
                memory = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in stream}
            total = memory["MemTotal"]
            available = memory.get("MemAvailable", memory.get("MemFree", 0))
            result["ram"] = {"total_bytes": total, "available_bytes": available,
                             "used_bytes": total - available,
                             "used_percent": (total - available) / total * 100.0}
        except (OSError, ValueError, KeyError) as exc:
            result["errors"].append(str(exc))
        return result
