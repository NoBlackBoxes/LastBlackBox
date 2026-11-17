"""Compare real reference, camera-only and Coral streams; all phases silent.

Run with the existing workshop Python. Reference/camera-only run 20 seconds
of measurements each, then Coral runs 310 seconds. No browser claims are made.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

from config import REPO_ROOT

ROOT = Path(__file__).resolve().parent


def read_frame(response, decode=False, dimensions=(600, 800)):
    assert response.readline().strip() == b"--frame"
    headers = {}
    while True:
        line = response.readline().strip()
        if not line:
            break
        key, value = line.split(b":", 1)
        headers[key.lower()] = value.strip()
    data = response.read(int(headers[b"content-length"]))
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    if decode:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[:2] == dimensions
    return len(data)


def cpu_ticks(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[13]) + int(fields[14])


def phase(name, command, url, seconds, dimensions=(600, 800)):
    logs = []
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "libs"))
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def collect():
        for line in process.stdout:
            logs.append(line.strip())
    thread = threading.Thread(target=collect)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                response = urllib.request.urlopen(url + "/stream/camera.mjpg", timeout=5)
                break
            except OSError:
                assert process.poll() is None and time.monotonic() < deadline, logs
                time.sleep(.2)
        with response:
            read_frame(response, True, dimensions)
            time.sleep(1)
            # Discard warm-up buffered data by reconnecting.
        with urllib.request.urlopen(url + "/stream/camera.mjpg", timeout=5) as response:
            start, ticks = time.monotonic(), cpu_ticks(process.pid)
            frames, size, gaps, previous = 0, 0, [], start
            while time.monotonic() - start < seconds:
                size += read_frame(response, frames % 30 == 0, dimensions)
                now = time.monotonic()
                gaps.append(now - previous)
                previous = now
                frames += 1
            elapsed = time.monotonic() - start
            cpu = 100 * (cpu_ticks(process.pid) - ticks) / os.sysconf("SC_CLK_TCK") / elapsed
            rss = next(line for line in Path(f"/proc/{process.pid}/status").read_text().splitlines() if line.startswith("VmRSS:"))
        result = {"phase": name, "seconds": round(elapsed, 2), "frames": frames,
                  "fps": round(frames / elapsed, 2), "max_gap": round(max(gaps), 3),
                  "mbps": round(8 * size / elapsed / 1e6, 2), "cpu_one_core_percent": round(cpu, 1), "rss": rss}
        assert max(gaps) < 3, result
        if name != "reference":
            with urllib.request.urlopen(url + "/detections.json", timeout=3) as response:
                state = json.load(response)
            result["inference_ms"] = round(state["inference_ms"], 2)
            result["inference_sequence"] = state["sequence"]
            if name == "coral":
                assert state["sequence"] > 0 and state["age"] < 1, state
                assert all(len(item["bbox"]) == 4 and 0 <= item["score"] <= 1 for item in state["detections"])
        print(json.dumps(result), flush=True)
        return result
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise AssertionError("Shutdown timed out")
        thread.join(timeout=2)
        (ROOT / f"{name}.log").write_text("\n".join(logs) + "\n")
        assert process.returncode == 0, logs[-20:]


def run():
    sys.path.insert(0, str(REPO_ROOT / "libs"))
    from NB3.Server.server import get_wifi_interface
    import netifaces
    interface = get_wifi_interface() or "lo"
    address = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]["addr"]
    original = [sys.executable, "-B", str(REPO_ROOT / "boxes/vision/stream-NB3/stream.py")]
    demo = [sys.executable, "-B", str(ROOT / "main.py"), "--silent", "--stream"]
    results = [phase("reference", original, f"http://{address}:1234", 20)]
    results.append(phase("camera_only", demo + ["--camera-only"], "http://127.0.0.1:8765", 20))
    # Camera-only must succeed before starting Coral integration testing.
    results.append(phase("coral", demo + ["--confidence", "0.5"], "http://127.0.0.1:8765", 310))
    (ROOT / "stream_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0", 8765))
    from vision import Vision
    vision = Vision(.5)
    try:
        vision.start_camera(800, 600)
        assert vision.detect(vision.camera.capture()) is not None
    finally:
        vision.close()
    print("PASS sequential camera-only/Coral hardware tests, clean shutdown and camera/Coral reuse. Mac browser validation remains manual.", flush=True)


if __name__ == "__main__":
    run()
