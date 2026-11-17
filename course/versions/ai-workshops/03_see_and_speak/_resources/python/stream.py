"""Thin adapter around the working stream-NB3 HTTP server and JPEG path."""
import json
import logging
from pathlib import Path
import socket
import sys
import threading
import time

from config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "libs"))
from NB3.Server.server import Server, get_wifi_interface


class Stream(Server):
    """Reuse NB3 HTTP/MJPEG framing; isolate socket writes from capture."""
    def __init__(self, port, width=800, height=600):
        self.condition = threading.Condition()
        self.delivery_lock = threading.Lock()
        self.jpeg = None
        self.sequence = self.delivered = 0
        self.last_frame = None
        self.width, self.height = width, height
        self.result = {"sequence": 0, "timestamp": 0, "detections": [], "inference_ms": 0}
        self.logger = logging.getLogger("demo3.http")
        super().__init__(root=str(Path(__file__).parent), port=port,
                         interface=get_wifi_interface() or "lo")
        self.ip_address = "0.0.0.0"
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.ip_address, self.port))
        self.port = self.server_socket.getsockname()[1]
        self.server_socket.listen(5)
        self.running = True
        self.server_thread = threading.Thread(target=self._accept_clients, daemon=True)
        self.sender = threading.Thread(target=self._deliver, daemon=True)
        self.server_thread.start()
        self.sender.start()

    def extract_stream_names(self, index_path):
        return ["camera"]

    def _accept_clients(self):
        while self.running:
            try:
                client, _ = self.server_socket.accept()
                client.settimeout(2)
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except OSError:
                if self.running:
                    self.logger.exception("HTTP accept failed")

    def serve_stream(self, client_socket, stream_name):
        # NB3 registers the socket before sending its HTTP headers. Serialize
        # that handshake with broadcast so a JPEG cannot precede the headers.
        with self.delivery_lock:
            super().serve_stream(client_socket, stream_name.split("?", 1)[0])

    def serve_file(self, client_socket, file_path):
        name = Path(file_path).name.split("?", 1)[0]
        if name == "detections.json":
            with self.condition:
                age = time.monotonic() - self.result["timestamp"]
                result = {**self.result, "age": age, "width": self.width, "height": self.height,
                          "video_sequence": self.sequence, "delivered": self.delivered,
                          "frame_age": None if self.last_frame is None else time.monotonic() - self.last_frame}
                if age > 1:
                    result["detections"] = []
            body = json.dumps(result).encode()
            client_socket.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
                                  + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        elif name in ("index.html", "viewer.html"):
            super().serve_file(client_socket, str(Path(__file__).parent / "viewer.html"))
        else:
            self.send_404(client_socket)

    def publish(self, jpeg):
        # No decoding, annotation or re-encoding in the video path.
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.last_frame = time.monotonic()
            self.condition.notify_all()
            return self.sequence, self.last_frame

    def set_detections(self, detections, sequence, timestamp, inference_ms):
        with self.condition:
            self.result = {"sequence": sequence, "timestamp": timestamp,
                           "inference_ms": inference_ms,
                           "detections": [{"label": d.label, "score": d.score, "bbox": d.bbox}
                                          for d in detections]}

    def _deliver(self):
        sequence = 0
        try:
            while self.running:
                with self.condition:
                    self.condition.wait_for(lambda: not self.running or self.sequence != sequence)
                    if not self.running:
                        return
                    sequence, jpeg = self.sequence, self.jpeg
                # Keep the working NB3 framing/broadcast. If a slow client
                # delays this sender, capture/inference continue independently.
                with self.delivery_lock:
                    super().update_stream("camera", jpeg)
                self.delivered = sequence
        except Exception:
            self.logger.exception("MJPEG sender failed")

    def close(self):
        self.stop()
        with self.condition:
            self.condition.notify_all()
        self.sender.join(timeout=3)
        self.server_thread.join(timeout=3)
