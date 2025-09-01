#!/usr/bin/env python3
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, send_file
import threading, io, time
from PIL import Image
import numpy as np

# --- Camera setup ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280//2, 720//2), "format": "RGB888"}  # tweak resolution/format as needed
)
picam2.configure(config)
picam2.start()

# --- Frame grab thread ---
latest_jpeg = None
cv = threading.Condition()
running = True

def capture_loop():
    global latest_jpeg, running
    while running:
        # Grab frame as numpy array (RGB888)
        frame = picam2.capture_array()[:,:,::-1]

        #################################################
        ### HERE WE CAN DO RANDOM STUFF TO THE FRAME  ###
        #################################################



        




        # And here it gets converted and so on...
        frame = frame.astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255

        if len(frame.shape) == 2:
            frame = np.repeat(frame[..., None], 3, 2).astype(np.uint8)

        # Encode to JPEG (quality ~85, adjust if you need smaller/larger)
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=85, optimize=True)
        jpeg_bytes = buf.getvalue()
        with cv:
            latest_jpeg = jpeg_bytes
            cv.notify_all()
        # tiny sleep to avoid pegging CPU if nothing is consuming
        time.sleep(0.01)

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

# --- Web app ---
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Raspberry Pi Camera</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#111; color:#eee; }
    header { padding: 12px 16px; background:#000; position: sticky; top:0; }
    main { display:flex; justify-content:center; align-items:center; min-height: calc(100vh - 52px); }
    img { max-width: 100%; height: auto; background:#000; }
    .bar { display:flex; gap:12px; align-items:center; }
    a,button { color:#eee; text-decoration:none; background:#222; border:1px solid #333; padding:8px 12px; border-radius:8px; }
    a:hover,button:hover { background:#333; }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <strong>Raspberry Pi Live Stream</strong>
      <a href="/snapshot.jpg" target="_blank">Snapshot</a>
    </div>
  </header>
  <main>
    <img src="/stream.mjpg" alt="Live Stream (MJPEG)" />
  </main>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/snapshot.jpg")
def snapshot():
    # return the latest JPEG once
    with cv:
        if latest_jpeg is None:
            # block briefly for first frame
            cv.wait(timeout=1.0)
        data = latest_jpeg or b""
    return Response(data, mimetype="image/jpeg")

@app.route("/stream.mjpg")
def mjpeg():
    def gen():
        boundary = "--frame"
        # Send an initial frame quickly if we have one
        last = None
        while True:
            with cv:
                if latest_jpeg is None:
                    cv.wait(timeout=1.0)
                # Only push when we have a new frame
                if latest_jpeg is last:
                    cv.wait(timeout=0.05)
                data = latest_jpeg
                last = data
            if data is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
                data + b"\r\n"
            )
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def cleanup():
    global running
    running = False
    try:
        picam2.stop()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        # 0.0.0.0 to serve on your LAN; change port if you like
        app.run(host="0.0.0.0", port=8000, threaded=True)
    finally:
        cleanup()
