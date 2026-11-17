"""Camera → Coral → a short sentence → ElevenLabs → NB3 speaker."""
import argparse
import logging
import socket
import time

import config


class Trigger:
    """Require consecutive observations and a global pause after each utterance."""
    def __init__(self, stable_frames, cooldown):
        self.stable_frames, self.cooldown = stable_frames, cooldown
        self.reset()
        self.ready_at = 0.0

    def reset(self):
        self.label, self.count = None, 0

    def update(self, label, now):
        if now < self.ready_at or label is None:
            self.reset()
            return False
        self.count = self.count + 1 if label == self.label else 1
        self.label = label
        return self.count >= self.stable_frames

    def spoken(self, now):
        self.ready_at = now + self.cooldown
        self.reset()


def sentence(label):
    article = "an" if label[0].lower() in "aeiou" else "a"
    return f"I see {article} {label}!"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("demo3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silent", action="store_true", help="Print responses without speech imports, credentials, API calls or playback")
    parser.add_argument("--humor", action="store_true", help="Use NB3's local playful responses instead of plain object announcements")
    parser.add_argument("--stream", action="store_true", help="Serve annotated MJPEG on the local network")
    parser.add_argument("--camera-only", action="store_true", help="Test the reference JPEG stream without Coral or speech")
    parser.add_argument("--width", type=int, default=800, help="Stream width (reference default: 800)")
    parser.add_argument("--height", type=int, default=600, help="Stream height (reference default: 600)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP streaming port (default: 8765)")
    parser.add_argument("--detect-only", action="store_true", help="Print speech triggers without cloud/audio")
    parser.add_argument("--image", help="Detect once in a saved image, without camera or speech")
    parser.add_argument("--say", metavar="TEXT", help="Test ElevenLabs and speaker only")
    parser.add_argument("--env-file", help="Existing dotenv credentials file (path only; never pass an API key)")
    parser.add_argument("--confidence", type=float, default=config.CONFIDENCE)
    parser.add_argument("--only", metavar="CLASS", type=lambda value: ' '.join(value.lower().split()),
                        help="Speak only about this COCO class (e.g. bottle or banana); keep all browser overlays")
    parser.add_argument("--stable-frames", type=int, default=config.STABLE_FRAMES)
    parser.add_argument("--cooldown", type=float, default=config.COOLDOWN_SECONDS)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1 or args.stable_frames < 1 or args.cooldown < 0:
        parser.error("confidence must be in (0,1], stable-frames >= 1 and cooldown >= 0")
    if sum(bool(value) for value in (args.say, args.image, args.detect_only)) > 1:
        parser.error("Choose only one of --say, --image or --detect-only")
    if args.silent and args.say:
        parser.error("--silent cannot be combined with --say")
    if args.only is not None:
        if args.say:
            parser.error("--only filters detections and cannot be combined with --say")
        try:
            labels = set(config.LABELS_PATH.read_text().splitlines()) - {"n/a", ""}
        except OSError as error:
            parser.error(f"Cannot load object labels: {error}")
        if args.only not in labels:
            parser.error(f"Unknown COCO class {args.only!r}; use a label such as bottle, banana or 'wine glass'")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if args.stream and (args.say or args.image):
        parser.error("--stream requires live camera mode")

    if args.camera_only and not args.stream:
        parser.error("--camera-only requires --stream")
    if args.width < 64 or args.height < 64 or args.width % 2 or args.height % 2:
        parser.error("camera dimensions must be even and at least 64")

    try:
        if args.say:
            from speech import Speech
            Speech(args.env_file).say(args.say)
            return
        personality = None
        if args.humor:
            from personality import Personality
            personality = Personality()
        if args.stream:
            run_stream(args, personality)
            return
        from vision import Vision
        vision = Vision(args.confidence)
        try:
            if args.image:
                import cv2
                frame = cv2.imread(args.image)
                if frame is None:
                    raise ValueError("Cannot read image")
                print_detections(vision.detect(frame))
                return
            speech = None
            if not (args.silent or args.detect_only):
                from speech import Speech
                speech = Speech(args.env_file)
            vision.start_camera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            trigger = Trigger(args.stable_frames, args.cooldown)
            while True:
                detections = vision.detect(vision.camera.capture())
                print_detections(detections)
                respond(detections, trigger, speech, args.silent, args.only, personality)
                time.sleep(config.FRAME_INTERVAL)
        finally:
            vision.close()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        from diagnostics import report_exception
        report_exception("Demo failed", error)
        raise SystemExit(1) from None


def respond(detections, trigger, speech, silent, only=None, personality=None):
    label = next((item.label for item in detections if item.label == "person"), None)

    if trigger.update(label, time.monotonic()):
        text = (
            "Welcome to the No Black Boxes workshop "
            "at the Global Innovation Forum! We're glad you're here."
        )
        prefix = "NB3 (silent)" if silent else "Would say" if speech is None else "NB3"
        print(f"{prefix}: {text}", flush=True)
        if speech is not None:
            speech.say(text)
        trigger.spoken(time.monotonic())



def run_stream(args, personality=None):
    # Same camera, encoder and JPEG capture as boxes/vision/stream-NB3/stream.py.
    import sys
    sys.path.insert(0, str(config.REPO_ROOT / "libs"))
    from NB3.Vision.camera import Camera
    from stream import Stream

    camera = Camera(width=args.width, height=args.height)
    stream = worker = None
    logger = logging.getLogger("demo3.video")
    try:
        camera.start()
        stream = Stream(args.port, args.width, args.height)
        print(f"Live video: http://{socket.gethostname()}.local:{args.port}", flush=True)
        if not args.camera_only:
            from vision import DetectorWorker
            trigger = Trigger(args.stable_frames, args.cooldown)
            speech = None
            if not (args.silent or args.detect_only):
                from speech import Speech
                speech = Speech(args.env_file)
            last_report = 0.0
            def on_result(detections, sequence, timestamp, inference_ms):
                nonlocal last_report
                stream.set_detections(detections, sequence, timestamp, inference_ms)
                if time.monotonic() - last_report >= 1:
                    print_detections(detections)
                    last_report = time.monotonic()
                respond(detections, trigger, speech, args.silent, args.only, personality)
            worker = DetectorWorker(args.confidence, on_result)
        print("Silent camera-only stream" if args.camera_only else "Streaming with asynchronous Coral detection", flush=True)
        metrics_at, frames = time.monotonic(), 0
        inference_failed = False
        while True:
            jpeg = camera.capture(mjpeg=True)
            sequence, timestamp = stream.publish(jpeg)
            if worker is not None:
                if not worker.error:
                    worker.submit(jpeg, sequence, timestamp)
                elif not inference_failed:
                    logger.error("Coral unavailable (%s); video continues without detections", worker.error)
                    inference_failed = True
            frames += 1
            now = time.monotonic()
            if now - metrics_at >= 10:
                logger.info("encoder/video %.1f fps; produced=%d delivered=%d clients=%d inference=%.1fms",
                            frames / (now - metrics_at), sequence, stream.delivered,
                            len(stream.stream_clients["camera"]), stream.result["inference_ms"])
                frames, metrics_at = 0, now
            time.sleep(.0333)  # The working stream-NB3 example's delivery cadence.
    finally:
        try:
            if stream is not None:
                stream.close()
        finally:
            try:
                camera.stop()
            finally:
                camera.handle.close()
                if worker is not None:
                    worker.close()


def print_detections(detections):
    print(", ".join(f"{detection.label}: {detection.score:.0%}" for detection in detections) or "No confident objects", flush=True)


if __name__ == "__main__":
    main()
