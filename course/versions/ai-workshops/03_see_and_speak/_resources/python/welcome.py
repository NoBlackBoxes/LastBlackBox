
"""NB3-11: Conversational welcome robot with live video streaming."""

import importlib.util
import logging
import os
import socket
import sys
import threading
import time

from dotenv import load_dotenv

from config import (
    AUDIO_PYTHON,
    REPO_ROOT,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONFIDENCE,
    STABLE_FRAMES,
)
from speech import Speech
from vision import DetectorWorker
from stream import Stream


PORT = 8765
ABSENT_FRAMES = 10

GREETING = (
    "Welcome to the No Black Boxes workshop "
    "at the Global Innovation Forum! "
    "I'm NB3. Ask me a question!"
)

SYSTEM_PROMPT = (
    "You are NB3, a friendly robot at the No Black Boxes workshop. "
    "Answer questions in one short sentence, using at most 15 words. "
    "Be clear, friendly, and occasionally funny. "
    "Never invent facts."
    "If you don't know something direct people to the organisers" 
)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sys.path.insert(0, str(AUDIO_PYTHON))
    sys.path.insert(0, str(REPO_ROOT / "libs"))

    load_dotenv(AUDIO_PYTHON.parents[1] / ".env")

    eleven_key = os.environ.get("ELEVENLABS_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not eleven_key or not openai_key:
        raise RuntimeError("Missing ElevenLabs or OpenAI API key")

    stt = load_module(
        "nb3_stt",
        AUDIO_PYTHON / "01_stt_transcribe_question.py",
    )
    robot = load_module(
        "nb3_robot",
        AUDIO_PYTHON / "02_robot_reply.py",
    )

    from NB3.Vision.camera import Camera

    speech = Speech()
    camera = Camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

    stream = None
    worker = None

    # Shared conversation state.
    lock = threading.Lock()
    state = "ready"
    stable = 0
    absent = 0
    inference_failed = False

    def conversation():
        nonlocal state

        try:
            print(f"NB3: {GREETING}", flush=True)
            speech.say(GREETING)

            print("Listening...", flush=True)
            question = stt.transcribe_from_mic(eleven_key)

            if not question:
                speech.say(
                    "Sorry, I didn't catch that. "
                    "Please try again later."
                )
                return

            print(f"Visitor: {question}", flush=True)

            answer = robot.ask_llm(
                question,
                openai_key,
                SYSTEM_PROMPT,
            )

            if answer:
                print(f"NB3: {answer}", flush=True)
                speech.say(answer)

        except Exception:
            logging.getLogger("welcome").exception(
                "Conversation failed"
            )

        finally:
            with lock:
                state = "waiting_clear"

            print(
                "Conversation finished. Waiting for camera to clear.",
                flush=True,
            )

    def on_result(detections, sequence, timestamp, inference_ms):
        nonlocal state, stable, absent

        # Update live video overlays.
        stream.set_detections(
            detections, sequence, timestamp, inference_ms
        )

        person_visible = any(
            item.label == "person" for item in detections
        )

        start_conversation = False

        with lock:
            if state == "ready":
                stable = stable + 1 if person_visible else 0

                if stable >= STABLE_FRAMES:
                    state = "busy"
                    stable = 0
                    absent = 0
                    start_conversation = True

            elif state == "waiting_clear":
                absent = 0 if person_visible else absent + 1

                if absent >= ABSENT_FRAMES:
                    state = "ready"
                    absent = 0
                    stable = 0
                    print(
                        "Ready for next visitor.",
                        flush=True,
                    )

        if start_conversation:
            threading.Thread(
                target=conversation,
                daemon=True,
                name="nb3-conversation",
            ).start()

    try:
        camera.start()

        # Existing HTTP video server.
        stream = Stream(PORT, CAMERA_WIDTH, CAMERA_HEIGHT)

        print(
            f"Live video: http://{socket.gethostname()}.local:{PORT}",
            flush=True,
        )

        # Existing asynchronous Coral detection worker.
        worker = DetectorWorker(CONFIDENCE, on_result)

        print(
            "NB3-11 ready. Waiting for a visitor.",
            flush=True,
        )

        while True:
            # Camera capture and streaming continue independently
            # of greeting, listening and answering.
            jpeg = camera.capture(mjpeg=True)

            sequence, timestamp = stream.publish(jpeg)

            if worker.error:
                if not inference_failed:
                    logging.error(
                        "Coral detection failed: %s", worker.error
                    )
                    inference_failed = True
            else:
                worker.submit(jpeg, sequence, timestamp)

            time.sleep(0.0333)

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)

    finally:
        if worker is not None:
            worker.close()

        if stream is not None:
            stream.close()

        try:
            camera.stop()
        finally:
            camera.handle.close()


if __name__ == "__main__":
    main()
