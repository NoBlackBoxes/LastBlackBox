"""
Purpose:
    Convert a question (from step 00 MP3 or from microphone) into text using
    ElevenLabs speech-to-text. Prompts to choose input source (file, mic, or quit).

Execution Flow:
    main()
      ├── [optional] record_from_mic() → my_01_recorded.wav
      └── transcribe(audio_path) → text

Side Effects:
    - Prompts user: [a] file, [b] mic, [q] quit
    - Reads my_00_question.mp3 (option a) or records to my_01_recorded.wav (option b)
    - Writes my_01_transcript.txt
    - Calls ElevenLabs speech-to-text API
    - Uses ELEVENLABS_API_KEY from environment or .env

Inputs:
    - ELEVENLABS_API_KEY (environment variable or .env)
    - User input: a, b, or q

Outputs:
    - my_01_transcript.txt
    - my_01_recorded.wav (only when option b: record from microphone)
"""

import os
from pathlib import Path

import dotenv
from elevenlabs import ElevenLabs
import sounddevice as sd
import soundfile as sf

RECORD_SECONDS = 5
SAMPLE_RATE = 44100


def transcribe(audio_path: Path, api_key: str) -> str:
    client = ElevenLabs(api_key=api_key)
    with audio_path.open("rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
        )

    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    return (text or "").strip()


def record_from_mic(out_path: Path) -> Path:
    print(f"Recording {RECORD_SECONDS} seconds... speak now.")
    recording = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(out_path, recording, SAMPLE_RATE)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workshop_root = script_dir.parent.parent
    dotenv.load_dotenv(workshop_root / ".env")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY in .env or environment.")

    choice = (
        input(
            "Choose input:\n"
            "  [a] Use prerecorded question from 00_tts_make_question_audio.py\n"
            "  [b] Record a new question with your microphone\n"
            "  [q] Quit\n"
            "Enter a, b, or q [a]: "
        )
        .strip()
        .lower()
        or "a"
    )

    if choice == "q":
        print("Goodbye.")
        return
    elif choice == "b":
        wav_path = script_dir / "my_01_recorded.wav"
        record_from_mic(wav_path)
        audio_path = wav_path
    else:
        audio_path = script_dir / "my_00_question.mp3"
        if not audio_path.exists():
            raise SystemExit(
                "Missing my_00_question.mp3. Run: python 00_tts_make_question_audio.py"
            )

    transcript = transcribe(audio_path, api_key)
    out_file = script_dir / "my_01_transcript.txt"
    out_file.write_text(transcript + "\n", encoding="utf-8")

    print(f"Transcript: {transcript if transcript else '(empty)'}")
    print(f"Saved: {out_file}")
    print()
    print("Next: python 02_robot_reply.py")


if __name__ == "__main__":
    main()
