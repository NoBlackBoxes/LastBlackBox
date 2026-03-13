"""
Purpose:
    Convert a question into text using ElevenLabs speech-to-text.
    Can use either:
      - the WAV created in step 00, or
      - a short microphone recording.

Execution Flow:
    main()
      ├── optional: record_from_mic() → my_01_recorded.wav
      ├── transcribe(audio_path) → text
      └── write my_01_transcript.txt

    transcribe_from_mic()
      └── record from microphone → in-memory WAV → text (no files)
         (used by demo.py for the live NB3 demo)

Side Effects (main only):
    - Prompts user: [a] file, [b] mic, [q] quit
    - Reads my_00_question.wav (option a) or records to my_01_recorded.wav (option b)
    - Writes my_01_transcript.txt
    - Calls ElevenLabs speech-to-text API

Inputs:
    - ELEVENLABS_API_KEY (from .env via env_keys.load_keys)
    - User input: a, b, or q (for main())

Outputs (main only):
    - my_01_transcript.txt
    - my_01_recorded.wav (only when option b: record from microphone)
"""

from pathlib import Path
import io

from elevenlabs import ElevenLabs
import sounddevice as sd
import soundfile as sf
from env_keys import load_keys
from utils.nb3_audio import record_from_nb3

# Constants
RECORD_SECONDS = 5
SAMPLE_RATE = 44100

def transcribe(audio_path: Path, api_key: str) -> str:
    """Send an audio file to ElevenLabs speech-to-text and return the transcript."""
    client = ElevenLabs(api_key=api_key)
    with audio_path.open("rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            language_code="en",  # avoid locale-dependent tags (e.g. Hungarian)
            tag_audio_events=False,  # no (laughter), (pause), etc. in transcript
        )

    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    return (text or "").strip()


def transcribe_from_mic(api_key: str) -> str:
    """
    Record from the microphone, send the audio to ElevenLabs speech-to-text,
    and return the transcribed text. Uses NB3 GPIO input (arecord) when
    available, otherwise sounddevice default. No files written (for demo).
    """
    print(f"Recording {RECORD_SECONDS} seconds... speak now.")
    wav_bytes = record_from_nb3(RECORD_SECONDS)
    if wav_bytes:
        buffer = io.BytesIO(wav_bytes)
    else:
        recording = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        buffer = io.BytesIO()
        sf.write(buffer, recording, SAMPLE_RATE, format="WAV")
        buffer.seek(0)

    print("Transcribing your question with ElevenLabs...")
    buffer.seek(0)
    client = ElevenLabs(api_key=api_key)
    result = client.speech_to_text.convert(
        file=buffer,
        model_id="scribe_v1",
        language_code="en",  # avoid locale-dependent tags (e.g. Hungarian)
        tag_audio_events=False,  # no (laughter), (pause), etc. in transcript
    )

    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    return (text or "").strip()

def record_from_mic(out_path: Path) -> Path:
    """
    Record from the microphone and save to a WAV file.
    Uses NB3 GPIO (arecord) when available, otherwise sounddevice default.
    """
    print(f"Recording {RECORD_SECONDS} seconds... speak now.")
    wav_bytes = record_from_nb3(RECORD_SECONDS)
    if wav_bytes:
        out_path.write_bytes(wav_bytes)
    else:
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
    # Where is this file?
    script_dir = Path(__file__).resolve().parent

    # Load the API keys (this script only uses the ElevenLabs key)
    eleven_key, _ = load_keys()

    # User input
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
        audio_path = script_dir / "my_00_question.wav"
        if not audio_path.exists():
            raise SystemExit(
                "Missing my_00_question.wav. Run: python 00_tts_make_question_audio.py"
            )

    # Transcribe the audio
    transcript = transcribe(audio_path, eleven_key)
    # Save the transcript to a file
    out_file = script_dir / "my_01_transcript.txt"
    out_file.write_text(transcript + "\n", encoding="utf-8")

    # Print the transcript and the output file
    print(f"Transcript: {transcript if transcript else '(empty)'}")
    print(f"Saved: {out_file}")
    print()
    print("Next: python 02_robot_reply.py")


if __name__ == "__main__":
    main()
