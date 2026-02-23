"""
Purpose:
    Create a question MP3 using ElevenLabs text-to-speech for use in step 01.
    Prompts for the question text (or uses default if Enter is pressed).

Execution Flow:
    main()
      └── prompt for question → text_to_speech → write MP3

Side Effects:
    - Prompts user for question text (stdin)
    - Writes my_00_question.mp3
    - Calls ElevenLabs text-to-speech API
    - Uses ELEVENLABS_API_KEY from environment or .env

Inputs:
    - ELEVENLABS_API_KEY (environment variable or .env)
    - User input: question text (or Enter for default)

Outputs:
    - my_00_question.mp3
"""

import os
from pathlib import Path

import dotenv
from elevenlabs import ElevenLabs


VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
MODEL_ID = "eleven_turbo_v2_5"
QUESTION_TEXT = "Hello NB3, what is your favorite neuroscience fact?"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workshop_root = script_dir.parent.parent
    dotenv.load_dotenv(workshop_root / ".env")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY in .env or environment.")

    user_text = input(
        f"Type your question (or press Enter to use default):\n[{QUESTION_TEXT}]\n> "
    ).strip()
    question_text = user_text or QUESTION_TEXT

    client = ElevenLabs(api_key=api_key)
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=question_text,
        model_id=MODEL_ID,
    )

    audio_data = b""
    for chunk in audio_stream:
        audio_data += chunk

    out_file = script_dir / "my_00_question.mp3"
    out_file.write_bytes(audio_data)
    print(f"Used question: {question_text!r}")
    print(f"Saved: {out_file}")
    print()
    print("Next: python 01_stt_transcribe_question.py")


if __name__ == "__main__":
    main()
