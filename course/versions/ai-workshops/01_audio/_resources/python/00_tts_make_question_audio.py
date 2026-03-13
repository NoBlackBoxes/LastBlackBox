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


# ------------------------------------------------------------------------------
# Edit this section to change the voice and model
# ------------------------------------------------------------------------------

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
MODEL_ID = "eleven_turbo_v2_5"
QUESTION_TEXT = "Hello NB3, what is your favorite neuroscience fact?"

# End of edit section ----------------------------------------------------------


def main() -> None:
    # Where is this file?
    script_dir = Path(__file__).resolve().parent
    # Where is the workshop root?
    workshop_root = script_dir.parent.parent
    # Load the environment variables
    dotenv.load_dotenv(workshop_root / ".env")

    # Get the API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY in .env or environment.")

    # User input
    user_text = input(
        f"Type your question (or press Enter to use default):\n[{QUESTION_TEXT}]\n> "
    ).strip()
    question_text = user_text or QUESTION_TEXT


    # Create the ElevenLabs client using the API key
    client = ElevenLabs(api_key=api_key)
    
    # Convert the text to speech using the ElevenLabs client
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=question_text,
        model_id=MODEL_ID,
    )

    # Convert the audio stream to bytes
    audio_data = b""
    for chunk in audio_stream:
        audio_data += chunk

    # Save the audio data to a file
    out_file = script_dir / "my_00_question.mp3"
    out_file.write_bytes(audio_data)

    # Print the question text and the output file
    print(f"Used question: {question_text!r}")
    print(f"Saved: {out_file}")
    print()
    print("Next: python 01_stt_transcribe_question.py")


if __name__ == "__main__":
    main()
