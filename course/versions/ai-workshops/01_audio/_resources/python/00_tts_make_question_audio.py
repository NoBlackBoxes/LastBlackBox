"""
Purpose:
    Create a question WAV using ElevenLabs text-to-speech for use in step 01.
    Prompts for the question text (or uses default if Enter is pressed).

Execution Flow:
    main()
      └── prompt for question → text_to_speech (MP3) → convert to WAV

Side Effects:
    - Prompts user for question text (stdin)
    - Writes my_00_question.wav (MP3 is only an intermediate)
    - Calls ElevenLabs text-to-speech API
    - Uses ELEVENLABS_API_KEY from environment or .env

Inputs:
    - ELEVENLABS_API_KEY (environment variable or .env)
    - User input: question text (or Enter for default)

Outputs:
    - my_00_question.wav
"""

from pathlib import Path

from elevenlabs import ElevenLabs
from env_keys import load_keys
from nb3_config import VOICE_ID, TTS_MODEL_ID
from utils.nb3_audio import mp3_to_wav_for_nb3


# ------------------------------------------------------------------------------
# Edit this section to change the default question text
# (voice and model now come from nb3_config.py)
# ------------------------------------------------------------------------------

QUESTION_TEXT = "Hello NB3, what is your favorite neuroscience fact?"

# End of edit section ----------------------------------------------------------


def main() -> None:
    # Where is this file?
    script_dir = Path(__file__).resolve().parent
    
    # Load the API keys (this script only uses the ElevenLabs key)
    eleven_key, _ = load_keys()

    # User input
    user_text = input(
        f"Type your question (or press Enter to use default):\n[{QUESTION_TEXT}]\n> "
    ).strip()
    question_text = user_text or QUESTION_TEXT


    # Create the ElevenLabs client using the API key
    client = ElevenLabs(api_key=eleven_key)
    
    # Convert the text to speech using the ElevenLabs client
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=question_text,
        model_id=TTS_MODEL_ID,
    )

    # Convert the audio stream to bytes
    audio_data = b""
    for chunk in audio_stream:
        audio_data += chunk

    # Save the audio data to a temporary MP3 file
    out_mp3 = script_dir / "my_00_question.mp3"
    out_mp3.write_bytes(audio_data)

    # Convert MP3 → WAV for NB3 / GPIO playback, then remove the MP3
    out_wav = script_dir / "my_00_question.wav"
    mp3_to_wav_for_nb3(out_mp3, out_wav)
    try:
        out_mp3.unlink()
    except OSError:
        pass

    # Print the question text and the output file
    print(f"Used question: {question_text!r}")
    print(f"Saved: {out_wav}")
    print()
    print("Next: python 01_stt_transcribe_question.py")


if __name__ == "__main__":
    main()
