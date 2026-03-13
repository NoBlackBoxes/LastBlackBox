"""
Purpose:
    Turn the transcript from step 01 into a short LLM reply using OpenAI,
    then speak it with ElevenLabs text-to-speech. Final step of the
    file-based 00 → 01 → 02 pipeline, and also the voice brain for demo.py.

Execution Flow (main):
    main()
      ├── read my_01_transcript.txt
      ├── ask_llm() → reply text
      ├── text_to_speech() → my_02_robot_reply.mp3 (intermediate)
      ├── convert MP3 → my_02_robot_reply.wav
      └── print and save the reply

Helpers:
    ask_llm()
      - Sends the question + system prompt to the OpenAI LLM.

    text_to_speech()
      - Turns reply text into an MP3 file using ElevenLabs.

    speak()
      - Turns reply text into live audio using ElevenLabs + ffplay.
      - Used by demo.py so the intro and robot reply share these settings.

Side Effects (main only):
    - Reads my_01_transcript.txt
    - Writes my_02_robot_reply.txt and my_02_robot_reply.wav
    - Calls OpenAI LLM API and ElevenLabs text-to-speech API

Inputs:
    - ELEVENLABS_API_KEY (from .env via env_keys.load_keys)
    - OPENAI_API_KEY (from .env via env_keys.load_keys)

Outputs (main only):
    - my_02_robot_reply.txt
    - my_02_robot_reply.wav
"""

from pathlib import Path
import shutil
import subprocess

from elevenlabs import ElevenLabs
from openai import OpenAI
from env_keys import load_keys
from nb3_config import VOICE_ID, TTS_MODEL_ID, LLM_MODEL, SYSTEM_PROMPT
from utils.nb3_audio import mp3_to_wav_for_nb3

# Define the ask_llm function
def ask_llm(user_text: str, openai_key: str, system_prompt: str) -> str:
    client = OpenAI(api_key=openai_key)
    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return response.output_text.strip()

# Define the text_to_speech function
def text_to_speech(text: str, eleven_key: str, out_file: Path) -> None:
    client = ElevenLabs(api_key=eleven_key)
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=TTS_MODEL_ID,
    )

    audio_data = b""
    for chunk in audio_stream:
        audio_data += chunk

    out_file.write_bytes(audio_data)


def speak(text: str, eleven_key: str) -> None:
    """
    Generate speech with ElevenLabs and play it via ffplay if available.

    Used by the demo script so that both the introduction and the robot
    reply use the same voice settings as this file.
    """
    if not text:
        return

    client = ElevenLabs(api_key=eleven_key)
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=TTS_MODEL_ID,
    )

    audio_data = b""
    for chunk in audio_stream:
        audio_data += chunk

    ffplay = shutil.which("ffplay")
    if not ffplay:
        print("(Audio generated, but ffplay (ffmpeg) is not installed;")
        print(" please install ffmpeg to hear the robot speak.)")
        return

    subprocess.run(
        [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-",
        ],
        input=audio_data,
        check=False,
    )

# Define the main function that calls the ask_llm function and text_to_speech function
def main() -> None:
    # Where is this file?
    script_dir = Path(__file__).resolve().parent

    # Load the environment variables and API keys
    eleven_key, openai_key = load_keys()

    # Load the transcript from the file
    in_file = script_dir / "my_01_transcript.txt"
    if not in_file.exists():
        raise SystemExit("Missing my_01_transcript.txt. Run: python 01_stt_transcribe_question.py")

    # Read the question from the file
    question = in_file.read_text(encoding="utf-8").strip()
    if not question:
        raise SystemExit("Transcript is empty. Re-run steps 00 and 01.")

    # User input
    user_system = input(
        "Type a system prompt to change the robot personality\n"
        "(or press Enter to use the friendly teacher default):\n"
        f"[{SYSTEM_PROMPT}]\n> "
    ).strip()
    system_prompt = user_system or SYSTEM_PROMPT

    # Ask the LLM
    reply = ask_llm(question, openai_key, system_prompt)
    # Save the reply to a text file
    reply_txt = script_dir / "my_02_robot_reply.txt"
    # Save the reply to audio files (MP3 intermediate, WAV kept)
    reply_mp3 = script_dir / "my_02_robot_reply.mp3"
    reply_wav = script_dir / "my_02_robot_reply.wav"

    # Save the reply to a text file
    reply_txt.write_text(reply + "\n", encoding="utf-8")
    # Save the reply to an audio file (MP3 from ElevenLabs)
    text_to_speech(reply, eleven_key, reply_mp3)
    # Convert MP3 → WAV for NB3 / GPIO playback, then remove the MP3
    mp3_to_wav_for_nb3(reply_mp3, reply_wav)
    try:
        reply_mp3.unlink()
    except OSError:
        pass

    # Print the question, reply, and the output files
    print(f"Question: {question}")
    print(f"Robot: {reply}")
    print(f"Saved: {reply_txt}")
    print(f"Saved: {reply_wav}")
    print()
    print("Done. (End of pipeline: 00 → 01 → 02)")


if __name__ == "__main__":
    main()
