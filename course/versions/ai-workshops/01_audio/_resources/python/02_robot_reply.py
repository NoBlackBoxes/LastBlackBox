"""
Purpose:
    Turn the transcript from step 01 into a short LLM reply using OpenAI,
    then speak it with ElevenLabs text-to-speech. Final step of the pipeline.

Execution Flow:
    main()
      ├── read my_01_transcript.txt
      ├── ask_llm() → reply text
      └── text_to_speech() → my_02_robot_reply.mp3

Side Effects:
    - Reads my_01_transcript.txt
    - Writes my_02_robot_reply.txt and my_02_robot_reply.mp3
    - Calls OpenAI LLM API and ElevenLabs text-to-speech API
    - Uses ELEVENLABS_API_KEY and OPENAI_API_KEY from environment or .env

Inputs:
    - ELEVENLABS_API_KEY (environment variable or .env)
    - OPENAI_API_KEY (environment variable or .env)

Outputs:
    - my_02_robot_reply.txt
    - my_02_robot_reply.mp3
"""

import os
from pathlib import Path

import dotenv
from elevenlabs import ElevenLabs
from openai import OpenAI


VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
TTS_MODEL_ID = "eleven_turbo_v2_5"
LLM_MODEL = "gpt-4.1-mini"
SYSTEM_PROMPT = "You are a friendly robot teacher. Reply in one short sentence."


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


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workshop_root = script_dir.parent.parent
    dotenv.load_dotenv(workshop_root / ".env")

    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not eleven_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY in .env or environment.")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY in .env or environment.")

    in_file = script_dir / "my_01_transcript.txt"
    if not in_file.exists():
        raise SystemExit("Missing my_01_transcript.txt. Run: python 01_stt_transcribe_question.py")

    question = in_file.read_text(encoding="utf-8").strip()
    if not question:
        raise SystemExit("Transcript is empty. Re-run steps 00 and 01.")

    user_system = input(
        "Type a system prompt to change the robot personality\n"
        "(or press Enter to use the friendly teacher default):\n"
        f"[{SYSTEM_PROMPT}]\n> "
    ).strip()
    system_prompt = user_system or SYSTEM_PROMPT

    reply = ask_llm(question, openai_key, system_prompt)
    reply_txt = script_dir / "my_02_robot_reply.txt"
    reply_mp3 = script_dir / "my_02_robot_reply.mp3"

    reply_txt.write_text(reply + "\n", encoding="utf-8")
    text_to_speech(reply, eleven_key, reply_mp3)

    print(f"Question: {question}")
    print(f"Robot: {reply}")
    print(f"Saved: {reply_txt}")
    print(f"Saved: {reply_mp3}")
    print()
    print("Done. (End of pipeline: 00 → 01 → 02)")


if __name__ == "__main__":
    main()
