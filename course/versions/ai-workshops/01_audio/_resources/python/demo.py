"""
Purpose:
    Live NB3 demo that:
      1) Speaks: "Hey I'm NB3, ask me anything"
      2) Listens to a microphone question
      3) Transcribes it with ElevenLabs speech-to-text
      4) Sends it to the OpenAI LLM used in 02_robot_reply.py
      5) Speaks the robot's reply

Notes:
    - This script reuses helper functions and settings from:
        * 01_stt_transcribe_question.py (RECORD_SECONDS, SAMPLE_RATE)
        * 02_robot_reply.py (VOICE_ID, TTS_MODEL_ID, SYSTEM_PROMPT)
    - It does NOT write any transcript, reply, or microphone audio files;
      everything stays in memory for the demo.
"""

import importlib.util
from pathlib import Path

from env_keys import load_keys

# define the _load_module function
def _load_module(name: str, path: Path):
    """Dynamically load a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def main() -> None:
    # Where is this file?
    script_dir = Path(__file__).resolve().parent

    # Load environment variables and API keys (same helper as other scripts)
    eleven_key, openai_key = load_keys()

    # Dynamically import the existing scripts so we can reuse their settings/logic.
    stt_path = script_dir / "01_stt_transcribe_question.py"
    robot_path = script_dir / "02_robot_reply.py"
    stt_mod = _load_module("nb3_stt", stt_path)
    robot_mod = _load_module("nb3_robot", robot_path)

    # Speak the intro line
    greeting = "Hey I'm NB3, ask me anything."
    print(f"NB3: {greeting}")
    print()
    robot_mod.speak(greeting, eleven_key)

    # 1) Record a question from the microphone and transcribe it (no files).
    question = stt_mod.transcribe_from_mic(eleven_key)
    if not question:
        raise SystemExit("Transcript is empty. Please try asking again.")

    print()
    print(f"You asked: {question}")
    print()

    # 3) Ask the LLM using the same helper and default system prompt
    #    from 02_robot_reply.py.
    system_prompt = getattr(robot_mod, "SYSTEM_PROMPT", "You are a helpful assistant.")
    reply = robot_mod.ask_llm(question, openai_key, system_prompt).strip()

    # 4) Speak (and also print) the reply (no reply text/audio files written).
    print(f"NB3: {reply}")
    robot_mod.speak(reply, eleven_key)
    print()
    print("Done. (demo.py: mic → STT → LLM, no files saved.)")


if __name__ == "__main__":
    main()

