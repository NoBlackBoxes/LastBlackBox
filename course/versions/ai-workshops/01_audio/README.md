# Audio : AI : ElevenLabs + OpenAI

This directory contains audio AI examples for the hackathon: simple scripts that turn speech into text, text into speech, and connect both through an LLM “robot”.

## Structure

- **`_resources/python/`** – Python scripts for the workshop
  - `00_tts_make_question_audio.py` – Text → speech (interactive question → creates question audio)
  - `01_stt_transcribe_question.py` – Speech → text (transcribes either the MP3 from `00` or a short microphone recording)
  - `02_robot_reply.py` – LLM reply + text → speech (robot answer)

## Quick Start

**⚠️ Setup Required**: Before running the scripts, make sure you have activated your Python environment (or set one up). Follow the [Virtual Environment Setup Guide](../../../../boxes/python/virtual_environments/README.md) if needed.

From `course/versions/ai-workshops/01_audio`:

```bash
pip install python-dotenv openai elevenlabs sounddevice soundfile
```

Create a `.env` file in this folder:

```bash
cd ~/NoBlackBoxes/LastBlackBox/course/versions/ai-workshops/01_audio
touch .env 
```

Edit the file to contain your API keys in the following format:

```env
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key
```

## Demo Scripts (run in order)

From `course/versions/ai-workshops/01_audio/_resources/python` run:

```bash
python 00_tts_make_question_audio.py
python 01_stt_transcribe_question.py
python 02_robot_reply.py
```

- `00` asks for a question on the command line and writes `my_00_question.mp3`
- `01` either:
  - reads `my_00_question.mp3` **or**
  - records ~5 seconds from the microphone (you choose at runtime),
  then writes `my_01_transcript.txt`
- `02` reads `my_01_transcript.txt`, asks for a **system prompt** to set the robot’s personality (or uses a friendly-teacher default), then writes:
  - `my_02_robot_reply.txt`
  - `my_02_robot_reply.mp3`

## Workshop Ideas

- **Personality challenge**:
  - When running `02_robot_reply.py`, type a different system prompt instead of using the default.
  - Examples: pirate robot, football coach robot, Shakespeare robot.
- **Voice challenge**:
  - Change `VOICE_ID` in `00_tts_make_question_audio.py` and `02_robot_reply.py`.
- **Prompt challenge**:
  - When running `00_tts_make_question_audio.py`, type different questions at the prompt.
- **Real voice challenge**:
  - In `01_stt_transcribe_question.py`, choose the microphone option to record a student asking the question, then re-run `02`.


## Notes

- Keep clips short to save ElevenLabs credits.
- If STT returns empty text, use clearer audio and less background noise.
- For reliability in class, stick to the numbered order `00 → 01 → 02`.
