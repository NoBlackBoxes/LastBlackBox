# Audio : AI (ElevenLabs + OpenAI)

Simple workshop: speech → text (STT), text → speech (TTS), and an LLM in between. Run the pipeline step-by-step (00 → 01 → 02) or use the one-shot demo.

---

## 1. Setup

The workshop environment is already active (`_tmp/LBB` or similar). Do this once:

1. **Git pull** at the repo root (wherever your clone is, e.g. `NoBlackBoxes` or `LastBlackBox`):
   ```bash
   cd /path/to/your/repo/root
   git pull
   ```
2. **Install requirements** and add your **API keys** (below).

### 1.1 System tools (required for conversion)

Scripts 00 and 02 convert ElevenLabs' MP3 output to WAV using **ffmpeg**. You need it for the pipeline and demo.

| Tool        | Used for                                      | Install (examples)                    |
|-------------|-----------------------------------------------|---------------------------------------|
| **ffmpeg**  | MP3 to WAV conversion (00, 02, demo)          | `apt install ffmpeg` (Debian/Pi), `brew install ffmpeg` (macOS) |
| **ffplay**  | Laptop playback (demo; comes with ffmpeg)     | Same as ffmpeg                        |
| **aplay**   | Pi GPIO playback (optional)                   | Usually with ALSA on Pi               |
| **arecord** | Pi GPIO recording (optional)                  | Usually with ALSA on Pi               |

### 1.2 Python packages

From repo root:

```bash
cd course/versions/ai-workshops/01_audio
pip install -r requirements.txt
```

Or install manually:

```bash
cd course/versions/ai-workshops/01_audio
pip install python-dotenv openai elevenlabs sounddevice soundfile
```

| Package        | Used for                          |
|----------------|-----------------------------------|
| python-dotenv  | Load API keys from `.env`         |
| openai         | LLM (script 02, demo)             |
| elevenlabs     | Speech-to-text and text-to-speech |
| sounddevice    | Microphone recording (script 01)  |
| soundfile      | WAV handling (script 01)         |

### 1.3 API keys

Create a `.env` file in the workshop folder. From repo root:

```bash
cd course/versions/ai-workshops/01_audio
touch .env
```

Edit `.env` and add (replace with your real keys):

```env
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key
```

Get keys from [ElevenLabs](https://elevenlabs.io/) and [OpenAI](https://platform.openai.com/).

---

## 2. What’s in this folder

| File / folder | Role |
|---------------|------|
| `_resources/python/00_tts_make_question_audio.py` | Type a question → WAV file (TTS) |
| `_resources/python/01_stt_transcribe_question.py` | WAV or mic → transcript (STT) |
| `_resources/python/02_robot_reply.py` | Transcript → LLM reply → WAV (no playback) |
| `_resources/python/demo.py` | Full demo: intro → listen → reply (no saved files) |
| `_resources/python/env_keys.py` | Loads API keys from `.env` |
| `_resources/python/nb3_config.py` | Voice, TTS model, LLM model, system prompt |
| `_resources/python/utils/nb3_audio.py` | Pi GPIO: record, MP3→WAV, play |

---

## 3. Run the pipeline (00 → 01 → 02)

From repo root:

```bash
cd course/versions/ai-workshops/01_audio/_resources/python
python 00_tts_make_question_audio.py
python 01_stt_transcribe_question.py
python 02_robot_reply.py
```

- **00** – Asks for a question, creates `my_00_question.wav`.
- **01** – Choose: (a) use `my_00_question.wav`, or (b) record from the mic. Writes `my_01_transcript.txt`.
- **02** – Reads the transcript, asks for a system prompt (or Enter for default), writes `my_02_robot_reply.txt` and `my_02_robot_reply.wav`. It does **not** play the reply; you play the WAV yourself.

### Playing the WAV

- **Laptop:** Open `my_02_robot_reply.wav` (or `my_00_question.wav`) in your player.
- **Raspberry Pi (GPIO / I2S):** Use the same device as for playback. From repo root:

```bash
cd course/versions/ai-workshops/01_audio/_resources/python
aplay -q -D plughw:3 -c2 -r 48000 -f S32_LE -t wav -V stereo my_02_robot_reply.wav
```

Use the same command for `my_00_question.wav` or any WAV in this folder.

**Recording on the Pi:** If `arecord` is available, script 01 and the demo record from the same GPIO device (`plughw:3`), so mic and speaker match.

---

## 4. Live demo (all-in-one)

From repo root:

```bash
cd course/versions/ai-workshops/01_audio/_resources/python
python demo.py
```

- Says: “Hey I’m NB3, ask me anything.”
- Records your question from the mic (Pi: GPIO if `arecord` is available).
- Transcribes with ElevenLabs, sends to the LLM, then speaks the reply.
- Does **not** save transcript or reply files; any temp audio is removed after use.

On a laptop you need **ffplay** (ffmpeg) to hear the robot. On the Pi, playback uses the GPIO device if **ffmpeg** and **aplay** are available.

---

## 5. Workshop ideas (get creative)

This is where you change the robot’s personality, voice, and questions. Here’s what to tweak and where.

### Personality (how the robot “thinks” and replies)

- **What to change:** The **system prompt** that tells the LLM how to behave.
- **Where:** When you run **02_robot_reply.py**, you’re asked for a system prompt; you can type one there. The default is in **`nb3_config.py`** → `SYSTEM_PROMPT`.
- **Ideas:** Try “You are a pirate. Reply in one short sentence.” / “You are a strict football coach.” / “You are Shakespeare. Reply in one short line of verse.” / “You are a snob who loves literature.”

### Voice (how the robot sounds)

- **What to change:** Which ElevenLabs **voice** and **model** are used for speech.
- **Where:** **`_resources/python/nb3_config.py`** — edit `VOICE_ID` and optionally `TTS_MODEL_ID`.
- **Finding voices:** Open the [ElevenLabs Voice Library](https://elevenlabs.io/voice-library). Pick a voice you like; its **Voice ID** is in the URL or in the voice settings. Paste that ID into `VOICE_ID` in `nb3_config.py`. The same file is used by script 00, 02, and the demo, so the voice changes everywhere.

### Question (what gets spoken or transcribed)

- **What to change:** The question text (for the pipeline: what 00 speaks, or what 01 transcribes from a file).
- **Where:** When you run **00_tts_make_question_audio.py**, you can type a new question at the prompt. To change the **default** question, edit **`00_tts_make_question_audio.py`** and look for `QUESTION_TEXT` near the top; change that string.

### Real voice (you ask the question)

- **What to do:** Use your own voice instead of the pre-recorded question.
- **Where:** Run **01_stt_transcribe_question.py**, choose option **[b]** to record from the microphone. Ask your question when it says “speak now”. Then run **02_robot_reply.py** as usual and play the reply WAV (e.g. copy the `aplay` command that 02 prints).

---

## 6. Notes

- Keep clips short to save ElevenLabs usage.
- If transcription is empty, use clearer audio and less background noise.
- Run the pipeline in order: 00 → 01 → 02.
