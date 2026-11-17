"""Reuse workshop TTS and WAV conversion; route checked playback to NB3."""
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave

from config import AUDIO_PYTHON, ALSA_DEVICE


class Speech:
    def __init__(self, env_file=None):
        for tool in ("ffmpeg", "aplay"):
            if shutil.which(tool) is None:
                raise RuntimeError(f"Required playback tool is missing: {tool}")
        sys.path.insert(0, str(AUDIO_PYTHON))
        import dotenv

        credentials = env_file or os.environ.get("NB3_ENV_FILE") or AUDIO_PYTHON.parents[1] / ".env"
        dotenv.load_dotenv(credentials)
        self.key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.key:
            raise RuntimeError("Missing ELEVENLABS_API_KEY. Set it in the environment, 01_audio/.env, or pass --env-file PATH. Do not put the key in a command argument.")
        # Reuse workshop voice settings/conversion without importing its LLM.
        from elevenlabs import ElevenLabs
        from nb3_config import VOICE_ID, TTS_MODEL_ID
        from utils.nb3_audio import mp3_to_wav_for_nb3
        self.client = ElevenLabs(api_key=self.key)
        self.voice, self.model = VOICE_ID, TTS_MODEL_ID
        self.convert = mp3_to_wav_for_nb3
        self.logger = logging.getLogger("demo3.speech")
        self.logger.info("Speaker: %s; ElevenLabs configured", ALSA_DEVICE)

    def say(self, text):
        with TemporaryDirectory(prefix="nb3-demo3-") as directory:
            mp3 = Path(directory) / "speech.mp3"
            wav = mp3.with_suffix(".wav")
            # Exactly one ElevenLabs request per trigger; do not call speak().
            audio = self.client.text_to_speech.convert(voice_id=self.voice, model_id=self.model,
                                                       text=text, output_format="mp3_44100_128")
            with mp3.open("wb") as output:
                for chunk in audio:
                    output.write(chunk)
            self.logger.info("ElevenLabs synthesis completed (%d bytes)", mp3.stat().st_size)
            self.convert(mp3, wav)
            # Existing converter suppresses ffmpeg errors; validate its output.
            with wave.open(str(wav), "rb") as audio:
                if audio.getframerate() != 48000 or audio.getnchannels() != 2 or audio.getnframes() == 0:
                    raise RuntimeError("Audio conversion did not produce 48 kHz stereo WAV")
            # WAV header supplies the actual PCM format; do not force S32_LE
            # over the existing converter's default PCM16 output.
            result = subprocess.run(["aplay", "-q", "-D", ALSA_DEVICE, str(wav)],
                                    capture_output=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f"ALSA playback failed on {ALSA_DEVICE}: {result.stderr.decode(errors='replace').strip()}")
            self.logger.info("ALSA playback completed on %s", ALSA_DEVICE)
