"""
NB3 GPIO / I2S audio helpers.

These utilities are used by the demo script to play MP3 audio
through the NB3 amplifier + speaker connected on the GPIO pins.

They keep all the ffmpeg / aplay details out of the main workshop
scripts (00, 01, 02, demo).
"""

import shutil
import subprocess
from pathlib import Path

# ALSA device for the NB3 I2S amplifier (see testing-audio docs)
NB3_DEVICE = "plughw:3"
NB3_RATE = 48000


def _have_tools() -> bool:
    """Return True if both ffmpeg and aplay are available."""
    return shutil.which("ffmpeg") is not None and shutil.which("aplay") is not None


def record_from_nb3(seconds: float) -> bytes | None:
    """
    Record from the NB3 GPIO / I2S input device using arecord.
    Returns WAV bytes, or None if arecord is not available.
    Uses the same device and format as the Pi playback (plughw:3, 48 kHz, 2 ch).
    """
    arecord = shutil.which("arecord")
    if not arecord:
        return None
    result = subprocess.run(
        [
            arecord,
            "-D",
            NB3_DEVICE,
            "-c2",
            "-r",
            str(NB3_RATE),
            "-f",
            "S32_LE",
            "-t",
            "wav",
            "-V",
            "stereo",
            "-d",
            str(int(seconds)),
            "-",
        ],
        capture_output=True,
        timeout=int(seconds) + 10,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def mp3_to_wav_for_nb3(mp3_path: Path, wav_path: Path) -> None:
    """
    Convert an MP3 file to a WAV file suitable for playback on NB3.

    The WAV will be 48 kHz, 2 channels. The exact internal sample
    format is handled by ffmpeg and ALSA.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp3_path),
            "-ar",
            str(NB3_RATE),
            "-ac",
            "2",
            str(wav_path),
        ],
        check=False,
    )


def play_wav_on_nb3(wav_path: Path) -> None:
    """
    Play a WAV file through the NB3 I2S / GPIO device using aplay.
    """
    if shutil.which("aplay") is None:
        return

    subprocess.run(
        [
            "aplay",
            "-D",
            NB3_DEVICE,
            "-c2",
            "-r",
            str(NB3_RATE),
            "-f",
            "S32_LE",
            "-t",
            "wav",
            "-V",
            "stereo",
            "-v",
            str(wav_path),
        ],
        check=False,
    )


def play_mp3_on_nb3(mp3_path: Path) -> None:
    """
    Convert an MP3 file to WAV and play it on NB3 GPIO audio.

    If ffmpeg or aplay are missing, this function does nothing.
    The MP3 file is treated as an intermediate and removed after
    conversion; the WAV file is kept.
    """
    if not _have_tools():
        return

    wav_path = mp3_path.with_suffix(".wav")
    mp3_to_wav_for_nb3(mp3_path, wav_path)
    play_wav_on_nb3(wav_path)
    try:
        mp3_path.unlink()
    except OSError:
        pass

