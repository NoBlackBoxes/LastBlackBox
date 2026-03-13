"""
Shared helper for loading API keys from the .env file.

All scripts in this workshop call load_keys() so there is
one simple place to change how keys are loaded.
"""

import os
from pathlib import Path

import dotenv


def load_keys() -> tuple[str, str]:
    """Load ELEVENLABS_API_KEY and OPENAI_API_KEY from the workshop .env."""
    # Where is this file?
    script_dir = Path(__file__).resolve().parent
    # Where is the workshop root?
    workshop_root = script_dir.parent.parent
    # Load the environment variables
    dotenv.load_dotenv(workshop_root / ".env")

    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not eleven_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY in .env or environment.")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY in .env or environment.")

    return eleven_key, openai_key


