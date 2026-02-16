# ElevenLabs API Test

This script tests the ElevenLabs API and helps understand credit usage for text-to-speech conversions.

**ElevenLabs** is a text-to-speech (TTS) service that converts written text into natural-sounding speech using AI. This test script demonstrates how to use their API to convert text to speech and track credit consumption.

## 📚 API Documentation

- **Main API Reference**: https://elevenlabs.io/docs/api-reference/introduction
- **Text-to-Speech Endpoint**: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
- **Python SDK**: https://github.com/elevenlabs/elevenlabs-python
- **Available Models**: https://elevenlabs.io/docs/api-reference/models

## Setup

### 1. Install Required Packages

```bash
pip install elevenlabs python-dotenv
```

### 2. Set Up API Key

You have two options:

**Option A: Create a `.env` file** (recommended)
Create a `.env` file in the workshop folder (`01_audio/`):
```
course/versions/ai-workshops/01_audio/.env
```

The file should contain:
```
ELEVENLABS_API_KEY=your_api_key_here
```

**Option B: Set Environment Variable**
```bash
export ELEVENLABS_API_KEY='your_api_key_here'
```

### 3. Run the Test

From the `01_audio/` folder:
```bash
python _resources/python/elevenlabs_api_test.py
```

## Basic Usage

### Simple Text-to-Speech Example

Here's a minimal example of how to use the ElevenLabs API:

```python
from elevenlabs import ElevenLabs

# Initialize client with your API key
client = ElevenLabs(api_key="your_api_key_here")

# Convert text to speech
audio_stream = client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Voice ID (Rachel)
    text="Hello, this is a test.",
    model_id="eleven_turbo_v2_5"     # Model: turbo (fast) or standard (high quality)
)

# Collect audio chunks
audio_data = b""
for chunk in audio_stream:
    audio_data += chunk

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio_data)
```

### Key Concepts

1. **Voice ID**: Each voice has a unique ID. You can:
   - Use pre-made voices (like "Rachel": `21m00Tcm4TlvDq8ikWAM`)
   - List available voices: `client.voices.search()`
   - Create custom voices (requires additional permissions)

2. **Models**: Choose based on speed vs quality:
   - `eleven_turbo_v2_5` - Fast and cheaper (~$0.06 per 1K chars)
   - `eleven_multilingual_v2` - Higher quality (~$0.12 per 1K chars)
   - `eleven_monolingual_v1` - English only, high quality

3. **Voice Settings**: Fine-tune the output:
   - `stability` (0.0-1.0): How consistent the voice is
   - `similarity_boost` (0.0-1.0): How similar to the original voice
   - `style` (0.0-1.0): Style exaggeration
   - `use_speaker_boost`: Enhance clarity

4. **Credits**: Charged per character in your input text
   - Counts every character including spaces and punctuation
   - Different models have different costs
   - Check your usage: `client.user.get()` (if you have permission)

### Common API Methods

```python
# Get user info (subscription, credits)
user_info = client.user.get()

# List available voices
voices = client.voices.search()

# Convert text to speech
audio = client.text_to_speech.convert(
    voice_id="voice_id",
    text="Your text here",
    model_id="eleven_turbo_v2_5"
)

# Get voice by ID
voice = client.voices.get(voice_id="voice_id")
```

## What the Script Does

1. **Tests API Connection** - Verifies the API key works
2. **Gets User Info** - Shows subscription tier and credit limits
3. **Lists Available Voices** - Shows voices you can use
4. **Tests Text-to-Speech** - Converts text to speech with different lengths:
   - Short test (~25 characters)
   - Medium test (~150 characters)
   - Long test (~200+ characters)
5. **Tracks Credit Usage** - Shows how many characters/credits each conversion uses
6. **Saves Audio Files** - Creates MP3 files you can listen to verify quality

## Understanding Credits

- **Credits are based on character count** - Each character in your text consumes credits
- **Turbo/Flash models**: ~$0.06 per 1,000 characters (cheaper, faster)
- **Standard models**: ~$0.12 per 1,000 characters (higher quality)
- **1 credit ≈ 1 character** (varies slightly by model)

### Credit Estimation Examples:
- Short message (50 chars): ~50 credits
- Medium message (200 chars): ~200 credits  
- Long message (1000 chars): ~1000 credits

## Output Files

The script creates test audio files in the workshop folder (`01_audio/`):
- `my_short_test.mp3`
- `my_medium_test.mp3`
- `my_long_test.mp3`

## Notes

- The script uses the `eleven_turbo_v2_5` model (cheaper option)
- Current limit: 500 credits (can be increased)
- Unused credits roll over to the next month (up to 2 months worth)
- Some API keys may have limited permissions (e.g., only TTS conversion, no user/voice read access)

## Troubleshooting

**"Missing permissions" error**: Your API key may only have permission for text-to-speech conversion. This is normal and TTS will still work.

**"Voice not found" error**: Make sure you're using a valid voice_id. Try the default voice ID: `21m00Tcm4TlvDq8ikWAM`

**Audio file is empty**: Make sure you're collecting all chunks from the generator:
```python
audio_data = b""
for chunk in audio_stream:
    audio_data += chunk
```

## Additional Resources

- [ElevenLabs Pricing](https://elevenlabs.io/pricing)
- [Voice Library](https://elevenlabs.io/voice-library)
- [API Status](https://status.elevenlabs.io/)
