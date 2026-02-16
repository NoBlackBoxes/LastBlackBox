"""
Test script for ElevenLabs API
Tests basic functionality and checks credit usage for text-to-speech

ElevenLabs is a text-to-speech API service that converts written text into
natural-sounding speech. This script demonstrates:
- How to authenticate with the API
- How to check your account status and credits
- How to list available voices
- How to convert text to speech
- How to track credit usage

API Documentation: https://elevenlabs.io/docs/api-reference/introduction
"""
import os
import dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from elevenlabs.core.api_error import ApiError


def test_elevenlabs_api():
    """
    Main test function that demonstrates ElevenLabs API usage.
    
    Order of operations:
    1. Credit status (check balance first, before using any credits)
    2. Voice listing (available voices for TTS)
    3. Text-to-speech conversion (with credit tracking)
    """
    print("=" * 60)
    print("Testing ElevenLabs API")
    print("=" * 60)
    
    # ========================================================================
    # STEP 1: Load API Key from Environment
    # ========================================================================
    # ElevenLabs requires an API key for authentication. The key can be stored
    # in a .env file (recommended) or as an environment variable.
    # Script lives in 01_audio/_resources/python/; .env lives in workshop root (01_audio/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workshop_root = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(workshop_root, ".env")
    
    # Try to load .env file if it exists
    if os.path.exists(env_path):
        dotenv.load_dotenv(env_path)
        print(f"✅ Loaded .env file from: {env_path}")
    
    # Retrieve the API key from environment variables
    # The key is set either from .env file or system environment
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("\n⚠️  ELEVENLABS_API_KEY not found!")
        print("Please either:")
        print("  1. Set environment variable: export ELEVENLABS_API_KEY='your_key_here'")
        print(f"  2. Create a .env file in: {workshop_root}")
        print("     with content: ELEVENLABS_API_KEY=your_key_here")
        return
    
    # ========================================================================
    # STEP 2: Initialize the ElevenLabs Client
    # ========================================================================
    # The client object is your interface to all ElevenLabs API operations.
    # It handles authentication, request formatting, and response parsing.
    try:
        client = ElevenLabs(api_key=api_key)
        print("\n✅ Successfully initialized ElevenLabs client")
    except Exception as e:
        print(f"\n❌ Failed to initialize client: {e}")
        return
    
    # ========================================================================
    # CREDIT STATUS (check first, before using any credits)
    # ========================================================================
    # Try to get account/credit info so you know your balance before running TTS.
    # Some API keys don't have permission to read user info; then we only see
    # quota when a request fails.
    print("\n" + "-" * 60)
    print("Credit status")
    print("-" * 60)
    try:
        user_info = client.user.get()
        tier = getattr(user_info.subscription, "tier", None)
        used = getattr(user_info.subscription, "character_count", None)
        limit = getattr(user_info.subscription, "character_limit", None)
        if tier:
            print(f"   Tier: {tier}")
        if used is not None and limit is not None:
            remaining = max(0, limit - used)
            print(f"   Credits: {remaining} remaining ({used} used / {limit} limit)")
        elif used is not None:
            print(f"   Characters used: {used}")
        elif limit is not None:
            print(f"   Character limit: {limit}")
        if getattr(user_info.subscription, "can_extend_character_limit", None):
            print(f"   Can extend limit: yes")
        print("   ✅ Credit info retrieved")
    except Exception as e:
        error_msg = str(e)
        if "missing_permissions" in error_msg or "user_read" in error_msg:
            print("   ⚠️  This API key cannot read credit balance (no user_read permission).")
            print("   You will only see quota when a request fails.")
        else:
            print(f"   ❌ Could not get credit info: {e}")
    
    # ========================================================================
    # TEST 1: List available voices
    # ========================================================================
    # ElevenLabs provides multiple pre-made voices, each with a unique voice_id.
    # You can also create custom voices, but that requires additional permissions.
    # Each voice has different characteristics (gender, accent, tone, etc.)
    print("\n" + "-" * 60)
    print("Voices")
    print("-" * 60)
    try:
        # Search for available voices (returns all voices if no search term provided)
        voices_response = client.voices.search()
        voices_list = voices_response.voices
        print(f"✅ Found {len(voices_list)} available voices")
        
        # Display first few voices as examples
        # Each voice has: name, voice_id, description, and other metadata
        print("\nSample voices:")
        for i, voice in enumerate(voices_list[:5]):
            print(f"  {i+1}. {voice.name} (ID: {voice.voice_id})")
        
        # Select the first voice for testing
        # The voice_id is required for text-to-speech conversion
        test_voice_id = voices_list[0].voice_id if voices_list else None
        if test_voice_id:
            print(f"\n✅ Using voice: {voices_list[0].name} ({test_voice_id})")
    except Exception as e:
        # If voice listing fails (e.g., no permission), use a default voice
        # This is a public voice ID that should work with most API keys
        error_msg = str(e)
        if "missing_permissions" in error_msg or "voices_read" in error_msg:
            print(f"⚠️  API key doesn't have permission to list voices")
            print(f"   Using default voice ID for testing")
        else:
            print(f"❌ Failed to list voices: {e}")
        # Default voice: Rachel (a popular ElevenLabs voice)
        test_voice_id = "21m00Tcm4TlvDq8ikWAM"
        print(f"✅ Using default voice ID: {test_voice_id}")
    
    if not test_voice_id:
        print("\n⚠️  Cannot proceed with TTS tests without a voice ID")
        return
    
    # ========================================================================
    # TEST 2: Text-to-Speech conversion with credit tracking
    # ========================================================================
    # This is the core functionality: converting text into speech audio.
    # 
    # Key concepts:
    # - voice_id: Which voice to use (from Test 2)
    # - text: The text you want to convert to speech
    # - model_id: Which AI model to use (turbo = faster/cheaper, standard = higher quality)
    # - voice_settings: Fine-tune the voice characteristics
    # - Credits: Charged per character in the input text
    print("\n" + "-" * 60)
    print("Text-to-speech")
    print("-" * 60)
    
    # Test with different text lengths to understand credit consumption
    # Credits are charged per character, so longer text = more credits
    test_texts = [
        ("Short test", "Hello, this is a test."),
        ("Medium test", "This is a medium length text to test the ElevenLabs API. It contains multiple sentences and should give us a good idea of how credits are calculated."),
        ("Long test", "This is a longer text sample to understand credit consumption better. " * 3),
    ]
    credits_per_full_run = sum(len(text) for _, text in test_texts)
    total_chars_used = 0
    
    for test_name, text in test_texts:
        print(f"\n📝 {test_name}:")
        print(f"   Text: \"{text[:50]}...\" (length: {len(text)} characters)")
        
        try:
            # ================================================================
            # Convert text to speech
            # ================================================================
            # The convert() method returns a generator that yields audio chunks
            # This streaming approach allows you to start playing audio before
            # the entire conversion is complete (useful for long texts)
            audio_stream = client.text_to_speech.convert(
                voice_id=test_voice_id,      # Which voice to use
                text=text,                    # Text to convert
                model_id="eleven_turbo_v2_5", # Model: turbo = fast/cheap, standard = high quality
                voice_settings=VoiceSettings(
                    stability=0.5,            # How stable/consistent the voice is (0.0-1.0)
                    similarity_boost=0.75,    # How similar to original voice (0.0-1.0)
                    style=0.0,                # Style exaggeration (0.0-1.0)
                    use_speaker_boost=True    # Enhance speaker clarity
                )
            )
            
            # ================================================================
            # Collect audio data from the stream
            # ================================================================
            # The convert() method returns a generator, so we need to iterate
            # through it to collect all audio chunks into a single byte string
            audio_data = b""
            for chunk in audio_stream:
                audio_data += chunk
            
            # ================================================================
            # Calculate credit usage
            # ================================================================
            # Credits are charged based on the number of characters in your input text
            # Each character counts, including spaces and punctuation
            # Different models have different credit costs:
            # - Turbo/Flash: ~1 credit per character
            # - Standard: ~1 credit per character (but higher quality)
            char_count = len(text)
            total_chars_used += char_count
            
            print(f"   ✅ Audio generated: {len(audio_data)} bytes")
            print(f"   📊 Characters used: {char_count}")
            print(f"   💰 Estimated cost: ~${char_count * 0.00006:.4f} (at $0.06 per 1K chars)")
            
            # ================================================================
            # Save audio to file
            # ================================================================
            # The audio is in MP3 format and can be played with any media player
            # Files are named with "my" prefix to match gitignore pattern
            # Save in workshop root (01_audio/) so they sit alongside README
            output_file = os.path.join(workshop_root, f"my_{test_name.lower().replace(' ', '_')}.mp3")
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            print(f"   💾 Saved to: {output_file}")
            
        except ApiError as e:
            # Handle quota and other API errors with a clear message
            body = getattr(e, "body", None) or {}
            detail = body.get("detail", {}) if isinstance(body, dict) else {}
            if isinstance(detail, dict) and detail.get("status") == "quota_exceeded":
                msg = detail.get("message", str(e))
                print(f"   ⚠️  Quota exceeded: {msg}")
            else:
                print(f"   ❌ API error: {e}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # SUMMARY: Credit Usage and Cost Information
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Credits used this run: {total_chars_used}")
    print(f"Estimated cost this run: ~${total_chars_used * 0.00006:.4f}")
    print(f"\n💡 This script's three tests use {credits_per_full_run} credits per full run")
    print(f"   (short {len(test_texts[0][1])} + medium {len(test_texts[1][1])} + long {len(test_texts[2][1])} characters).")
    print(f"   With a 500 credit quota, running it twice uses ~{2 * credits_per_full_run} credits, so the second run can hit quota.")
    print(f"\n   - 1 credit ≈ 1 character (including spaces)")
    print(f"   - Turbo/Flash: ~$0.06 per 1,000 chars; standard: ~$0.12 per 1,000 chars")
    print(f"\n📈 Rough guide: 50 chars ≈ 50 credits, 200 chars ≈ 200 credits, 1000 chars ≈ 1000 credits")
    print("=" * 60)


if __name__ == "__main__":
    test_elevenlabs_api()
