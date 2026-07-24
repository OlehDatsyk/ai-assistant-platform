"""
voice_service.py
Speech-to-text (Whisper) and text-to-speech (OpenAI TTS) for voice chat.
Streaming here means chunked delivery of synthesized audio bytes; full duplex
streaming audio would require a websocket client capable of playing PCM chunks,
which the /static/js/voice.js frontend module implements against these endpoints.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


def _client() -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for voice features (STT/TTS).")
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def transcribe_audio(file_bytes: bytes, filename: str = "audio.webm") -> str:
    """Speech-to-text via Whisper."""
    client = _client()
    file_tuple = (filename, file_bytes)
    transcript = await client.audio.transcriptions.create(
        model=settings.openai_stt_model, file=file_tuple
    )
    return transcript.text


async def synthesize_speech(text: str, voice: str = "alloy") -> bytes:
    """Text-to-speech; returns raw MP3 bytes."""
    client = _client()
    response = await client.audio.speech.create(
        model=settings.openai_tts_model, voice=voice, input=text
    )
    return response.read()
