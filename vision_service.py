"""
vision_service.py
Image understanding: OCR, captioning, and receipt analysis via multimodal
providers (OpenAI GPT-4o / Anthropic Claude vision). Images are passed as
base64 data URLs.
"""
from __future__ import annotations

import base64
import json

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


async def _openai_vision(image_bytes: bytes, prompt: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    data_url = f"data:image/png;base64,{_b64(image_bytes)}"
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


async def _anthropic_vision(image_bytes: bytes, prompt: str) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    resp = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(image_bytes)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


async def _run_vision(image_bytes: bytes, prompt: str, provider_name: str) -> str:
    if provider_name == "anthropic" and settings.anthropic_api_key:
        return await _anthropic_vision(image_bytes, prompt)
    if settings.openai_api_key:
        return await _openai_vision(image_bytes, prompt)
    raise RuntimeError("No vision-capable provider is configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY).")


async def ocr_image(image_bytes: bytes, provider_name: str = "openai") -> str:
    return await _run_vision(image_bytes, "Extract all text from this image verbatim.", provider_name)


async def caption_image(image_bytes: bytes, provider_name: str = "openai") -> str:
    return await _run_vision(image_bytes, "Describe this image in one detailed sentence.", provider_name)


async def analyze_receipt(image_bytes: bytes, provider_name: str = "openai") -> dict:
    prompt = (
        "Extract structured data from this receipt image. Respond ONLY with JSON: "
        '{"merchant": str, "date": str, "items": [{"name": str, "price": number}], "total": number}'
    )
    text = await _run_vision(image_bytes, prompt, provider_name)
    cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": text, "note": "Could not parse structured JSON from the model output."}
