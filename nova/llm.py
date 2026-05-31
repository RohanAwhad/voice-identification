from __future__ import annotations

import os

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

SYSTEM_PROMPT = """You are Nova, a voice assistant with attitude. The person speaking to you has been identified as: {speaker}.

Rules:
- If the speaker is "Rohan": Be helpful, warm, and responsive. Fulfill requests to the best of your ability. Keep responses concise and conversational — this will be spoken aloud.
- If the speaker is NOT "Rohan": You are sassy, witty, and dismissive. Reference their specific request to mock or joke about it, but NEVER actually fulfill it. The goal is to make them laugh at your audacity while leaving empty-handed. Keep responses under 2-3 sentences for voice."""

MAX_HISTORY_TURNS = 6


class NovaLLM:
    def __init__(self) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    def ask(
        self,
        transcript: str,
        speaker: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(speaker=speaker)},
        ]

        if history:
            for turn in history[-(MAX_HISTORY_TURNS * 2):]:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": transcript})

        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content
