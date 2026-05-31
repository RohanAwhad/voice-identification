# DeepSeek API — Deep Dive

> Docs: [api-docs.deepseek.com](https://api-docs.deepseek.com) | Python: OpenAI-compatible SDK

## TL;DR

DeepSeek API is OpenAI-compatible. Use the `openai` Python package with `base_url="https://api.deepseek.com"`. For voice assistant: use `deepseek-v4-flash`, disable thinking mode, keep responses short for TTS.

## Quick Reference

| Item | Value |
|------|-------|
| Base URL | `https://api.deepseek.com` |
| Chat Endpoint | `POST /chat/completions` |
| Auth | `Authorization: Bearer sk-xxx` |
| Python SDK | `pip install openai` (not official — just compatible) |
| Concurrent requests | 2,500 (v4-flash) / 500 (v4-pro) |

## Model IDs

| Model | Use case |
|-------|----------|
| `deepseek-v4-flash` | Fast, cheap. Voice assistant default. |
| `deepseek-v4-pro` | Slower, smarter. Overkill for voice. |

Note: Old IDs `deepseek-chat` / `deepseek-reasoner` are deprecated. Stop working 2026-07-24.

## Pricing (per 1M tokens)

| Model | Input (cache miss) | Output |
|-------|-------------------|--------|
| `deepseek-v4-flash` | $0.14 | $0.28 |
| `deepseek-v4-pro` | $0.435 (75% off promo) | $0.87 (75% off) |

Voice assistant estimate: ~100 input tokens + ~50 output tokens per interaction → ~$0.00005 per turn.

## Disabling Thinking Mode

**CRITICAL**: Thinking is ON by default for v4 models. Without disabling, the model generates hidden chain-of-thought before answering, adding latency and consuming tokens silently.

```python
extra_body={"thinking": {"type": "disabled"}}
```

When thinking is disabled:
- `temperature` and `top_p` now work (silently ignored when thinking enabled)
- No hidden CoT tokens consumed
- Faster response time

## Python Client (OpenAI-Compatible)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxxxxxxx",
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are Nova, a sassy voice assistant..."},
        {"role": "user", "content": "What time is it?"},
    ],
    temperature=0.7,
    max_tokens=256,
    extra_body={"thinking": {"type": "disabled"}},
)

text = response.choices[0].message.content
```

## Streaming (SSE)

```python
stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[...],
    stream=True,
    extra_body={"thinking": {"type": "disabled"}},
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Key Parameters

| Param | Notes |
|-------|-------|
| `temperature` | Only works when thinking disabled. 0.7 recommended for voice. |
| `top_p` | Only works when thinking disabled. |
| `max_tokens` | Always works. Set low for voice (~128-256) for faster TTS. |
| `stream` | `True` for SSE streaming. |
| `response_format` | `{"type": "text"}` (default) or `{"type": "json_object"}` |
| `frequency_penalty` | Deprecated — no effect. |
| `presence_penalty` | Deprecated — no effect. |

## System Prompt (Nova-Specific)

```
system:
You are Nova, a voice assistant with attitude. The person speaking to you has been identified as: {speaker_identity}.

Rules:
- If the speaker is "Rohan": Be helpful, warm, and responsive. Fulfill requests to the best of your ability. Keep responses concise and conversational — this will be spoken aloud.
- If the speaker is NOT "Rohan": You are sassy, witty, and dismissive. Reference their specific request to mock or joke about it, but NEVER actually fulfill it. The goal is to make them laugh at your audacity while leaving empty-handed. Keep responses under 2-3 sentences for voice.

user: {transcript}
```

## Gotchas

1. **Thinking is ON by default** — always explicitly `{"thinking": {"type": "disabled"}}`
2. **No official SDK** — use `openai` package with `base_url` swap
3. **Old model IDs deprecated** — `deepseek-chat`/`deepseek-reasoner` stop 2026-07-24
4. **temperature/top_p silently ignored** when thinking enabled — no error, just no effect
5. **HTTP 429 rate limit** — 2,500 concurrent for v4-flash. For single-user voice assistant: never an issue.
6. **Connection keep-alive**: 10 min max before close.

## Our Usage for Nova

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

def ask_nova(transcript: str, speaker: str) -> str:
    system_prompt = f"""You are Nova, a voice assistant with attitude. The speaker is: {speaker}.
If "Rohan": be helpful and concise.
If NOT "Rohan": be sassy, reference their request mockingly, NEVER fulfill it. 2-3 sentences max."""

    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0.7,
        max_tokens=256,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return resp.choices[0].message.content
```
