from __future__ import annotations

import subprocess

KOKORO_BIN = "/Users/rawhad/1_Projects/personal_projects/kokoro-rust/target/release/kokoro-tts"


def speak(text: str) -> None:
    subprocess.run([KOKORO_BIN, text], check=True)
